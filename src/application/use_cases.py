"""Application use cases"""
import uuid
from typing import List
from pathlib import Path
from src.domain.models import StoryDocument, StoryChunk, SearchResult, GeneratedResponse
from src.domain.repositories import (
    VectorStoreRepository,
    EmbeddingRepository,
    LLMRepository
)
from src.domain.services import DocumentLoaderService, TextChunkingService


class IngestStoriesUseCase:
    """Use case for ingesting stories into the vector database"""
    
    def __init__(
        self,
        document_loader: DocumentLoaderService,
        text_chunker: TextChunkingService,
        embedding_service: EmbeddingRepository,
        vector_store: VectorStoreRepository
    ):
        """
        Initialize the ingest use case
        
        Args:
            document_loader: Service for loading documents
            text_chunker: Service for chunking text
            embedding_service: Service for generating embeddings
            vector_store: Repository for storing chunks
        """
        self.document_loader = document_loader
        self.text_chunker = text_chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    async def execute(self, file_path: str) -> dict:
        """
        Execute the story ingestion process
        
        Args:
            file_path: Path to the PDF file to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        # Load document
        document = await self.document_loader.load_pdf(file_path)
        
        if not document.content:
            raise ValueError(f"No content extracted from {file_path}")
        
        # Chunk text by story
        text_chunks = self.text_chunker.chunk_text(document.content)
        
        if not text_chunks:
            raise ValueError(f"No stories found in {file_path}")
        
        # Generate embeddings for all chunks
        embeddings = await self.embedding_service.generate_embeddings_batch(text_chunks)
        
        # Create StoryChunk objects
        story_chunks = []
        for i, (chunk_text, embedding) in enumerate(zip(text_chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk = StoryChunk(
                id=chunk_id,
                content=chunk_text,
                metadata={
                    'source': document.source,
                    'title': document.title or Path(file_path).stem,
                    'author': document.author,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                },
                embedding=embedding
            )
            story_chunks.append(chunk)
        
        # Store in vector database
        await self.vector_store.add_chunks(story_chunks)
        
        return {
            'file_path': file_path,
            'title': document.title,
            'stories_ingested': len(story_chunks),
            'chunks_created': len(story_chunks)
        }


class SearchStoriesUseCase:
    """Use case for searching stories using semantic search"""
    
    def __init__(
        self,
        embedding_service: EmbeddingRepository,
        vector_store: VectorStoreRepository
    ):
        """
        Initialize the search use case
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Repository for searching chunks
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    async def execute(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Execute the semantic search
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects
        """
        if not query or not query.strip():
            return []
        
        # Generate embedding for query
        query_embedding = await self.embedding_service.generate_embedding(query)
        
        # Search vector store
        results = await self.vector_store.search_similar(query_embedding, top_k=top_k)
        
        return results


class GenerateResponseUseCase:
    """Use case for generating responses with citations"""
    
    def __init__(
        self,
        search_use_case: SearchStoriesUseCase,
        llm_service: LLMRepository
    ):
        """
        Initialize the response generation use case
        
        Args:
            search_use_case: Use case for searching stories
            llm_service: Service for generating LLM responses
        """
        self.search_use_case = search_use_case
        self.llm_service = llm_service
    
    async def execute(
        self, 
        query: str, 
        top_k: int = 3
    ) -> GeneratedResponse:
        """
        Execute the response generation process
        
        Args:
            query: User's query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            GeneratedResponse with answer and citations
        """
        # Search for relevant chunks
        search_results = await self.search_use_case.execute(query, top_k=top_k)
        
        if not search_results:
            return GeneratedResponse(
                response="I couldn't find any relevant stories to answer your query.",
                citations=[],
                query=query
            )
        
        # Extract chunks from search results
        context_chunks = [result.chunk for result in search_results]
        
        # Generate response using LLM
        response_text = await self.llm_service.generate_response(
            query=query,
            context_chunks=context_chunks
        )
        
        return GeneratedResponse(
            response=response_text,
            citations=context_chunks,
            query=query
        )

