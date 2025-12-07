"""ChromaDB vector store implementation"""
import uuid
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.domain.repositories import VectorStoreRepository
from src.domain.models import StoryChunk, SearchResult


class ChromaDBVectorStore(VectorStoreRepository):
    """ChromaDB implementation of VectorStoreRepository"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "story_chunks"):
        """
        Initialize ChromaDB vector store
        
        Args:
            db_path: Path to store ChromaDB database
            collection_name: Name of the collection to use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy initialization of ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client
    
    @property
    def collection(self):
        """Lazy initialization of ChromaDB collection"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(name=self.collection_name)
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Story chunks for semantic search"}
                )
        return self._collection
    
    async def add_chunks(self, chunks: List[StoryChunk]) -> None:
        """
        Add story chunks to the vector store
        
        Args:
            chunks: List of StoryChunk objects to add
        """
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if not chunk.embedding:
                raise ValueError(f"Chunk {chunk.id} must have an embedding before adding to vector store")
            
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            
            # Ensure metadata is serializable
            metadata = {
                **chunk.metadata,
                'chunk_id': chunk.id
            }
            # ChromaDB requires string values for metadata
            metadatas.append({
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in metadata.items()
            })
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        Search for similar chunks using embedding similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        if not query_embedding:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        search_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results.get('distances') else None
                # Convert distance to similarity score (ChromaDB uses distance, lower is better)
                score = 1.0 - distance if distance is not None else 1.0
                
                # Reconstruct chunk from results
                content = results['documents'][0][i] if results.get('documents') else ""
                metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                
                chunk = StoryChunk(
                    id=chunk_id,
                    content=content,
                    metadata=metadata
                )
                
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=i + 1
                ))
        
        return search_results
    
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[StoryChunk]:
        """
        Retrieve a chunk by its ID
        
        Args:
            chunk_id: ID of the chunk to retrieve
            
        Returns:
            StoryChunk if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[chunk_id])
            
            if not results['ids'] or len(results['ids']) == 0:
                return None
            
            idx = 0
            content = results['documents'][idx] if results.get('documents') else ""
            metadata = results['metadatas'][idx] if results.get('metadatas') else {}
            
            return StoryChunk(
                id=chunk_id,
                content=content,
                metadata=metadata
            )
        except Exception:
            return None
    
    async def clear_all(self) -> None:
        """Clear all chunks from the store"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None  # Reset collection so it gets recreated
        except Exception:
            pass  # Collection might not exist

