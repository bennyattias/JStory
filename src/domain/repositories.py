"""Repository interfaces following Dependency Inversion Principle"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.models import StoryChunk, SearchResult


class VectorStoreRepository(ABC):
    """Interface for vector store operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[StoryChunk]) -> None:
        """Add story chunks to the vector store"""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query_embedding: List[float], 
        top_k: int = 3
    ) -> List[SearchResult]:
        """Search for similar chunks using embedding similarity"""
        pass
    
    @abstractmethod
    async def get_chunk_by_id(self, chunk_id: str) -> Optional[StoryChunk]:
        """Retrieve a chunk by its ID"""
        pass
    
    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all chunks from the store"""
        pass


class EmbeddingRepository(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass


class LLMRepository(ABC):
    """Interface for LLM operations"""
    
    @abstractmethod
    async def generate_response(
        self,
        query: str,
        context_chunks: List[StoryChunk]
    ) -> str:
        """Generate a response using the query and context chunks"""
        pass

