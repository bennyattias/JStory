"""OpenAI embedding service implementation"""
from typing import List
from openai import AsyncOpenAI
from src.domain.repositories import EmbeddingRepository


class OpenAIEmbeddingService(EmbeddingRepository):
    """OpenAI embedding service"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embedding service
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the embedding model to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        
        return response.data[0].embedding
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=valid_texts
        )
        
        # Return embeddings in the same order as input
        embeddings_dict = {
            item.index: item.embedding 
            for item in response.data
        }
        
        embeddings = []
        valid_idx = 0
        for text in texts:
            if text and text.strip():
                embeddings.append(embeddings_dict[valid_idx])
                valid_idx += 1
            else:
                # For empty texts, return empty embedding (shouldn't happen in practice)
                embeddings.append([])
        
        return embeddings

