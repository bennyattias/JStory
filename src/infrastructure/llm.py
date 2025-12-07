"""OpenAI LLM service implementation"""
from typing import List
from openai import AsyncOpenAI
from src.domain.repositories import LLMRepository
from src.domain.models import StoryChunk


class OpenAILLMService(LLMRepository):
    """OpenAI LLM service for generating responses"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI LLM service
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the model to use
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
    
    async def generate_response(
        self,
        query: str,
        context_chunks: List[StoryChunk]
    ) -> str:
        """
        Generate a response using the query and context chunks
        
        Args:
            query: User's query
            context_chunks: Relevant story chunks to use as context
            
        Returns:
            Generated response string
        """
        if not context_chunks:
            return "I couldn't find any relevant stories to answer your query."
        
        # Build context from chunks with citations
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.metadata.get('source', 'Unknown')
            title = chunk.metadata.get('title', 'Untitled')
            context_parts.append(
                f"[Source {i}: {title} from {source}]\n{chunk.content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        system_prompt = """You are a helpful assistant that answers questions about stories.
When answering, reference the specific sources provided. Be accurate and cite which source
each piece of information comes from using [Source 1], [Source 2], etc."""
        
        user_prompt = f"""Based on the following story excerpts, please answer this question: {query}

Story Excerpts:
{context}

Please provide a comprehensive answer with citations."""
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

