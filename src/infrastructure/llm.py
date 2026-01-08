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
            return "No relevant stories found."
        
        # Build context from chunks without source labels
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(chunk.content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a story retriever. When asked for a story or type of story, 
return the story content directly without any commentary, explanation, or personal take. 
Do not address the user or use phrases like "I", "you", "based on", or "the story says". 
Simply present the story as it is. Write in third person narrative style only."""
        
        user_prompt = f"""Query: {query}

Story Excerpts:
{context}

Return the story content directly without commentary or addressing the user."""
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

