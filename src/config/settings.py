"""Application settings using Pydantic"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration"""
    openai_api_key: str
    chroma_db_path: str = "./chroma_db"
    openai_model: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000  # Legacy parameter, kept for compatibility
    chunk_overlap: int = 200  # Legacy parameter, kept for compatibility
    min_words: int = 1  # Minimum words per chunk
    max_words: int = 300  # Maximum words per chunk
    overlap_words: int = 50  # Number of words to overlap between chunks
    collection_name: str = "story_chunks"
    data_dir: str = "./data"  # Directory containing preloaded story files
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()

