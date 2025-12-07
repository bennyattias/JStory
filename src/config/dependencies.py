"""Dependency injection container"""
from functools import lru_cache
from src.config.settings import Settings, settings
from src.domain.repositories import (
    VectorStoreRepository,
    EmbeddingRepository,
    LLMRepository
)
from src.domain.services import (
    DocumentLoaderService,
    TextChunkingService
)
from src.infrastructure.vector_store import ChromaDBVectorStore
from src.infrastructure.embeddings import OpenAIEmbeddingService
from src.infrastructure.llm import OpenAILLMService
from src.infrastructure.document_loader import PDFDocumentLoader
from src.infrastructure.chunking import LangChainTextChunker


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (singleton)"""
    return settings


def get_vector_store_repository() -> VectorStoreRepository:
    """Get vector store repository instance"""
    config = get_settings()
    return ChromaDBVectorStore(
        db_path=config.chroma_db_path,
        collection_name=config.collection_name
    )


def get_embedding_repository() -> EmbeddingRepository:
    """Get embedding repository instance"""
    config = get_settings()
    return OpenAIEmbeddingService(
        api_key=config.openai_api_key,
        model_name=config.embedding_model
    )


def get_llm_repository() -> LLMRepository:
    """Get LLM repository instance"""
    config = get_settings()
    return OpenAILLMService(
        api_key=config.openai_api_key,
        model_name=config.openai_model
    )


def get_document_loader_service() -> DocumentLoaderService:
    """Get document loader service instance"""
    return PDFDocumentLoader()


def get_text_chunking_service() -> TextChunkingService:
    """Get text chunking service instance"""
    config = get_settings()
    return LangChainTextChunker(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )

