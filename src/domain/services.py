"""Domain service interfaces"""
from abc import ABC, abstractmethod
from typing import List
from src.domain.models import StoryDocument, StoryChunk


class DocumentLoaderService(ABC):
    """Interface for loading documents"""
    
    @abstractmethod
    async def load_pdf(self, file_path: str) -> StoryDocument:
        """Load a PDF file and return a StoryDocument"""
        pass


class TextChunkingService(ABC):
    """Interface for text chunking"""
    
    @abstractmethod
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List[str]:
        """Split text into chunks"""
        pass

