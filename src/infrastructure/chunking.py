"""Text chunking implementation - splits text into word-based chunks with overlap"""
import re
from typing import List
from src.domain.services import TextChunkingService


class LangChainTextChunker(TextChunkingService):
    """Text chunking service that splits text into chunks of 1-300 words with overlap"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        min_words: int = 1,
        max_words: int = 300,
        overlap_words: int = 50
    ):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Legacy parameter (kept for compatibility, not used)
            chunk_overlap: Legacy parameter (kept for compatibility, not used)
            min_words: Minimum words per chunk (default: 1)
            max_words: Maximum words per chunk (default: 300)
            overlap_words: Number of words to overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
    
    def _count_words(self, text: str) -> int:
        """Count the number of words in text"""
        if not text or not text.strip():
            return 0
        # Split by whitespace and filter out empty strings
        words = [w for w in text.split() if w.strip()]
        return len(words)
    
    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words while preserving word boundaries"""
        if not text or not text.strip():
            return []
        # Split by whitespace and filter out empty strings
        return [w for w in text.split() if w.strip()]
    
    def _words_to_text(self, words: List[str], start_idx: int, end_idx: int) -> str:
        """Convert a slice of words back to text"""
        return ' '.join(words[start_idx:end_idx])
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks of 1-300 words with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Ignored (kept for interface compatibility)
            chunk_overlap: Ignored (kept for interface compatibility)
            
        Returns:
            List of text chunks, each containing 1-300 words with overlap
        """
        if not text or not text.strip():
            return []
        
        # Split text into words
        words = self._split_into_words(text)
        total_words = len(words)
        
        if total_words == 0:
            return []
        
        # If text is shorter than max_words, return as single chunk
        if total_words <= self.max_words:
            return [text.strip()]
        
        chunks = []
        start_idx = 0
        
        while start_idx < total_words:
            # Calculate end index for this chunk
            end_idx = min(start_idx + self.max_words, total_words)
            
            # Extract chunk
            chunk_words = words[start_idx:end_idx]
            chunk_text = self._words_to_text(words, start_idx, end_idx)
            
            # Only add chunk if it meets minimum word requirement
            if len(chunk_words) >= self.min_words:
                chunks.append(chunk_text.strip())
            
            # If we've reached the end, break
            if end_idx >= total_words:
                break
            
            # Move start index forward, accounting for overlap
            # Next chunk starts at current end minus overlap
            start_idx = max(start_idx + 1, end_idx - self.overlap_words)
            
            # Safety check to prevent infinite loops
            if start_idx >= end_idx:
                start_idx = end_idx
        
        return chunks if chunks else [text.strip()]

