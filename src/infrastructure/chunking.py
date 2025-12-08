"""Text chunking implementation - splits by story boundaries"""
import re
from typing import List
from src.domain.services import TextChunkingService


class LangChainTextChunker(TextChunkingService):
    """Text chunking service that splits text by story boundaries"""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum size of each chunk (used if story is too long)
            chunk_overlap: Number of characters to overlap between chunks (if needed)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Common patterns for story boundaries
        self.story_patterns = [
            r'^Story\s+\d+[:\-]',  # "Story 1:", "Story 1-", etc.
            r'^Chapter\s+\d+[:\-]',  # "Chapter 1:", etc.
            r'^STORY\s+\d+[:\-]',  # "STORY 1:"
            r'^\d+\.\s+[A-Z]',  # "1. Title" (numbered story)
            r'^[A-Z][a-z]+\s+\d+[:\-]',  # "Tale 1:", "Fable 2:", etc.
            r'\n\n[A-Z][^.!?]*\n\n',  # Double newline followed by title-like text
        ]
    
    def _find_story_boundaries(self, text: str) -> List[int]:
        """
        Find story boundary positions in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of positions where stories start
        """
        boundaries = [0]  # Start of text is always a boundary
        
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this line matches a story boundary pattern
            is_boundary = False
            for pattern in self.story_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_boundary = True
                    break
            
            # Also check for double newlines (common story separator)
            if i > 0 and not line_stripped and i < len(lines) - 1:
                # Empty line - check if next line looks like a story start
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and len(next_line) < 100 and not next_line.endswith('.'):
                        # Could be a title
                        is_boundary = True
            
            if is_boundary and current_pos > 0:
                boundaries.append(current_pos)
            
            current_pos += len(line) + 1  # +1 for newline
        
        return boundaries
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ) -> List[str]:
        """
        Split text into chunks by story boundaries.
        Prioritizes empty line separation (double newlines) for .txt files.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size (for very long stories)
            chunk_overlap: Override default chunk overlap (if needed)
            
        Returns:
            List of text chunks, each representing a complete story
        """
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        # First, try splitting by empty lines (double newlines) - most common format
        # This handles the case where stories are separated by empty lines
        if '\n\n' in text or text.count('\n') > text.count(' ') / 10:
            # Split by double newlines (empty lines)
            potential_chunks = text.split('\n\n')
            
            # Filter out empty chunks and strip whitespace
            chunks = [chunk.strip() for chunk in potential_chunks if chunk.strip()]
            
            # If we got multiple chunks from empty line splitting, use those
            if len(chunks) > 1:
                # Handle very long stories
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > chunk_size * 3:  # Only split if 3x the chunk size
                        # Use recursive splitting for very long stories
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            length_function=len,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        sub_chunks = splitter.split_text(chunk)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)
                return final_chunks
        
        # Fallback to pattern-based boundary detection
        boundaries = self._find_story_boundaries(text)
        
        # If no boundaries found, treat entire text as one story
        if len(boundaries) <= 1:
            return [text.strip()] if text.strip() else []
        
        # Extract stories based on boundaries
        chunks = []
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            
            story_text = text[start:end].strip()
            
            if not story_text:
                continue
            
            # If a story is extremely long, we might need to split it further
            # But try to keep stories intact as much as possible
            if len(story_text) > chunk_size * 3:  # Only split if 3x the chunk size
                # Use recursive splitting for very long stories
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                sub_chunks = splitter.split_text(story_text)
                chunks.extend(sub_chunks)
            else:
                chunks.append(story_text)
        
        return chunks

