"""Tests for text chunking service"""
import pytest
from src.infrastructure.chunking import LangChainTextChunker


def test_chunk_by_story_boundaries():
    """Test that text is chunked by story boundaries"""
    chunker = LangChainTextChunker()
    
    text = """Story 1: The First Tale
This is the first story content. It has multiple sentences.
And more content here.

Story 2: The Second Tale
This is the second story. It also has content.
More sentences here.

Story 3: The Third Tale
Final story with its own content."""
    
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 3
    assert "Story 1" in chunks[0]
    assert "Story 2" in chunks[1]
    assert "Story 3" in chunks[2]
    assert "Story 1" not in chunks[1]  # Ensure no overlap


def test_chunk_with_numbered_stories():
    """Test chunking with numbered story format"""
    chunker = LangChainTextChunker()
    
    text = """1. The Beginning
First story content here.

2. The Middle
Second story content.

3. The End
Third story content."""
    
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) >= 2  # Should find at least 2 stories


def test_empty_text():
    """Test handling of empty text"""
    chunker = LangChainTextChunker()
    
    chunks = chunker.chunk_text("")
    assert chunks == []


def test_single_story():
    """Test handling of text with no story boundaries"""
    chunker = LangChainTextChunker()
    
    text = "This is a single story without any boundaries. It just continues."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_chunk_by_empty_lines():
    """Test that text is chunked by empty lines (double newlines)"""
    chunker = LangChainTextChunker()
    
    text = """First story content here.
It has multiple lines.
And more content.

Second story starts here.
This is the second story.
It also has multiple lines.

Third story content.
Final story in the text."""
    
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 3
    assert "First story" in chunks[0]
    assert "Second story" in chunks[1]
    assert "Third story" in chunks[2]
    assert "First story" not in chunks[1]  # Ensure no overlap
    assert "Second story" not in chunks[2]  # Ensure no overlap

