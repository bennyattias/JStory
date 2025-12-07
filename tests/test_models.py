"""Tests for domain models"""
import pytest
from src.domain.models import StoryChunk, StoryDocument, SearchResult


def test_story_chunk_creation():
    """Test creating a valid StoryChunk"""
    chunk = StoryChunk(
        id="test-id",
        content="Test content",
        metadata={"source": "test.pdf"}
    )
    
    assert chunk.id == "test-id"
    assert chunk.content == "Test content"
    assert chunk.metadata["source"] == "test.pdf"
    assert chunk.embedding is None


def test_story_chunk_validation():
    """Test that StoryChunk validates required fields"""
    with pytest.raises(ValueError):
        StoryChunk(id="", content="test", metadata={})
    
    with pytest.raises(ValueError):
        StoryChunk(id="test", content="", metadata={})


def test_story_document_creation():
    """Test creating a StoryDocument"""
    doc = StoryDocument(
        source="test.pdf",
        title="Test Book",
        content="Story content"
    )
    
    assert doc.source == "test.pdf"
    assert doc.title == "Test Book"
    assert doc.content == "Story content"
    assert doc.chunks == []


def test_search_result_creation():
    """Test creating a SearchResult"""
    chunk = StoryChunk(
        id="test-id",
        content="Test content",
        metadata={}
    )
    
    result = SearchResult(chunk=chunk, score=0.95, rank=1)
    
    assert result.chunk == chunk
    assert result.score == 0.95
    assert result.rank == 1

