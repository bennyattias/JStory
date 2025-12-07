"""Domain models for stories and chunks"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class StoryChunk:
    """Represents a chunk of text from a story"""
    id: str
    content: str
    metadata: dict
    embedding: Optional[list[float]] = None
    
    def __post_init__(self):
        """Validate chunk data"""
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")


@dataclass
class StoryDocument:
    """Represents a complete story document"""
    source: str  # File path or source identifier
    title: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None
    chunks: list[StoryChunk] = None
    
    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    chunk: StoryChunk
    score: float
    rank: int


@dataclass
class GeneratedResponse:
    """Represents a generated response with citations"""
    response: str
    citations: list[StoryChunk]
    query: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

