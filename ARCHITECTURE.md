# JStory Architecture

## Overview

JStory is built following **SOLID principles** and **Clean Architecture** patterns, ensuring:
- **Loose coupling** between components
- **High testability** through dependency injection
- **Separation of concerns** across layers
- **Single Responsibility** for each component

## Architecture Layers

```
┌─────────────────────────────────────────┐
│      Presentation Layer (API/UI)        │
│  - FastAPI endpoints                   │
│  - HTML/JavaScript frontend            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Application Layer (Use Cases)      │
│  - IngestStoriesUseCase                 │
│  - SearchStoriesUseCase                 │
│  - GenerateResponseUseCase              │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      Domain Layer (Core Business)       │
│  - Models (StoryChunk, StoryDocument)   │
│  - Repository Interfaces                │
│  - Service Interfaces                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│   Infrastructure Layer (Implementations)│
│  - ChromaDBVectorStore                  │
│  - OpenAIEmbeddingService               │
│  - OpenAILLMService                     │
│  - PDFDocumentLoader                    │
│  - LangChainTextChunker                 │
└─────────────────────────────────────────┘
```

## Component Responsibilities

### Domain Layer
- **Models**: Core business entities (`StoryChunk`, `StoryDocument`, `SearchResult`)
- **Repositories**: Interfaces defining data access contracts
- **Services**: Interfaces defining service contracts

### Infrastructure Layer
- **VectorStoreRepository**: ChromaDB implementation for vector storage
- **EmbeddingRepository**: OpenAI embeddings implementation
- **LLMRepository**: OpenAI ChatGPT implementation
- **DocumentLoaderService**: PDF loading implementation
- **TextChunkingService**: Story-based chunking implementation

### Application Layer
- **IngestStoriesUseCase**: Orchestrates story ingestion workflow
- **SearchStoriesUseCase**: Handles semantic search
- **GenerateResponseUseCase**: Combines search + LLM generation

### Presentation Layer
- **FastAPI App**: RESTful API endpoints
- **Web UI**: Single-page application for user interaction

## Data Flow

### Story Ingestion Flow
```
PDF File → PDFDocumentLoader → StoryDocument
    ↓
StoryDocument.content → LangChainTextChunker → [Story Chunks]
    ↓
[Story Chunks] → OpenAIEmbeddingService → [Embeddings]
    ↓
[StoryChunks with Embeddings] → ChromaDBVectorStore → Stored
```

### Query Flow
```
User Query → OpenAIEmbeddingService → Query Embedding
    ↓
Query Embedding → ChromaDBVectorStore.search → [Relevant Chunks]
    ↓
[Relevant Chunks] + Query → OpenAILLMService → Generated Response
    ↓
Response + Citations → User
```

## Design Patterns

1. **Dependency Injection**: All dependencies injected via constructor
2. **Repository Pattern**: Abstract data access behind interfaces
3. **Use Case Pattern**: Business logic encapsulated in use cases
4. **Strategy Pattern**: Different implementations for services/repositories

## Testing Strategy

- **Unit Tests**: Test each component in isolation with mocks
- **Integration Tests**: Test use cases with real dependencies
- **Mock External Services**: OpenAI and ChromaDB mocked in tests

## Key Features

- ✅ Story-based chunking (not arbitrary text splitting)
- ✅ OpenAI embeddings for semantic search
- ✅ ChromaDB for vector storage
- ✅ ChatGPT for response generation with citations
- ✅ Clean architecture with SOLID principles
- ✅ Fully testable with dependency injection
- ✅ Professional code structure

