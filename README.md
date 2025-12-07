# JStory - Semantic Story Search with RAG

JStory is a Proof of Concept (POC) application that uses Retrieval-Augmented Generation (RAG) and Vector Databases to find stories using semantic search.

## Architecture

The project follows SOLID principles with a clean architecture:

```
JStory/
├── src/
│   ├── domain/              # Domain models and interfaces
│   ├── infrastructure/      # External dependencies (Vector DB, OpenAI, PDF)
│   ├── application/         # Use cases (business logic)
│   ├── presentation/        # Web API and UI
│   └── config/              # Configuration and DI
├── data/                    # PDF story files
├── tests/                   # Unit and integration tests
└── requirements.txt
```

## Features

- Load PDF documents containing stories
- Split documents into semantic chunks
- Generate embeddings using OpenAI
- Store embeddings in ChromaDB vector database
- Semantic search to find relevant story chunks
- Generate responses using ChatGPT with citations

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
CHROMA_DB_PATH=./chroma_db
OPENAI_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

3. **Place story files in the `data/` directory**
   - **PDF files**: Stories should be separated by clear boundaries (e.g., "Story 1:", "Chapter 1:", etc.)
   - **TXT files**: Stories should be separated by empty lines (double newlines)
   - At least 3 books with 200+ total stories for the POC

4. **Run the application:**
```bash
uvicorn src.presentation.api:app --reload
```

5. **Access the UI at `http://localhost:8000`**

## Usage

### Via Web UI:
1. Upload PDF files using the upload interface
2. Enter a query in the search box
3. View the generated response with citations

### Via API:
1. **Ingest stories:** POST to `/api/ingest` with PDF or TXT files
   ```bash
   curl -X POST "http://localhost:8000/api/ingest" \
     -F "files=@data/story1.txt" \
     -F "files=@data/story2.pdf"
   ```

2. **Search stories:** POST to `/api/search` with a query
   ```bash
   curl -X POST "http://localhost:8000/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "Tell me about stories with dragons"}'
   ```

### Via Command Line:
```bash
# For TXT files (stories separated by empty lines)
python scripts/ingest_stories.py data/story1.txt data/story2.txt

# For PDF files
python scripts/ingest_stories.py data/story1.pdf data/story2.pdf
```

## Architecture

The project follows **SOLID principles** with clean architecture:

- **Domain Layer**: Core business models and interfaces (repositories, services)
- **Infrastructure Layer**: External dependencies (ChromaDB, OpenAI, PDF processing)
- **Application Layer**: Use cases (business logic)
- **Presentation Layer**: FastAPI web API and UI

## Testing

```bash
pytest tests/
```

## How It Works

1. **Document Loading**: PDFs are loaded and text is extracted
2. **Story Chunking**: Text is split by story boundaries (not arbitrary sizes)
3. **Embedding Generation**: Each story chunk is converted to embeddings using OpenAI
4. **Vector Storage**: Embeddings are stored in ChromaDB for semantic search
5. **Query Processing**: User queries are embedded and matched against stored chunks
6. **Response Generation**: ChatGPT generates responses using relevant chunks as context with citations

