"""FastAPI application"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler
)
from typing import List
import os
import logging
from pathlib import Path

from src.config.dependencies import (
    get_vector_store_repository,
    get_embedding_repository,
    get_llm_repository,
    get_document_loader_service,
    get_text_chunking_service,
    get_settings
)
from src.application.use_cases import (
    IngestStoriesUseCase,
    SearchStoriesUseCase,
    GenerateResponseUseCase
)
from src.domain.repositories import (
    VectorStoreRepository,
    EmbeddingRepository,
    LLMRepository
)
from src.domain.services import DocumentLoaderService, TextChunkingService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="JStory API", version="0.1.0")


# Global exception handler to ensure all errors return JSON
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return JSON"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors and return JSON"""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body}
    )

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Preload all stories from the data directory on startup"""
    settings = get_settings()
    data_dir = Path(settings.data_dir)
    
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist. Skipping story preload.")
        return
    
    # Get all story files (PDF and TXT)
    story_files = []
    for ext in ['*.pdf', '*.txt']:
        story_files.extend(data_dir.glob(ext))
        story_files.extend(data_dir.glob(ext.upper()))
    
    if not story_files:
        logger.info(f"No story files found in {data_dir}. Stories will need to be uploaded manually.")
        return
    
    logger.info(f"Found {len(story_files)} story file(s) to preload...")
    
    # Initialize use case for ingestion
    ingest_use_case = get_ingest_use_case()
    
    # Process each file
    total_ingested = 0
    for file_path in story_files:
        try:
            logger.info(f"Preloading stories from: {file_path.name}")
            result = await ingest_use_case.execute(str(file_path))
            total_ingested += result.get('stories_ingested', 0)
            logger.info(f"✓ Successfully preloaded {result.get('stories_ingested', 0)} chunks from {file_path.name}")
        except Exception as e:
            logger.error(f"✗ Error preloading {file_path.name}: {str(e)}")
            # Continue with other files even if one fails
    
    logger.info(f"Startup complete: Preloaded {total_ingested} total chunks from {len(story_files)} file(s)")


# Dependency injection for use cases
def get_ingest_use_case() -> IngestStoriesUseCase:
    """Get ingest use case with dependencies"""
    return IngestStoriesUseCase(
        document_loader=get_document_loader_service(),
        text_chunker=get_text_chunking_service(),
        embedding_service=get_embedding_repository(),
        vector_store=get_vector_store_repository()
    )


def get_search_use_case() -> SearchStoriesUseCase:
    """Get search use case with dependencies"""
    return SearchStoriesUseCase(
        embedding_service=get_embedding_repository(),
        vector_store=get_vector_store_repository()
    )


def get_generate_use_case() -> GenerateResponseUseCase:
    """Get generate response use case with dependencies"""
    return GenerateResponseUseCase(
        search_use_case=get_search_use_case(),
        llm_service=get_llm_repository()
    )


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>JStory - Semantic Story Search</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .content {
                padding: 40px;
            }
            .section {
                margin-bottom: 40px;
            }
            .section h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                background: #f8f9fa;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #e9ecef;
                border-color: #764ba2;
            }
            input[type="file"] {
                margin: 20px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                width: 100%;
                max-width: 400px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 1.1em;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .search-box {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .search-box input {
                flex: 1;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1.1em;
            }
            .search-box input:focus {
                outline: none;
                border-color: #667eea;
            }
            .results {
                margin-top: 30px;
            }
            .result-item {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .result-item h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .result-item .score {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 10px;
            }
            .result-item .content {
                color: #333;
                line-height: 1.6;
            }
            .response {
                background: #e7f3ff;
                border-left: 4px solid #2196F3;
                padding: 20px;
                margin-top: 20px;
                border-radius: 5px;
            }
            .response h3 {
                color: #2196F3;
                margin-bottom: 10px;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            .error {
                background: #ffebee;
                border-left: 4px solid #f44336;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                color: #c62828;
            }
            .success {
                background: #e8f5e9;
                border-left: 4px solid #4caf50;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                color: #2e7d32;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📚 JStory</h1>
                <p>Semantic Story Search with RAG</p>
            </div>
            <div class="content">
                <div class="section">
                    <h2>📚 Preloaded Stories</h2>
                    <div class="upload-area" style="background: #e8f5e9; border-color: #4caf50;">
                        <p>Stories have been automatically preloaded from the data directory.<br><small>You can start searching immediately!</small></p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Search Stories</h2>
                    <div class="search-box">
                        <input type="text" id="queryInput" placeholder="Enter your question about the stories..." onkeypress="handleKeyPress(event)">
                        <button onclick="searchStories()">Search</button>
                    </div>
                    <div id="searchResults"></div>
                </div>
            </div>
        </div>
        
        <script>
            async function searchStories() {
                const query = document.getElementById('queryInput').value.trim();
                const resultsDiv = document.getElementById('searchResults');
                
                if (!query) {
                    resultsDiv.innerHTML = '<div class="error">Please enter a search query</div>';
                    return;
                }
                
                resultsDiv.innerHTML = '<div class="loading">Searching stories and generating response...</div>';
                
                try {
                    const response = await fetch('/api/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query })
                    });
                    
                    let result;
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        result = await response.json();
                    } else {
                        const text = await response.text();
                        resultsDiv.innerHTML = `<div class="error">Error: Server returned non-JSON response: ${text.substring(0, 200)}</div>`;
                        return;
                    }
                    
                    if (response.ok) {
                        let html = '';
                        
                        // Show generated response
                        if (result.response) {
                            html += '<div class="response">';
                            html += '<h3>🤖 Generated Response</h3>';
                            html += '<p>' + result.response.replace(/\\n/g, '<br>') + '</p>';
                            html += '</div>';
                        }
                        
                        resultsDiv.innerHTML = html;
                    } else {
                        resultsDiv.innerHTML = `<div class="error">Error: ${result.detail || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    searchStories();
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/api/ingest")
async def ingest_stories(
    files: List[UploadFile] = File(...),
    use_case: IngestStoriesUseCase = Depends(get_ingest_use_case)
):
    """Ingest PDF or TXT files into the vector database"""
    results = []
    
    for file in files:
        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if file_ext not in ['pdf', 'txt']:
            raise HTTPException(
                status_code=400, 
                detail=f"{file.filename} is not a supported file type. Supported: .pdf, .txt"
            )
        
        # Save file temporarily
        file_path = DATA_DIR / file.filename
        # For text files, we need to handle encoding properly
        if file_ext == 'txt':
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
        
        try:
            # Ingest the file
            result = await use_case.execute(str(file_path))
            results.append(result)
        except Exception as e:
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")
    
    return results


@app.post("/api/search")
async def search_stories(
    query: dict,
    use_case: GenerateResponseUseCase = Depends(get_generate_use_case)
):
    """Search stories and generate response"""
    if not query.get('query'):
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        result = await use_case.execute(query['query'], top_k=3)
        
        return {
            'query': result.query,
            'response': result.response,
            'citations': [
                {
                    'content': chunk.content,
                    'metadata': chunk.metadata
                }
                for chunk in result.citations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching stories: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

