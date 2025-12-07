"""FastAPI application"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import os
from pathlib import Path

from src.config.dependencies import (
    get_vector_store_repository,
    get_embedding_repository,
    get_llm_repository,
    get_document_loader_service,
    get_text_chunking_service
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

app = FastAPI(title="JStory API", version="0.1.0")

# Create data directory if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


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
                    <h2>📤 Upload Stories</h2>
                    <div class="upload-area">
                        <p>Upload PDF or TXT files containing stories<br><small>(TXT files: stories separated by empty lines)</small></p>
                        <input type="file" id="fileInput" accept=".pdf,.txt" multiple>
                        <button onclick="uploadFiles()">Upload & Ingest</button>
                        <div id="uploadStatus"></div>
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
            async function uploadFiles() {
                const input = document.getElementById('fileInput');
                const statusDiv = document.getElementById('uploadStatus');
                
                if (!input.files || input.files.length === 0) {
                    statusDiv.innerHTML = '<div class="error">Please select at least one PDF file</div>';
                    return;
                }
                
                statusDiv.innerHTML = '<div class="loading">Uploading and processing files...</div>';
                
                const formData = new FormData();
                for (let file of input.files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/api/ingest', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let html = '<div class="success">';
                        result.forEach(r => {
                            html += `<p>✓ ${r.title || r.file_path}: ${r.stories_ingested} stories ingested</p>`;
                        });
                        html += '</div>';
                        statusDiv.innerHTML = html;
                        input.value = '';
                    } else {
                        statusDiv.innerHTML = `<div class="error">Error: ${result.detail || 'Unknown error'}</div>`;
                    }
                } catch (error) {
                    statusDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
            
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
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        let html = '';
                        
                        // Show generated response
                        if (result.response) {
                            html += '<div class="response">';
                            html += '<h3>🤖 Generated Response</h3>';
                            html += '<p>' + result.response.replace(/\\n/g, '<br>') + '</p>';
                            html += '</div>';
                        }
                        
                        // Show citations
                        if (result.citations && result.citations.length > 0) {
                            html += '<div class="results">';
                            html += '<h3>📖 Relevant Story Excerpts</h3>';
                            result.citations.forEach((citation, idx) => {
                                html += '<div class="result-item">';
                                html += `<h3>Source ${idx + 1}: ${citation.metadata?.title || 'Untitled'}</h3>`;
                                if (citation.metadata?.source) {
                                    html += `<p class="score">From: ${citation.metadata.source}</p>`;
                                }
                                html += `<div class="content">${citation.content.substring(0, 500)}${citation.content.length > 500 ? '...' : ''}</div>`;
                                html += '</div>';
                            });
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

