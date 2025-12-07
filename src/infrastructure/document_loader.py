"""Document loader implementation for PDF and TXT files"""
import os
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
from src.domain.services import DocumentLoaderService
from src.domain.models import StoryDocument


class PDFDocumentLoader(DocumentLoaderService):
    """Loads PDF and TXT documents and extracts text"""
    
    async def load_pdf(self, file_path: str) -> StoryDocument:
        """
        Load a PDF or TXT file and extract text content
        
        Args:
            file_path: Path to the PDF or TXT file
            
        Returns:
            StoryDocument with extracted content
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()
        
        # Extract filename as title
        filename = file_path_obj.name
        title = file_path_obj.stem
        
        try:
            if file_ext == '.txt':
                # Load text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    full_text = f.read()
            elif file_ext == '.pdf':
                # Load PDF file
                reader = PdfReader(file_path)
                text_content = []
                
                for page in reader.pages:
                    text_content.append(page.extract_text())
                
                full_text = "\n".join(text_content)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}. Supported: .txt, .pdf")
            
            return StoryDocument(
                source=file_path,
                title=title,
                content=full_text
            )
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")

