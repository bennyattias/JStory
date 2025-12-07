"""Script to ingest PDF or TXT stories from command line"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.dependencies import (
    get_ingest_use_case
)


async def main():
    """Main ingestion function"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_stories.py <file1> [file2] ...")
        print("Supported formats: .pdf, .txt (stories separated by empty lines)")
        sys.exit(1)
    
    use_case = get_ingest_use_case()
    
    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File not found: {file_path}")
            continue
        
        file_ext = path.suffix.lower()
        if file_ext not in ['.pdf', '.txt']:
            print(f"Warning: Skipping unsupported file type {file_ext}: {file_path}")
            print("Supported formats: .pdf, .txt")
            continue
        
        try:
            print(f"Processing: {file_path}")
            result = await use_case.execute(str(path))
            print(f"✓ Successfully ingested {result['stories_ingested']} stories from {result['title']}")
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

