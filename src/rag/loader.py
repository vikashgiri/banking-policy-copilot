from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader

from src.core.logging import get_logger

logger = get_logger(__name__)


def load_pdf(file_path: str) -> List[Dict]:
    """
    Load text from a PDF file page by page.
    Returns a list of pages with page number and text.
    """
    path = Path(file_path)

    # Stop early if file does not exist
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    pages: List[Dict] = []

    # Read each page text
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        # Skip empty pages
        if not text:
            continue

        pages.append(
            {
                "page_number": i + 1,
                "content": text,
                "source": path.name,
            }
        )

    logger.info("Loaded PDF '%s' with %d pages", path.name, len(pages))
    return pages