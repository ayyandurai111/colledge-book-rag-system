"""
PDF Type Detector
-----------------
Decides which chunking strategy to use for a loaded textbook PDF.

Strategy:
  "toc"  → TOC/bookmark-based chunking   (preferred when bookmarks exist)
  "font" → Font-size/bold-based chunking  (fallback for plain text PDFs)
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_TOC_ENTRIES = 3   # at least this many entries to trust the TOC


def detect_pdf_type(doc: dict) -> str:
    """
    Return "toc" or "font" based on the loaded document dict.

    Args:
        doc: Output of pdf_loader.load_pdf()
    """
    toc = doc.get("toc", [])
    if len(toc) >= MIN_TOC_ENTRIES:
        logger.info(f"Using TOC-based chunking ({len(toc)} TOC entries)")
        return "toc"

    logger.info("Using font-size/bold-based chunking (no usable TOC)")
    return "font"
