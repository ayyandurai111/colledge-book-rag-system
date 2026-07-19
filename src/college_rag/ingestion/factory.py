"""
factory.py
----------
Routes a file path to the correct extractor based on its extension.
"""
import os
from typing import List

from college_rag.exceptions import UnsupportedFileTypeError
from college_rag.ingestion.docx_extractor import DocxExtractor
from college_rag.ingestion.pdf_extractor import PDFExtractor
from college_rag.models import TextBlock

_EXTRACTORS = {
    ".pdf": PDFExtractor(),
    ".docx": DocxExtractor(),
}


def extract(path: str) -> List[TextBlock]:
    """Given a file path, returns TextBlocks via the appropriate extractor.

    Raises:
        UnsupportedFileTypeError: For file types other than .pdf/.docx.
        EmptyDocumentError: If no text could be extracted.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    extractor = _EXTRACTORS.get(ext)
    if extractor is None:
        supported = ", ".join(sorted(_EXTRACTORS.keys()))
        raise UnsupportedFileTypeError(
            f"Unsupported file type: '{ext}'. Supported types: {supported}"
        )
    return extractor.extract(path)
