"""
docx_extractor.py
------------------
Text extraction from DOCX college textbooks. DOCX files carry proper
"Heading" paragraph styles, so chapter/section detection here is more
reliable than the PDF heuristic.
"""
import os
from typing import List

import docx
from docx.opc.exceptions import PackageNotFoundError

from college_rag.exceptions import EmptyDocumentError
from college_rag.ingestion.base import BaseExtractor
from college_rag.models import TextBlock


class DocxExtractor(BaseExtractor):
    """Extracts text from DOCX files using python-docx."""

    def extract(self, path: str) -> List[TextBlock]:
        if not os.path.exists(path):
            # python-docx raises the same PackageNotFoundError for both a
            # missing file AND a corrupt/non-zip file, so we must check
            # existence ourselves to give callers an accurate exception type.
            raise FileNotFoundError(f"File not found: {path}")

        try:
            document = docx.Document(path)
        except PackageNotFoundError as e:
            raise EmptyDocumentError(f"Not a valid DOCX file: {path} ({e})") from e

        blocks: List[TextBlock] = []
        current_heading = ""
        fname = os.path.basename(path)
        buffer = []

        for para in document.paragraphs:
            style_name = (para.style.name or "").lower() if para.style else ""
            text = para.text.strip()
            if not text:
                continue

            if "heading" in style_name or "title" in style_name:
                if buffer:
                    joined = "\n".join(buffer).strip()
                    if joined:
                        blocks.append(TextBlock(
                            text=joined, source_file=fname, heading=current_heading,
                        ))
                    buffer = []
                current_heading = text
            else:
                buffer.append(text)

        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                blocks.append(TextBlock(
                    text=joined, source_file=fname, heading=current_heading,
                ))

        if not blocks:
            raise EmptyDocumentError(f"Could not extract any text from '{fname}'.")
        return blocks
