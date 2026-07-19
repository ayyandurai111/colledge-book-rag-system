"""
pdf_extractor.py
----------------
Text extraction from PDF college textbooks, with heading/chapter detection.
"""
import os
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from college_rag.exceptions import EmptyDocumentError
from college_rag.ingestion.base import BaseExtractor
from college_rag.ingestion.heading_utils import looks_like_heading
from college_rag.models import TextBlock


class PDFExtractor(BaseExtractor):
    """Extracts text from PDF files using pypdf."""

    def extract(self, path: str) -> List[TextBlock]:
        try:
            reader = PdfReader(path)
        except PdfReadError as e:
            raise EmptyDocumentError(f"Could not read PDF file: {path} ({e})") from e
        except FileNotFoundError:
            raise
        except Exception as e:  # pypdf can raise assorted low-level errors on corrupt files
            raise EmptyDocumentError(f"Could not read PDF file: {path} ({e})") from e

        if getattr(reader, "is_encrypted", False):
            # Try an empty-password decrypt (common for "restricted but not really locked" PDFs)
            try:
                reader.decrypt("")
            except Exception:
                pass

        blocks: List[TextBlock] = []
        current_heading = ""
        fname = os.path.basename(path)

        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            lines = raw.split("\n")
            buffer = []

            for line in lines:
                if looks_like_heading(line):
                    if buffer:
                        text = "\n".join(buffer).strip()
                        if text:
                            blocks.append(TextBlock(
                                text=text, source_file=fname,
                                page_number=page_num, heading=current_heading,
                            ))
                        buffer = []
                    current_heading = line.strip()
                else:
                    buffer.append(line)

            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    blocks.append(TextBlock(
                        text=text, source_file=fname,
                        page_number=page_num, heading=current_heading,
                    ))

        if not blocks:
            raise EmptyDocumentError(
                f"Could not extract any text from '{fname}' "
                "(it may be a scanned/image-only PDF — OCR would be required)."
            )
        return blocks
