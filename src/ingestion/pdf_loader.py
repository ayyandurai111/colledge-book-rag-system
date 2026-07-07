import fitz  # PyMuPDF
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def load_pdf(file_path: str) -> dict:
    """
    Load a textbook PDF and return a structured document dict:

    {
      "source":   "filename.pdf",
      "pdf_type": "toc" | "font",          # detected strategy
      "toc":      [...],                    # raw TOC if present
      "pages": [
        {
          "page":   int,
          "text":   str,
          "blocks": [ { "text", "font_size", "is_bold", "bbox" } ]
        },
        ...
      ]
    }
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    logger.info(f"Loading textbook PDF: {file_path}")

    with fitz.open(str(path)) as doc:
        toc = doc.get_toc(simple=False)   # [[level, title, page], ...]
        pdf_type = "toc" if toc else "font"
        logger.info(f"PDF type detected: {pdf_type} ({len(toc)} TOC entries)")

        pages = []
        for page_num, page in enumerate(doc, start=1):
            # Rich block extraction: font sizes + bold flags
            blocks = _extract_blocks(page)
            plain_text = "\n".join(b["text"] for b in blocks if b["text"].strip())

            if plain_text.strip():
                pages.append({
                    "page":   page_num,
                    "text":   plain_text,
                    "blocks": blocks
                })

    logger.info(f"Loaded {len(pages)} pages from '{path.name}'")
    return {
        "source":   path.name,
        "pdf_type": pdf_type,
        "toc":      toc,
        "pages":    pages
    }


def load_pdfs_from_dir(dir_path: str) -> list[dict]:
    """Load all PDFs in a directory. Returns list of document dicts."""
    dir_ = Path(dir_path)
    docs = []
    pdf_files = list(dir_.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in '{dir_path}'")
    for pdf_file in pdf_files:
        docs.append(load_pdf(str(pdf_file)))
    return docs


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _extract_blocks(page) -> list[dict]:
    """
    Extract text blocks with font metadata from a PyMuPDF page.
    Returns list of { text, font_size, is_bold, bbox }.
    """
    blocks = []
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block in raw.get("blocks", []):
        if block.get("type") != 0:          # 0 = text block
            continue
        for line in block.get("lines", []):
            line_text = ""
            max_font_size = 0.0
            is_bold = False

            for span in line.get("spans", []):
                span_text = span.get("text", "")
                font_size = span.get("size", 0.0)
                flags = span.get("flags", 0)
                bold = bool(flags & 2**4)    # bit 4 = bold in PyMuPDF

                line_text += span_text
                if font_size > max_font_size:
                    max_font_size = font_size
                if bold:
                    is_bold = True

            if line_text.strip():
                blocks.append({
                    "text":      line_text,
                    "font_size": round(max_font_size, 2),
                    "is_bold":   is_bold,
                    "bbox":      block.get("bbox", [])
                })

    return blocks
