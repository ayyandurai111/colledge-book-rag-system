"""
heading_utils.py
----------------
Shared heuristics for detecting which lines of plain text extracted from
a PDF/DOCX are likely chapter/section headings.

(DOCX has proper "Heading" paragraph styles we can use directly — this
heuristic mainly matters for PDF, since style information is lost when
extracting plain text from a PDF.)
"""
import re

_HEADING_PATTERNS = [
    # English markers ("Chapter 3", "Unit 2", "Section 1") and their Tamil
    # equivalents (அத்தியாயம் = chapter, பாடம் = lesson/unit) — the system
    # still needs to recognize Tamil-language textbook structure even though
    # the codebase itself is documented in English.
    re.compile(r'^\s*(chapter|unit|section|அத்தியாயம்|பாடம்)\s+\d+', re.IGNORECASE),
    re.compile(r'^\s*\d+(\.\d+)*\s+[A-Z][a-zA-Z ]{3,60}$'),
]

_MAX_HEADING_LENGTH = 90
_MIN_HEADING_WORDS = 3
_MAX_HEADING_WORDS = 12


def looks_like_heading(line: str) -> bool:
    """Treats short, uppercase lines, or patterns like "Chapter 3" as headings.

    This is a heuristic — not 100% accurate. False positives/negatives can
    occur, so critical logic should not depend on this alone.
    """
    line = line.strip()
    if not line:
        return False
    if len(line) > _MAX_HEADING_LENGTH:
        return False

    for pattern in _HEADING_PATTERNS:
        if pattern.match(line):
            return True

    words = line.split()
    if line.isupper() and _MIN_HEADING_WORDS <= len(words) <= _MAX_HEADING_WORDS:
        return True

    return False
