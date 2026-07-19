"""
exceptions.py
-------------
Custom exception hierarchy used throughout this package. Since every
exception derives from the same base class, callers can catch all
library errors with a single `except CollegeRAGError`.
"""


class CollegeRAGError(Exception):
    """Base class for all custom exceptions in this package."""


class UnsupportedFileTypeError(CollegeRAGError):
    """Raised when a file type other than PDF/DOCX is provided."""


class EmptyDocumentError(CollegeRAGError):
    """Raised when no text could be extracted from a file."""


class IndexNotBuiltError(CollegeRAGError):
    """Raised when a search is attempted before the index has been built/loaded."""


class EmptyIndexError(CollegeRAGError):
    """Raised when attempting to build an index with zero chunks."""


class EmbeddingError(CollegeRAGError):
    """Raised when the embedding model fails to load or encode text."""
