"""
base.py
-------
A common interface shared by all file-type extractors. To support a new
file type (e.g. .pptx, .txt), extend this class and implement `extract()`.
"""
from abc import ABC, abstractmethod
from typing import List

from college_rag.models import TextBlock


class BaseExtractor(ABC):
    """Abstract base class for text extractors."""

    @abstractmethod
    def extract(self, path: str) -> List[TextBlock]:
        """Extracts a list of TextBlocks from the given file.

        Args:
            path: Absolute or relative path to the file.

        Returns:
            A list of TextBlock objects (only ones with non-empty text).

        Raises:
            EmptyDocumentError: If no text could be extracted at all.
        """
        raise NotImplementedError
