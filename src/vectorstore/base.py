from abc import ABC, abstractmethod


class BaseVectorStore(ABC):

    @abstractmethod
    def add(self, chunks: list[dict]) -> None:
        """Add chunks with embeddings to the store."""
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Search and return top-K similar chunks."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return number of stored vectors."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all vectors."""
        pass
