from abc import ABC, abstractmethod
from typing import List, Dict, Any


class VectorStore(ABC):
    """
    Abstract store for chunk storage and retrieval.
    """

    @abstractmethod
    def add(self, items: List[Dict[str, Any]]) -> None:
        """Store chunks with embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Dense/vector similarity search."""
        raise NotImplementedError

    @abstractmethod
    def search_keyword(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Sparse/keyword search."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Total stored chunks."""
        raise NotImplementedError