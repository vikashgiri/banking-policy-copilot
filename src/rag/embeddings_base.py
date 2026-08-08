from abc import ABC, abstractmethod
from typing import List


class EmbeddingsProvider(ABC):
    """
    Abstract embedding provider.
    Any embedding backend must implement this contract.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert texts into vectors.
        """
        raise NotImplementedError