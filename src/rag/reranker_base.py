from abc import ABC, abstractmethod
from typing import List, Dict, Any


class Reranker(ABC):
    """
    Abstract reranker contract.
    """

    @abstractmethod
    def rerank(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Re-score chunks for a question and return top_k.
        """
        raise NotImplementedError