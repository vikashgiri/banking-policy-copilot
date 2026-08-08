from src.rag.reranker_base import Reranker
from src.rag.reranker_simple import SimpleReranker


def get_reranker() -> Reranker:
    """
    Return configured reranker.
    Currently local simple reranker.
    """
    return SimpleReranker()