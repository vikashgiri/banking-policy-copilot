from src.rag.vector_store_base import VectorStore
from src.rag.pg_vector_store import PgVectorStore


def get_vector_store() -> VectorStore:
    """
    Return configured vector store implementation.
    Currently only pgvector.
    """
    return PgVectorStore()