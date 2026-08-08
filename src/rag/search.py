from typing import List, Dict, Any

from src.rag.embeddings_factory import get_embeddings_provider
from src.rag.vector_store_factory import get_vector_store
from src.rag.hybrid import rrf_merge
from src.core.logging import get_logger

logger = get_logger(__name__)


def hybrid_search_chunks(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval:
    1) dense vector search
    2) keyword search
    3) RRF merge
    """
    # Convert question to embedding for dense search
    embedder = get_embeddings_provider()
    query_vector = embedder.embed_texts([question])[0]

    store = get_vector_store()
    try:
        # Get candidates from both retrieval methods
        dense_results = store.search(query_vector, top_k=max(top_k * 2, 6))
        sparse_results = store.search_keyword(question, top_k=max(top_k * 2, 6))
    finally:
        store.close()

    # Merge both ranked lists
    merged = rrf_merge(dense_results, sparse_results, top_k=top_k)

    logger.info(
        "Hybrid search question='%s' dense=%d sparse=%d merged=%d",
        question,
        len(dense_results),
        len(sparse_results),
        len(merged),
    )
    return merged