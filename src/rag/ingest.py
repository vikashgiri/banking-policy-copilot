from src.rag.loader import load_pdf
from src.rag.chunker import chunk_pages
from src.rag.embeddings_factory import get_embeddings_provider
from src.rag.vector_store_factory import get_vector_store
from src.core.logging import get_logger

logger = get_logger(__name__)


def ingest_pdf(file_path: str) -> dict:
    """
    Ingest one PDF into vector database.
    """
    # 1) Load pages from PDF
    pages = load_pdf(file_path)

    # 2) Split pages into chunks
    chunks = chunk_pages(pages)

    # 3) Create embeddings for chunk texts
    embedder = get_embeddings_provider()
    texts = [c["content"] for c in chunks]
    vectors = embedder.embed_texts(texts)

    # 4) Prepare rows for vector store
    items = []
    for chunk, vector in zip(chunks, vectors):
        items.append(
            {
                "content": chunk["content"],
                "embedding": vector,
                "metadata": {
                    "source": chunk["source"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                },
            }
        )

    # 5) Save into vector DB
    store = get_vector_store()
    try:
        store.add(items)
        total = store.count()
    finally:
        store.close()

    result = {
        "pages": len(pages),
        "chunks": len(chunks),
        "stored_total": total,
    }
    logger.info("Ingest finished: %s", result)
    return result