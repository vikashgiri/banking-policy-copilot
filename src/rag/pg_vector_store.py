from typing import List, Dict, Any

import psycopg2
from psycopg2.extras import Json

from src.config import settings
from src.core.logging import get_logger
from src.rag.vector_store_base import VectorStore

logger = get_logger(__name__)


class PgVectorStore(VectorStore):
    """Postgres + pgvector implementation."""

    def __init__(self) -> None:
        # Connect using DATABASE_URL from config
        if not settings.database_url:
            raise ValueError("DATABASE_URL is required")
        self.conn = psycopg2.connect(settings.database_url)
        self.conn.autocommit = True

    def add(self, items: List[Dict[str, Any]]) -> None:
        """Insert chunk text + embedding + metadata."""
        if not items:
            return

        with self.conn.cursor() as cur:
            for item in items:
                content = item["content"]
                embedding = item["embedding"]
                metadata = item.get("metadata", {}) or {}

                source = metadata.get("source")
                page_number = metadata.get("page_number")
                chunk_index = metadata.get("chunk_index")

                # pgvector needs vector as text like [0.1,0.2]
                embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"

                cur.execute(
                    """
                    INSERT INTO documents
                    (content, embedding, source, page_number, chunk_index, metadata)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                    """,
                    (
                        content,
                        embedding_str,
                        source,
                        page_number,
                        chunk_index,
                        Json(metadata),
                    ),
                )

        logger.info("Inserted %d chunks", len(items))

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top similar chunks using cosine distance."""
        embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source, page_number, chunk_index, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, top_k),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            content, source, page_number, chunk_index, metadata, score = row
            meta = metadata if isinstance(metadata, dict) else {}
            meta = {
                **meta,
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
            }
            results.append(
                {
                    "content": content,
                    "metadata": meta,
                    "score": float(score) if score is not None else 0.0,
                }
            )
        return results

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def search_keyword(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Sparse search using Postgres full-text ranking.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    content,
                    source,
                    page_number,
                    chunk_index,
                    metadata,
                    ts_rank(
                        to_tsvector('english', content),
                        plainto_tsquery('english', %s)
                    ) AS score
                FROM documents
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, top_k),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            content, source, page_number, chunk_index, metadata, score = row
            meta = metadata if isinstance(metadata, dict) else {}
            meta = {
                **meta,
                "source": source,
                "page_number": page_number,
                "chunk_index": chunk_index,
            }
            results.append(
                {
                    "content": content,
                    "metadata": meta,
                    "score": float(score) if score is not None else 0.0,
                }
            )

        logger.info("Keyword search returned %d results", len(results))
        return results

    def close(self) -> None:
        self.conn.close()