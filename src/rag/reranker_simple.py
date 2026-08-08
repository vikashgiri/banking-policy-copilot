from typing import List, Dict, Any

from src.rag.reranker_base import Reranker
from src.core.logging import get_logger

logger = get_logger(__name__)


class SimpleReranker(Reranker):
    """
    Local reranker using keyword overlap + retrieval score.
    """

    def rerank(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        q_terms = set(question.lower().split())

        scored: List[Dict[str, Any]] = []
        for chunk in chunks:
            content = (chunk.get("content") or "").lower()
            c_terms = set(content.split())

            # How many question words appear in chunk
            overlap = len(q_terms.intersection(c_terms))
            base = float(chunk.get("score") or 0.0)

            # Final rerank score
            final_score = base + (0.1 * overlap)

            new_item = {
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "score": final_score,
            }
            scored.append(new_item)

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:top_k]

        logger.info(
            "Reranked candidates=%d kept=%d question='%s'",
            len(chunks),
            len(top),
            question,
        )
        return top