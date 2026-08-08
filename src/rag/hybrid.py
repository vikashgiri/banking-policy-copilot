from typing import List, Dict, Any


def rrf_merge(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 5,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    score = sum 1 / (k + rank)
    """
    fused: Dict[str, Dict[str, Any]] = {}

    def add_list(results: List[Dict[str, Any]]) -> None:
        for rank, item in enumerate(results, start=1):
            # Use content + source + page as identity key
            meta = item.get("metadata", {})
            key = f"{meta.get('source')}::{meta.get('page_number')}::{meta.get('chunk_index')}::{item.get('content','')[:80]}"

            if key not in fused:
                fused[key] = {
                    "content": item["content"],
                    "metadata": meta,
                    "score": 0.0,
                }

            fused[key]["score"] += 1.0 / (k + rank)

    add_list(dense_results)
    add_list(sparse_results)

    merged = list(fused.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]