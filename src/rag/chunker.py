from typing import List, Dict


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict]:
    """
    Split each page text into smaller overlapping chunks.
    """
    chunks: List[Dict] = []

    for page in pages:
        text = page["content"]
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()

            if piece:
                chunks.append(
                    {
                        "content": piece,
                        "source": page["source"],
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            # Move forward but keep overlap
            start = end - chunk_overlap
            if start < 0:
                start = 0
            if end >= len(text):
                break

    return chunks