from typing import List

import requests

from src.config import settings
from src.core.logging import get_logger
from src.rag.embeddings_base import EmbeddingsProvider

logger = get_logger(__name__)


class OllamaEmbeddingsProvider(EmbeddingsProvider):
    """
    Ollama-based embedding provider.
    Implements EmbeddingsProvider contract.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        # Read from config by default, allow override for testing
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.embedding_model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []

        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            vectors.append(data["embedding"])

        logger.info(
            "Ollama embeddings created count=%d model=%s",
            len(vectors),
            self.model,
        )
        return vectors