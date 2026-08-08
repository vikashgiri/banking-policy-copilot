import requests

from src.config import settings
from src.core.logging import get_logger
from src.llm.base import ChatProvider

logger = get_logger(__name__)


class OllamaChatProvider(ChatProvider):
    """Ollama chat model implementation."""

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.chat_model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        text = (data.get("response") or "").strip()
        logger.info("Ollama generated response length=%d", len(text))
        return text