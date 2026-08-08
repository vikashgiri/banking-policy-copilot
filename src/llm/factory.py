from src.config import settings
from src.llm.base import ChatProvider
from src.llm.ollama_chat import OllamaChatProvider


def get_chat_provider() -> ChatProvider:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaChatProvider()

    raise ValueError(f"Unsupported chat provider: {provider}")