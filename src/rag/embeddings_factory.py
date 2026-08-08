from src.config import settings
from src.rag.embeddings_base import EmbeddingsProvider
from src.rag.embeddings_ollama import OllamaEmbeddingsProvider


def get_embeddings_provider() -> EmbeddingsProvider:
    """
    Create embeddings provider based on configuration.
    """
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        return OllamaEmbeddingsProvider()

    # Future:
    # if provider == "openai":
    #     return OpenAIEmbeddingsProvider()

    raise ValueError(f"Unsupported embeddings provider: {provider}")