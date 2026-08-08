from pathlib import Path
import os
from dotenv import load_dotenv

# Always load .env from project root (not current working directory guess)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    app_name: str = os.getenv("APP_NAME", "banking-policy-copilot")
    app_env: str = os.getenv("APP_ENV", "development")

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    database_url: str = os.getenv("DATABASE_URL", "")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.2")

settings = Settings()