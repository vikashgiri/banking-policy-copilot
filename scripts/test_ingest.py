from src.rag.ingest import ingest_pdf
from src.core.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    result = ingest_pdf("data/raw/sample_policy.pdf")
    logger.info("Ingest result: %s", result)


if __name__ == "__main__":
    main()