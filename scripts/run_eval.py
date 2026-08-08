import json
from pathlib import Path

from src.eval.runner import run_evaluation
from src.core.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    summary = run_evaluation("data/eval/questions.json")

    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "eval" / "last_report.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Total: %s", summary["total"])
    logger.info("Passed: %s", summary["passed"])
    logger.info("Failed: %s", summary["failed"])
    logger.info("Pass rate: %s", summary["pass_rate"])
    logger.info("Report saved: %s", output_path)


if __name__ == "__main__":
    main()