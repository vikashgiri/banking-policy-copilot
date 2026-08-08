import json
from pathlib import Path
from typing import List, Dict, Any

from src.rag.pipeline import answer_question
from src.core.logging import get_logger

logger = get_logger(__name__)

def load_eval_questions(path: str = "data/eval/questions.json") -> List[Dict[str, Any]]:
    p = Path(path)

    # If relative, resolve from project root
    if not p.is_absolute():
        root = Path(__file__).resolve().parents[2]
        p = root / p

    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def score_answer(answer: str, must_include_any: List[str]) -> Dict[str, Any]:
    text = (answer or "").lower()
    matched = [k for k in must_include_any if k.lower() in text]
    return {
        "passed": len(matched) > 0,
        "matched_keywords": matched,
    }


def run_evaluation(path: str = "data/eval/questions.json") -> Dict[str, Any]:
    questions = load_eval_questions(path)
    results = []
    passed_count = 0

    for item in questions:
        qid = item["id"]
        question = item["question"]
        expected = item.get("must_include_any", [])

        logger.info("Evaluating %s: %s", qid, question)
        output = answer_question(question)
        answer = output.get("answer", "")

        score = score_answer(answer, expected)
        if score["passed"]:
            passed_count += 1

        results.append(
            {
                "id": qid,
                "question": question,
                "answer": answer,
                "passed": score["passed"],
                "matched_keywords": score["matched_keywords"],
            }
        )

    return {
        "total": len(questions),
        "passed": passed_count,
        "failed": len(questions) - passed_count,
        "pass_rate": round(passed_count / max(len(questions), 1), 2),
        "results": results,
    }