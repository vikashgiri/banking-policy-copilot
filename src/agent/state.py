from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    # Original user question
    question: str

    # Optional rewritten question for better retrieval
    rewritten_question: str

    # Retrieved/reranked chunks
    chunks: List[Dict[str, Any]]

    # Final model answer
    answer: str

    # Step log for debugging
    steps: List[str]

    # Optional error message
    error: Optional[str]