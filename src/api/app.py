from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import settings
from src.rag.pipeline import answer_question
from src.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(title=settings.app_name)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    mode: str = Field(default="agentic")  # agentic | pipeline


class AskResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    logger.info("API /ask mode=%s question=%s", request.mode, request.question)

    # Pipeline mode: hybrid + rerank + generate
    if request.mode == "pipeline":
        result = answer_question(request.question)
        return AskResponse(
            question=result["question"],
            answer=result["answer"],
        )

    # Agentic mode: LangGraph rewrite → retrieve → answer
    from src.agent.graph import build_agent_graph

    agent = build_agent_graph()
    final_state = agent.invoke(
        {
            "question": request.question,
            "rewritten_question": "",
            "chunks": [],
            "answer": "",
            "steps": [],
            "error": None,
        }
    )

    return AskResponse(
        question=request.question,
        answer=final_state.get("answer", ""),
    )