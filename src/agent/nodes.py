from src.agent.state import AgentState
from src.llm.factory import get_chat_provider
from src.core.logging import get_logger
from src.rag.search import hybrid_search_chunks
from src.rag.reranker_factory import get_reranker

logger = get_logger(__name__)


def rewrite_node(state: AgentState) -> AgentState:
    """
    Rewrite user question into a clearer search query.
    """
    question = state["question"]
    llm = get_chat_provider()

    prompt = f"""Rewrite the question to make it clearer for document search.
Return only the rewritten question.

Question: {question}
Rewritten:"""

    rewritten = llm.generate(prompt).strip() or question

    steps = list(state.get("steps") or [])
    steps.append(f"rewrite: {rewritten}")

    logger.info("Rewrite node: %s -> %s", question, rewritten)

    return {
        **state,
        "rewritten_question": rewritten,
        "steps": steps,
        "error": None,
    }






def retrieve_node(state: AgentState) -> AgentState:
    """
    Retrieve and rerank relevant policy chunks.
    """
    query = state.get("rewritten_question") or state["question"]

    # Hybrid retrieve candidates
    chunks = hybrid_search_chunks(query, top_k=9)

    # Rerank to best few
    reranker = get_reranker()
    chunks = reranker.rerank(query, chunks, top_k=3)

    steps = list(state.get("steps") or [])
    steps.append(f"retrieve: {len(chunks)} chunks")

    logger.info("Retrieve node got %d chunks", len(chunks))

    return {
        **state,
        "chunks": chunks,
        "steps": steps,
        "error": None,
    }





def answer_node(state: AgentState) -> AgentState:
    """
    Generate final grounded answer from chunks.
    """
    question = state["question"]
    chunks = state.get("chunks") or []

    if not chunks:
        answer = "I don't know based on the available information."
        steps = list(state.get("steps") or [])
        steps.append("answer: no chunks")
        return {
            **state,
            "answer": answer,
            "steps": steps,
            "error": None,
        }

    # Build context with citations
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        page = chunk.get("metadata", {}).get("page_number", "?")
        context_parts.append(
            f"[{i}] (source={source}, page={page})\n{chunk.get('content', '')}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a banking policy assistant.
Answer ONLY using the context below.
If answer is not in context, say: "I don't know based on the available information."
Include citation numbers like [1], [2] when possible.

Context:
{context}

Question: {question}

Answer:"""

    llm = get_chat_provider()
    answer = llm.generate(prompt).strip()

    steps = list(state.get("steps") or [])
    steps.append("answer: generated")

    logger.info("Answer node produced response")

    return {
        **state,
        "answer": answer,
        "steps": steps,
        "error": None,
    }