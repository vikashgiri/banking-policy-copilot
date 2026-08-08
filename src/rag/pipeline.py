from src.rag.search import hybrid_search_chunks
from src.llm.factory import get_chat_provider
from src.core.logging import get_logger
from src.rag.reranker_factory import get_reranker
logger = get_logger(__name__)


def build_prompt(question: str, chunks: list) -> str:
    # Build context with simple citation numbers
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page_number", "?")
        context_parts.append(
            f"[{i}] (source={source}, page={page})\n{chunk['content']}"
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
    return prompt



def answer_question(question: str, top_k: int = 3) -> dict:
    # 1) Hybrid retrieve more candidates
    chunks = hybrid_search_chunks(question, top_k=max(top_k * 3, 9))

    # 2) Rerank and keep best chunks
    reranker = get_reranker()
    chunks = reranker.rerank(question, chunks, top_k=top_k)

    # 3) Build grounded prompt
    prompt = build_prompt(question, chunks)

    # 4) Generate answer
    llm = get_chat_provider()
    answer = llm.generate(prompt)

    logger.info("Reranked Hybrid RAG answer generated for question: %s", question)

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
    }