from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import rewrite_node, retrieve_node, answer_node
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_agent_graph():
    """
    Build agentic RAG graph.
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)

    # Define flow order
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()
    logger.info("LangGraph agent compiled")
    return app