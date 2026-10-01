import uuid

from backend.agent.graph import build_query_graph, default_checkpointer, query_graph
from backend.agent.state import (
    MAX_AGENT_STEPS,
    MAX_CHART_TRIES,
    MAX_SQL_TRIES,
    LlmCall,
    QueryState,
)


def ask(
    question: str,
    thread_id: str | None = None,
    llm_call: LlmCall | None = None,
) -> QueryState:
    if not question or not question.strip():
        return {"question": question, "final_answer": "Question cannot be empty."}

    graph = query_graph
    if llm_call:
        graph = build_query_graph(
            llm_call=llm_call,
            checkpointer=default_checkpointer,
        )

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    return graph.invoke({"question": question.strip()}, config=config)


__all__ = [
    "MAX_AGENT_STEPS",
    "MAX_CHART_TRIES",
    "MAX_SQL_TRIES",
    "QueryState",
    "ask",
    "build_query_graph",
]
