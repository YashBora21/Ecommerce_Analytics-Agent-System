from backend.agent.prompt import MEMORY_SUMMARY_PROMPT
from backend.agent.state import LlmCall, MAX_RECENT_MESSAGES, QueryState


def format_evidence(state: QueryState) -> str:
    return "\n\n".join(
        f"SQL {index}: {item['sql']}\nResult {index}:\n{item['result']}"
        for index, item in enumerate(state.get("evidence", []), start=1)
    )


def latest_sql_result(state: QueryState) -> str:
    for item in reversed(state.get("evidence", [])):
        if item.get("kind", "sql") == "sql":
            return item["result"]
    raise ValueError("No SQL result is available")


def recent_context(state: QueryState) -> str:
    parts = []
    if state.get("history_summary"):
        parts.append(f"Earlier summary: {state['history_summary']}")
    if state.get("last_sql"):
        parts.append(f"Last successful SQL: {state['last_sql']}")
    history = state.get("history", [])[-MAX_RECENT_MESSAGES:]
    if history:
        messages = "\n".join(
            f"{message['role']}: {message['content']}" for message in history
        )
        parts.append(f"Recent messages:\n{messages}")
    return "\n\n".join(parts)


def make_remember(llm_call: LlmCall):
    def remember(state: QueryState, answer: str) -> QueryState:
        history = list(state.get("history", []))
        history.extend(
            [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": answer},
            ]
        )
        summary = state.get("history_summary", "")
        if len(history) > MAX_RECENT_MESSAGES:
            older = history[:-MAX_RECENT_MESSAGES]
            dialogue = "\n".join(
                f"{message['role']}: {message['content']}" for message in older
            )
            try:
                summary = llm_call(
                    [
                        {"role": "system", "content": MEMORY_SUMMARY_PROMPT},
                        {
                            "role": "user",
                            "content": f"Existing summary: {summary}\n\nOlder messages:\n{dialogue}",
                        },
                    ]
                ).strip()
            except Exception:
                summary = summary or dialogue[:500]
            history = history[-MAX_RECENT_MESSAGES:]

        memory: QueryState = {
            "final_answer": answer,
            "history": history,
            "history_summary": summary,
        }
        for item in reversed(state.get("evidence", [])):
            if item.get("kind", "sql") == "sql":
                memory["last_sql"] = item["sql"]
                break
        return memory

    return remember
