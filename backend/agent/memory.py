import csv
import io
import re

from backend.agent.prompt import MEMORY_SUMMARY_PROMPT
from backend.agent.state import LlmCall, MAX_RECENT_MESSAGES, QueryState


def format_evidence(state: QueryState) -> str:
    return "\n\n".join(
        f"SQL {index}: {item['sql']}\nResult {index}:\n{item['result']}"
        for index, item in enumerate(state.get("evidence", []), start=1)
    )


def sql_result(
    state: QueryState,
    required_columns: set[str] | None = None,
) -> str:
    required = required_columns or set()
    for item in reversed(state.get("evidence", [])):
        if item.get("kind", "sql") != "sql":
            continue
        if not item.get("complete", True):
            continue
        if required:
            header = next(csv.reader(io.StringIO(item["result"])), [])
            if not required <= set(header):
                continue
        if item.get("artifact_id"):
            raise ValueError(
                "Large artifact results must be aggregated with query_sql before using this tool"
            )
        return item["result"]
    raise ValueError("No complete SQL result with the required columns is available")


def recent_context(state: QueryState) -> str:
    parts = [
        "Conversation memory is context, not evidence. Use it only to resolve an "
        "explicit follow-up; never carry old filters into a self-contained question."
    ]
    if state.get("history_summary"):
        parts.append(f"Earlier summary: {state['history_summary']}")
    artifacts = [
        {
            "artifact_id": item.get("artifact_id"),
            "relation": item.get("relation"),
            "row_count": item.get("row_count"),
            "columns": item.get("columns"),
            "has_limit": bool(item.get("has_limit", False)),
            "expires_at": item.get("expires_at"),
        }
        for item in state.get("artifacts", [])
        if item.get("complete")
    ]
    if artifacts:
        parts.append(f"Available result artifacts: {artifacts}")
    history = state.get("history", [])[-MAX_RECENT_MESSAGES:]
    if history:
        messages = "\n".join(
            f"{message['role']}: {message['content']}" for message in history
        )
        parts.append(f"Recent messages:\n{messages}")
    return "\n\n".join(parts)


def asks_to_recall_previous_result(question: str) -> bool:
    return bool(
        re.search(
            r"\b(previous|earlier|last)\s+(answer|result|response|query)\b|"
            r"\b(that|those)\s+(answer|result|response|query|results)\b|"
            r"\bwhat\s+was\s+(it|that)\b",
            question,
            re.IGNORECASE,
        )
    )


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
