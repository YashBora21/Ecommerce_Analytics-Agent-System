import time

from backend.agent.state import (
    MAX_AGENT_STEPS,
    MAX_CHART_TRIES,
    MAX_RUN_SECONDS,
    MAX_SQL_TRIES,
    MAX_STATISTICS_TRIES,
    MAX_TOTAL_TOKENS,
    QueryState,
)
from backend.agent.telemetry import get_usage


def budget_exhausted(state: QueryState) -> bool:
    return (
        state.get("steps", 0) >= MAX_AGENT_STEPS
        or get_usage()["total_tokens"] >= MAX_TOTAL_TOKENS
        or time.time() - state.get("started_at", time.time()) >= MAX_RUN_SECONDS
    )


def after_inspection(state: QueryState) -> str:
    return "partial" if state.get("error") else "agent"


def after_agent(state: QueryState) -> str:
    if state.get("error"):
        return "partial"
    action = state.get("action", {}).get("name")
    if action in {"answer", "clarify", "stop"}:
        return "done"
    if budget_exhausted(state):
        return "partial"
    return {
        "query_sql": "sql",
        "calculate_statistics": "statistics",
        "create_chart": "chart",
    }.get(action, "partial")


def after_sql(state: QueryState) -> str:
    if not state.get("error"):
        return "agent"
    if state.get("sql_tries", 0) < MAX_SQL_TRIES and not budget_exhausted(state):
        return "agent"
    return "partial"


def after_statistics(state: QueryState) -> str:
    if not state.get("error"):
        return "agent"
    if state.get("statistics_tries", 0) >= MAX_STATISTICS_TRIES or budget_exhausted(state):
        return "partial"
    return "agent"


def after_chart(state: QueryState) -> str:
    if not state.get("error"):
        return "done"
    if state.get("chart_tries", 0) >= MAX_CHART_TRIES or budget_exhausted(state):
        return "partial"
    return "agent"
