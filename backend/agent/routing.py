from backend.agent.state import MAX_AGENT_STEPS, MAX_CHART_TRIES, MAX_SQL_TRIES, QueryState


def after_guardrail(state: QueryState) -> str:
    return "failure" if state.get("error") else state["route"]


def after_generation(state: QueryState) -> str:
    if not state.get("error"):
        return "execute"
    return "generate_sql" if state.get("sql_tries", 0) < MAX_SQL_TRIES else "partial"


def after_execution(state: QueryState) -> str:
    if not state.get("error"):
        return "assess"
    return "generate_sql" if state.get("sql_tries", 0) < MAX_SQL_TRIES else "partial"


def after_assessment(state: QueryState) -> str:
    if state.get("error"):
        return "partial"
    action = state.get("action", {}).get("name")
    if action == "answer":
        return "done"
    if action == "query_sql" and state.get("steps", 0) >= MAX_AGENT_STEPS:
        return "partial"
    return {
        "query_sql": "generate_sql",
        "calculate_statistics": "statistics",
        "create_chart": "chart",
    }.get(action, "partial")


def after_tool(state: QueryState) -> str:
    return "partial" if state.get("error") else "assess"


def after_chart(state: QueryState) -> str:
    if not state.get("error"):
        return "done"
    if state.get("chart_tries", 0) >= MAX_CHART_TRIES:
        return "partial"
    return "assess"
