import os

from typing import Any, Callable, TypedDict


Message = dict[str, str]
Evidence = dict[str, Any]
LlmCall = Callable[[list[Message]], str]
SqlRunner = Callable[..., str]

MAX_SQL_TRIES = 3
MAX_AGENT_STEPS = 12
MAX_CHART_TRIES = 2
MAX_RECENT_MESSAGES = 6
MAX_TOTAL_TOKENS = int(os.getenv("MAX_AGENT_TOKENS", "30000"))
MAX_RUN_SECONDS = float(os.getenv("MAX_RUN_SECONDS", "300"))


class QueryState(TypedDict, total=False):
    question: str
    owner_id: str
    action: dict
    evidence: list[Evidence]
    artifacts: list[dict]
    schema_context: str
    allowed_tables: list[str]
    analysis: dict
    chart: str
    error: str
    final_answer: str
    sql: str
    sql_tries: int
    chart_tries: int
    steps: int
    started_at: float
    history: list[Message]
    history_summary: str
    last_sql: str
