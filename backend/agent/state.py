from typing import Callable, TypedDict


Message = dict[str, str]
Evidence = dict[str, str]
LlmCall = Callable[[list[Message]], str]
SqlRunner = Callable[[str], str]

MAX_SQL_TRIES = 5
MAX_AGENT_STEPS = 5
MAX_CHART_TRIES = 2
MAX_RECENT_MESSAGES = 6


class QueryState(TypedDict, total=False):
    question: str
    action: dict
    evidence: list[Evidence]
    chart: str
    error: str
    final_answer: str
    sql: str
    sql_tries: int
    chart_tries: int
    steps: int
    history: list[Message]
    history_summary: str
    last_sql: str

