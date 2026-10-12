import os
import time
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from backend.agent.llm import call_groq
from backend.agent.memory import make_remember
from backend.agent.nodes.assess import make_assess_result
from backend.agent.nodes.schema import make_inspect_schema
from backend.agent.nodes.sql import make_execute_query
from backend.agent.nodes.terminal import partial_answer
from backend.agent.nodes.tools import make_calculate_statistic, make_create_plot
from backend.agent.routing import (
    after_agent,
    after_chart,
    after_inspection,
    after_sql,
    after_statistics,
)
from backend.agent.state import LlmCall, QueryState, SqlRunner
from backend.agent.telemetry import logged_node, reset_usage
from backend.tools.artifacts import owned_artifacts
from backend.tools.chart import run_chart
from backend.tools.schema import discover_schema
from backend.tools.sql import run_managed_sql
from backend.tools.sql_analysis import analyze_sql
from backend.tools.statistics import run_statistics


_checkpoint_pool = None


def begin_turn(_state: QueryState) -> QueryState:
    reset_usage()
    return {
        "action": {},
        "evidence": [],
        "schema_context": "",
        "allowed_tables": [],
        "analysis": {},
        "chart": "",
        "error": "",
        "final_answer": "",
        "incomplete": False,
        "sql": "",
        "sql_tries": 0,
        "chart_tries": 0,
        "statistics_tries": 0,
        "steps": 0,
        "started_at": time.time(),
    }


def _production_schema_runner(question, state):
    discovered = discover_schema(question)
    artifacts = owned_artifacts(state.get("owner_id", ""))
    discovered["artifacts"] = artifacts
    return discovered

def _state_artifact_relations(state: QueryState) -> set[str]:
    now = datetime.now(timezone.utc)
    relations = set()
    for artifact in state.get("artifacts", []):
        expires_at = artifact.get("expires_at")
        if not artifact.get("complete") or not artifact.get("relation") or not expires_at:
            continue
        if datetime.fromisoformat(expires_at) > now:
            relations.add(artifact["relation"].lower())
    return relations


def _production_analyzer(query, state):
    return analyze_sql(
        query,
        set(state.get("allowed_tables", [])),
        _state_artifact_relations(state),
        os.getenv("DATABASE_URL"),
    )


def _production_sql_runner(query, state):
    analysis = state.get("analysis", {})
    return run_managed_sql(
        query,
        owner_id=state.get("owner_id", ""),
        allowed_tables=set(state.get("allowed_tables", [])),
        allowed_artifacts=_state_artifact_relations(state),
        estimated_rows=analysis.get("planner", {}).get("plan_rows", 0),
    )


def build_query_graph(
    llm_call: LlmCall = call_groq,
    sql_runner: SqlRunner | None = None,
    checkpointer=None,
    statistics_runner=None,
    chart_runner=None,
    schema_runner=None,
    sql_analyzer=None,
):
    if sql_runner is not None:
        original_runner = sql_runner
        sql_runner = lambda query, _state: original_runner(query)
        schema_runner = schema_runner or (
            lambda _question, _state: {"context": "", "allowed_tables": []}
        )
        sql_analyzer = sql_analyzer or (
            lambda query, _state: {"status": "safe", **analyze_sql(query)}
        )
    else:
        sql_runner = _production_sql_runner
        schema_runner = schema_runner or _production_schema_runner
        sql_analyzer = sql_analyzer or _production_analyzer
    statistics_runner = statistics_runner or run_statistics.invoke
    chart_runner = chart_runner or run_chart.invoke
    remember = make_remember(llm_call)

    graph = StateGraph(QueryState)
    nodes = {
        "begin": begin_turn,
        "inspect_schema": make_inspect_schema(schema_runner),
        "agent": make_assess_result(llm_call, remember),
        "sql": make_execute_query(sql_runner, sql_analyzer),
        "statistics": make_calculate_statistic(statistics_runner, remember),
        "chart": make_create_plot(chart_runner, remember),
        "partial": partial_answer,
    }
    for name, node in nodes.items():
        graph.add_node(name, logged_node(name, node))

    graph.add_edge(START, "begin")
    graph.add_edge("begin", "inspect_schema")
    graph.add_conditional_edges(
        "inspect_schema", after_inspection, {"agent": "agent", "partial": "partial"}
    )
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {
            "done": END,
            "sql": "sql",
            "statistics": "statistics",
            "chart": "chart",
            "partial": "partial",
        },
    )
    graph.add_conditional_edges("sql", after_sql, {"agent": "agent", "partial": "partial"})
    graph.add_conditional_edges(
        "statistics",
        after_statistics,
        {"done": END, "agent": "agent", "partial": "partial"},
    )
    graph.add_conditional_edges(
        "chart", after_chart, {"done": END, "agent": "agent", "partial": "partial"}
    )
    graph.add_edge("partial", END)
    return graph.compile(checkpointer=checkpointer)


def _default_checkpointer():
    global _checkpoint_pool
    database_url = os.getenv("CHECKPOINT_DATABASE_URL")
    if not database_url:
        return MemorySaver()
    from langgraph.checkpoint.postgres import PostgresSaver

    _checkpoint_pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=int(os.getenv("CHECKPOINT_POOL_SIZE", "5")),
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    saver = PostgresSaver(_checkpoint_pool)
    saver.setup()
    return saver


default_checkpointer = _default_checkpointer()
query_graph = build_query_graph(checkpointer=default_checkpointer)
