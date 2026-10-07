from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agent.llm import call_groq
from backend.agent.memory import make_remember
from backend.agent.nodes.assess import make_assess_result
from backend.agent.nodes.sql import make_execute_query, make_generate_sql
from backend.agent.nodes.terminal import make_partial_answer
from backend.agent.nodes.tools import make_calculate_statistic, make_create_plot
from backend.agent.routing import (
    after_assessment,
    after_chart,
    after_execution,
    after_generation,
    after_tool,
)
from backend.agent.state import LlmCall, QueryState, SqlRunner
from backend.agent.telemetry import logged_node, reset_usage
from backend.tools.chart import run_chart
from backend.tools.sql import run_sql
from backend.tools.statistics import run_statistics


def begin_turn(_state: QueryState) -> QueryState:
    reset_usage()
    return {
        "action": {},
        "evidence": [],
        "chart": "",
        "error": "",
        "final_answer": "",
        "sql": "",
        "sql_tries": 0,
        "chart_tries": 0,
        "steps": 0,
    }


def build_query_graph(
    llm_call: LlmCall = call_groq,
    sql_runner: SqlRunner | None = None,
    checkpointer=None,
    statistics_runner=None,
    chart_runner=None,
):
    sql_runner = sql_runner or (lambda query: run_sql.invoke({"query": query}))
    statistics_runner = statistics_runner or run_statistics.invoke
    chart_runner = chart_runner or run_chart.invoke
    remember = make_remember(llm_call)

    graph = StateGraph(QueryState)
    graph.add_node("begin", logged_node("begin", begin_turn))
    graph.add_node("generate_sql", logged_node("generate_sql", make_generate_sql(llm_call)))
    graph.add_node("execute", logged_node("execute", make_execute_query(sql_runner)))
    graph.add_node("assess", logged_node("assess", make_assess_result(llm_call, remember)))
    graph.add_node("statistics", logged_node("statistics", make_calculate_statistic(statistics_runner)))
    graph.add_node("chart", logged_node("chart", make_create_plot(chart_runner, remember)))
    graph.add_node("partial", logged_node("partial", make_partial_answer(remember)))

    graph.add_edge(START, "begin")
    graph.add_edge("begin", "generate_sql")
    graph.add_conditional_edges(
        "generate_sql",
        after_generation,
        {
            "done": END,
            "execute": "execute",
            "generate_sql": "generate_sql",
            "partial": "partial",
        },
    )
    graph.add_conditional_edges(
        "execute",
        after_execution,
        {
            "assess": "assess",
            "generate_sql": "generate_sql",
            "partial": "partial",
        },
    )
    graph.add_conditional_edges(
        "assess",
        after_assessment,
        {
            "done": END,
            "generate_sql": "generate_sql",
            "statistics": "statistics",
            "chart": "chart",
            "partial": "partial",
        },
    )
    graph.add_conditional_edges(
        "statistics",
        after_tool,
        {"assess": "assess", "partial": "partial"},
    )
    graph.add_conditional_edges(
        "chart",
        after_chart,
        {"done": END, "assess": "assess", "partial": "partial"},
    )
    graph.add_edge("partial", END)

    return graph.compile(checkpointer=checkpointer)


default_checkpointer = MemorySaver()
query_graph = build_query_graph(checkpointer=default_checkpointer)