import json
import os
import re
import uuid
from typing import Callable, TypedDict

from groq import Groq, GroqError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.agent.prompt import (
    GUARDRAIL_SYSTEM_PROMPT,
    MEMORY_SUMMARY_PROMPT,
    RESULT_ASSESSMENT_PROMPT,
    SQL_SYSTEM_PROMPT,
)
from backend.tools.chart import run_chart
from backend.tools.sql import run_sql
from backend.tools.statistics import run_statistics


Message = dict[str, str]
Evidence = dict[str, str]
LlmCall = Callable[[list[Message]], str]
SqlRunner = Callable[[str], str]
MAX_SQL_TRIES = 3
MAX_AGENT_STEPS = 5
MAX_RECENT_MESSAGES = 6

GREETING_REPLY = (
    "Hi! I can answer questions about the anonymized ecommerce sample, including "
    "orders, revenue, products, customers, sellers, payments, reviews, and delivery."
)
OUT_OF_SCOPE_REPLY = (
    "That question is outside this ecommerce dataset. I can help with orders, "
    "revenue, products, customers, sellers, payments, reviews, and delivery."
)


class QueryState(TypedDict, total=False):
    question: str
    history: list[Message]
    history_summary: str
    last_sql: str
    sql: str
    sql_result: str
    query_results: list[Evidence]
    next_query_goal: str
    next_action: str
    final_answer: str
    error: str
    sql_tries: int
    agent_steps: int
    is_in_scope: bool
    is_greeting: bool
    guardrail_reason: str
    statistics_request: dict
    chart_request: dict
    chart: str


def call_groq(messages: list[Message]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    options = {}
    json_prompts = {
        GUARDRAIL_SYSTEM_PROMPT,
        RESULT_ASSESSMENT_PROMPT,
        SQL_SYSTEM_PROMPT,
    }
    if messages[0]["content"] in json_prompts:
        options["response_format"] = {"type": "json_object"}

    response = Groq(api_key=api_key).chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        messages=messages,
        temperature=0,
        **options,
    )
    return response.choices[0].message.content or ""


def is_greeting(question: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", question.lower()).strip()
    return normalized in {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
    }


def build_query_graph(
    llm_call: LlmCall = call_groq,
    sql_runner: SqlRunner | None = None,
    checkpointer=None,
    statistics_runner=None,
    chart_runner=None,
):
    sql_runner = sql_runner or (lambda query: run_sql.invoke({"query": query}))
    statistics_runner = statistics_runner or (lambda request: run_statistics.invoke(request))
    chart_runner = chart_runner or (lambda request: run_chart.invoke(request))

    def format_evidence(state: QueryState) -> str:
        evidence = state.get("query_results", [])
        return "\n\n".join(
            f"SQL {index}: {item['sql']}\nResult {index}:\n{item['result']}"
            for index, item in enumerate(evidence, start=1)
        )

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
        if state.get("query_results"):
            memory["last_sql"] = state["query_results"][-1]["sql"]
        return memory

    def guardrail(state: QueryState) -> QueryState:
        if is_greeting(state["question"]):
            return {"is_greeting": True, "is_in_scope": False, "error": ""}

        context = recent_context(state)
        user_content = state["question"]
        if context:
            user_content = f"{context}\n\nCurrent message: {user_content}"
        try:
            decision = json.loads(
                llm_call(
                    [
                        {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ]
                )
            )
            in_scope = decision.get("is_in_scope")
            greeting = decision.get("is_greeting")
            reason = decision.get("reason")
            if not isinstance(in_scope, bool) or not isinstance(greeting, bool):
                raise ValueError("guardrail decisions must be booleans")
            if not isinstance(reason, str):
                raise ValueError("guardrail reason must be text")
            return {
                "is_in_scope": in_scope,
                "is_greeting": greeting,
                "guardrail_reason": reason,
                "error": "",
            }
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError, GroqError) as error:
            return {"error": f"Could not classify question: {error}"}

    def generate_sql(state: QueryState) -> QueryState:
        tries = state.get("sql_tries", 0) + 1
        steps = state.get("agent_steps", 0) + 1
        user_content = state["question"]
        if state.get("next_query_goal"):
            user_content += f"\n\nAdditional query goal: {state['next_query_goal']}"
            user_content += f"\n\nEvidence already collected:\n{format_evidence(state)}"
        if state.get("error") and state.get("sql"):
            user_content += (
                f"\n\nPrevious SQL: {state['sql']}\n"
                f"Validation error: {state['error']}\nReturn corrected SQL."
            )

        messages: list[Message] = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
        context = recent_context(state)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": user_content})
        try:
            plan = json.loads(llm_call(messages))
            sql = plan.get("sql")
            if not isinstance(sql, str) or not sql.strip():
                raise ValueError("Groq response did not contain SQL")
            return {
                "sql": sql.strip(),
                "sql_tries": tries,
                "agent_steps": steps,
                "error": "",
            }
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError, GroqError) as error:
            return {
                "sql_tries": tries,
                "agent_steps": steps,
                "error": f"Could not generate SQL: {error}",
            }

    def execute_query(state: QueryState) -> QueryState:
        result = sql_runner(state["sql"])
        if result.startswith("Rejected:"):
            return {"error": result}
        evidence = list(state.get("query_results", []))
        evidence.append({"sql": state["sql"], "result": result})
        return {
            "sql_result": result,
            "query_results": evidence,
            "sql_tries": 0,
            "error": "",
        }

    def assess_result(state: QueryState) -> QueryState:
        steps = state.get("agent_steps", 0) + 1
        prompt = (
            f"Question: {state['question']}\n\n"
            f"SQL evidence:\n{format_evidence(state)}\n\n"
            f"Chart already generated: {'yes' if state.get('chart') else 'no'}"
        )
        try:
            decision = json.loads(
                llm_call(
                    [
                        {"role": "system", "content": RESULT_ASSESSMENT_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
            )
            action = decision.get("action")
            reason = decision.get("reason")
            arguments = decision.get("arguments")
            if action not in {"answer", "query_sql", "calculate_statistics", "create_chart"}:
                raise ValueError("invalid next action")
            if not isinstance(reason, str) or not isinstance(arguments, dict):
                raise ValueError("action reason and arguments are invalid")

            if action == "answer":
                answer = arguments.get("final_answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("final answer is empty")
                return {
                    **remember(state, answer.strip()),
                    "next_action": action,
                    "agent_steps": steps,
                    "error": "",
                }

            if action == "calculate_statistics":
                operation = arguments.get("operation", "")
                column = arguments.get("column", "")
                second_column = arguments.get("second_column", "")
                percentile = arguments.get("percentile")
                if not all(isinstance(value, str) for value in (operation, column, second_column)):
                    raise ValueError("statistics arguments are invalid")
                if not operation.strip() or not column.strip():
                    raise ValueError("statistics operation and column are required")
                return {
                    "next_action": action,
                    "statistics_request": {
                        "csv_data": state["sql_result"],
                        "operation": operation.strip(),
                        "column": column.strip(),
                        "second_column": second_column.strip(),
                        "percentile": percentile,
                    },
                    "agent_steps": steps,
                    "error": "",
                }

            if action == "create_chart":
                chart_type = arguments.get("chart_type", "")
                x_column = arguments.get("x_column", "")
                y_column = arguments.get("y_column", "")
                title = arguments.get("title", "")
                if not all(
                    isinstance(value, str)
                    for value in (chart_type, x_column, y_column, title)
                ):
                    raise ValueError("chart arguments are invalid")
                if not chart_type.strip() or not x_column.strip() or not y_column.strip():
                    raise ValueError("chart type, x column, and y column are required")
                return {
                    "next_action": action,
                    "chart_request": {
                        "csv_data": state["sql_result"],
                        "chart_type": chart_type.strip(),
                        "x_column": x_column.strip(),
                        "y_column": y_column.strip(),
                        "title": title.strip(),
                    },
                    "agent_steps": steps,
                    "error": "",
                }

            goal = arguments.get("next_query_goal")
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError("next query goal is empty")
            return {
                "next_action": action,
                "next_query_goal": goal.strip(),
                "agent_steps": steps,
                "error": "",
            }
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError, GroqError) as error:
            return {
                "agent_steps": steps,
                "error": f"Could not assess query results: {error}",
            }

    def calculate_statistic(state: QueryState) -> QueryState:
        result = statistics_runner(state["statistics_request"])
        if result.startswith("Rejected:"):
            return {"error": result}
        evidence = list(state.get("query_results", []))
        request = state["statistics_request"]
        label = f"STATISTICS {request['operation']}({request['column']})"
        evidence.append({"sql": label, "result": result})
        return {
            "query_results": evidence,
            "next_action": "",
            "statistics_request": {},
            "chart_request": {},
            "chart": "",
            "error": "",
        }

    def create_plot(state: QueryState) -> QueryState:
        result = chart_runner(state["chart_request"])
        if result.startswith("Rejected:"):
            return {"error": result}
        return {
            "chart": result,
            "next_action": "",
            "chart_request": {},
            "error": "",
        }

    def greeting(state: QueryState) -> QueryState:
        return remember(state, GREETING_REPLY)

    def out_of_scope(state: QueryState) -> QueryState:
        return remember(state, OUT_OF_SCOPE_REPLY)

    def failure(state: QueryState) -> QueryState:
        return remember(state, state["error"])

    def partial_answer(state: QueryState) -> QueryState:
        evidence = format_evidence(state)[:3000]
        if evidence:
            reason = f" Reason: {state['error']}" if state.get("error") else ""
            answer = (
                "I could not fully complete the analysis within the allowed steps."
                f"{reason} Here is the evidence collected so far:\n{evidence}"
            )
        else:
            answer = (
                "I could not build a valid query within the allowed attempts. "
                "Please rephrase the question."
            )
        return remember(state, answer)

    def after_guardrail(state: QueryState) -> str:
        if state.get("error"):
            return "failure"
        if state.get("is_greeting"):
            return "greeting"
        return "generate_sql" if state.get("is_in_scope") else "out_of_scope"

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
        if state.get("next_action") == "answer":
            return "done"
        if state.get("agent_steps", 0) >= MAX_AGENT_STEPS:
            return "partial"
        if state.get("next_action") == "calculate_statistics":
            return "statistics"
        if state.get("next_action") == "create_chart":
            return "chart"
        return "generate_sql"

    def after_tool(state: QueryState) -> str:
        return "partial" if state.get("error") else "assess"

    graph = StateGraph(QueryState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute", execute_query)
    graph.add_node("assess", assess_result)
    graph.add_node("statistics", calculate_statistic)
    graph.add_node("chart", create_plot)
    graph.add_node("greeting", greeting)
    graph.add_node("out_of_scope", out_of_scope)
    graph.add_node("failure", failure)
    graph.add_node("partial", partial_answer)
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        after_guardrail,
        {
            "generate_sql": "generate_sql",
            "greeting": "greeting",
            "out_of_scope": "out_of_scope",
            "failure": "failure",
        },
    )
    graph.add_conditional_edges(
        "generate_sql",
        after_generation,
        {"execute": "execute", "generate_sql": "generate_sql", "partial": "partial"},
    )
    graph.add_conditional_edges(
        "execute",
        after_execution,
        {"assess": "assess", "generate_sql": "generate_sql", "partial": "partial"},
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
        after_tool,
        {"assess": "assess", "partial": "partial"},
    )
    for terminal in ("greeting", "out_of_scope", "failure", "partial"):
        graph.add_edge(terminal, END)
    return graph.compile(checkpointer=checkpointer)


default_checkpointer = MemorySaver()
query_graph = build_query_graph(checkpointer=default_checkpointer)


def ask(
    question: str,
    thread_id: str | None = None,
    llm_call: LlmCall | None = None,
) -> QueryState:
    if not question or not question.strip():
        return {"question": question, "final_answer": "Question cannot be empty."}

    graph = (
        build_query_graph(llm_call=llm_call, checkpointer=default_checkpointer)
        if llm_call
        else query_graph
    )
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    return graph.invoke(
        {
            "question": question.strip(),
            "sql_tries": 0,
            "agent_steps": 0,
            "query_results": [],
            "next_query_goal": "",
            "next_action": "",
            "statistics_request": {},
            "chart_request": {},
            "chart": "",
            "error": "",
        },
        config=config,
    )


