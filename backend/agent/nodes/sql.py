from typing import Annotated, Literal

from groq import GroqError
from pydantic import BaseModel, StringConstraints, ValidationError, model_validator

from backend.agent.memory import format_evidence, recent_context
from backend.agent.prompt import SQL_SYSTEM_PROMPT
from backend.agent.state import LlmCall, Message, QueryState, SqlRunner


Text = Annotated[str, StringConstraints(strip_whitespace=True)]


class SqlPlan(BaseModel):
    action: Literal["query", "answer"] = "query"
    sql: Text = ""
    reason: Text = ""
    final_answer: Text = ""

    @model_validator(mode="after")
    def required_content(self):
        if self.action == "query" and not self.sql:
            raise ValueError("SQL is empty")
        if self.action == "answer" and not self.final_answer:
            raise ValueError("direct answer is empty")
        return self


def make_generate_sql(llm_call: LlmCall):
    def generate_sql(state: QueryState) -> QueryState:
        tries = state.get("sql_tries", 0) + 1
        steps = state.get("steps", 0) + 1
        content = state["question"]
        action = state.get("action", {})
        goal = action.get("args", {}).get("next_query_goal")
        if goal:
            content += f"""

Additional query goal: {goal}

Evidence already collected:
{format_evidence(state)}"""
        if state.get("error") and state.get("sql"):
            content += f"""

Previous SQL: {state['sql']}
Validation error: {state['error']}
Return corrected SQL."""
        messages: list[Message] = [{"role": "system", "content": SQL_SYSTEM_PROMPT}]
        context = recent_context(state)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": content})
        try:
            plan = SqlPlan.model_validate_json(llm_call(messages))
            if plan.action == "answer":
                return {
                    "final_answer": plan.final_answer,
                    "steps": steps,
                    "error": "",
                }

            sql = plan.sql
            previous_sql = {
                item["sql"]
                for item in state.get("evidence", [])
                if item.get("kind", "sql") == "sql"
            }
            if sql in previous_sql:
                return {
                    "sql": sql,
                    "sql_tries": tries,
                    "steps": steps,
                    "error": (
                        "Rejected: this exact query was already run. Choose a "
                        "different grouping, metric, or dimension."
                    ),
                }
            return {"sql": sql, "sql_tries": tries, "steps": steps, "error": ""}
        except (ValidationError, RuntimeError, GroqError) as error:
            return {
                "sql_tries": tries,
                "steps": steps,
                "error": f"Could not generate SQL: {error}",
            }

    return generate_sql


def make_execute_query(sql_runner: SqlRunner):
    def execute_query(state: QueryState) -> QueryState:
        result = sql_runner(state["sql"])
        if result.startswith("Rejected:"):
            return {"error": result}
        evidence = list(state.get("evidence", []))
        evidence.append({"kind": "sql", "sql": state["sql"], "result": result})
        return {"evidence": evidence, "sql_tries": 0, "error": ""}

    return execute_query

