import json
import unittest

from langgraph.checkpoint.memory import MemorySaver

from backend.agent.agent import MAX_AGENT_STEPS, MAX_SQL_TRIES, ask, build_query_graph
from backend.agent.prompt import (
    GUARDRAIL_SYSTEM_PROMPT,
    MEMORY_SUMMARY_PROMPT,
    RESULT_ASSESSMENT_PROMPT,
    SQL_SYSTEM_PROMPT,
)


def guardrail_json(in_scope=True) -> str:
    return json.dumps(
        {
            "is_in_scope": in_scope,
            "is_greeting": False,
            "reason": "Ecommerce question" if in_scope else "General knowledge",
        }
    )


def assessment_json(action, answer="", goal="") -> str:
    arguments = {"final_answer": answer} if action == "answer" else {"next_query_goal": goal}
    return json.dumps(
        {
            "action": action,
            "reason": "Enough evidence" if action == "answer" else "More data needed",
            "arguments": arguments,
        }
    )


class QueryGraphTests(unittest.TestCase):
    def test_question_reaches_grounded_answer(self) -> None:
        def fake_llm(messages):
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps({"sql": "SELECT COUNT(*) AS n FROM orders", "reason": "count"})
            self.assertEqual(system, RESULT_ASSESSMENT_PROMPT)
            self.assertIn("n\n5000", messages[1]["content"])
            return assessment_json("answer", "The sample contains 5,000 orders.")

        result = build_query_graph(fake_llm, lambda query: "n\n5000").invoke(
            {"question": "How many orders are there?"}
        )

        self.assertEqual(result["final_answer"], "The sample contains 5,000 orders.")
        self.assertEqual(result["agent_steps"], 2)
        self.assertEqual(len(result["query_results"]), 1)

    def test_agent_can_request_second_sql_query(self) -> None:
        planner_calls = 0
        assessment_calls = 0

        def fake_llm(messages):
            nonlocal planner_calls, assessment_calls
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                planner_calls += 1
                if planner_calls == 1:
                    return json.dumps({"sql": "SELECT SUM(order_value) AS revenue FROM orders", "reason": "revenue"})
                self.assertIn("Additional query goal", messages[-1]["content"])
                self.assertIn("Evidence already collected", messages[-1]["content"])
                return json.dumps({"sql": "SELECT COUNT(*) AS orders FROM orders", "reason": "count"})
            assessment_calls += 1
            if assessment_calls == 1:
                return assessment_json("query_sql", goal="Get the order count")
            self.assertIn("SQL 1", messages[1]["content"])
            self.assertIn("SQL 2", messages[1]["content"])
            return assessment_json("answer", "Revenue is 100 across 5 orders.")

        def fake_sql(query):
            return "revenue\n100" if "SUM" in query else "orders\n5"

        result = build_query_graph(fake_llm, fake_sql).invoke(
            {"question": "What are revenue and order count?"}
        )

        self.assertEqual(result["final_answer"], "Revenue is 100 across 5 orders.")
        self.assertEqual(len(result["query_results"]), 2)
        self.assertEqual(result["agent_steps"], 4)

    def test_agent_can_calculate_statistics_from_sql_rows(self) -> None:
        assessment_calls = 0

        def fake_llm(messages):
            nonlocal assessment_calls
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps({"sql": "SELECT delivery_days FROM orders WHERE delivery_days >= 0", "reason": "raw values"})
            assessment_calls += 1
            if assessment_calls == 1:
                return json.dumps(
                    {
                        "action": "calculate_statistics",
                        "reason": "Median is required",
                        "arguments": {
                            "operation": "median",
                            "column": "delivery_days",
                            "second_column": "",
                            "percentile": None,
                        },
                    }
                )
            self.assertIn("STATISTICS median(delivery_days)", messages[1]["content"])
            return assessment_json("answer", "Median delivery time is 3 days.")

        def fake_statistics(request):
            self.assertEqual(request["csv_data"], "delivery_days\n2\n3\n8")
            self.assertEqual(request["operation"], "median")
            return '{"ok":true,"operation":"median","value":3.0,"count":3}'

        result = build_query_graph(
            fake_llm,
            lambda query: "delivery_days\n2\n3\n8",
            statistics_runner=fake_statistics,
        ).invoke({"question": "What is median delivery time?"})

        self.assertEqual(result["final_answer"], "Median delivery time is 3 days.")
        self.assertEqual(len(result["query_results"]), 2)
        self.assertEqual(result["agent_steps"], 3)

    def test_agent_can_create_chart_from_sql_rows(self) -> None:
        assessment_calls = 0

        def fake_llm(messages):
            nonlocal assessment_calls
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps(
                    {
                        "sql": "SELECT customer_state, SUM(order_value) AS revenue FROM orders GROUP BY customer_state",
                        "reason": "revenue by state",
                    }
                )
            assessment_calls += 1
            if assessment_calls == 1:
                return json.dumps(
                    {
                        "action": "create_chart",
                        "reason": "The user requested a chart",
                        "arguments": {
                            "chart_type": "bar",
                            "x_column": "customer_state",
                            "y_column": "revenue",
                            "title": "Revenue by state",
                        },
                    }
                )
            self.assertIn("Chart already generated: yes", messages[1]["content"])
            return assessment_json("answer", "Revenue by state is shown in the chart.")

        def fake_chart(request):
            self.assertEqual(request["csv_data"], "customer_state,revenue\nSao Paulo,100")
            self.assertEqual(request["chart_type"], "bar")
            return '{"data":[{"type":"bar"}],"layout":{}}'

        result = build_query_graph(
            fake_llm,
            lambda query: "customer_state,revenue\nSao Paulo,100",
            chart_runner=fake_chart,
        ).invoke({"question": "Chart revenue by state"})

        self.assertEqual(result["final_answer"], "Revenue by state is shown in the chart.")
        self.assertIn('"type":"bar"', result["chart"])
        self.assertEqual(result["agent_steps"], 3)

    def test_invalid_next_action_is_rejected(self) -> None:
        def fake_llm(messages):
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps({"sql": "SELECT COUNT(*) AS n FROM orders", "reason": "count"})
            return json.dumps({"action": "delete_data", "reason": "bad", "arguments": {}})

        result = build_query_graph(fake_llm, lambda query: "n\n5000").invoke(
            {"question": "How many orders?"}
        )

        self.assertIn("invalid next action", result["final_answer"])

    def test_retry_counter_resets_after_successful_round(self) -> None:
        planner_calls = 0
        assessment_calls = 0

        def fake_llm(messages):
            nonlocal planner_calls, assessment_calls
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                planner_calls += 1
                sql = {
                    1: "SELECT SUM(order_value) AS revenue FROM orders",
                    2: "SELECT missing FROM orders",
                    3: "SELECT COUNT(*) AS orders FROM orders",
                }[planner_calls]
                return json.dumps({"sql": sql, "reason": "plan"})
            assessment_calls += 1
            if assessment_calls == 1:
                return assessment_json("query_sql", goal="Get order count")
            return assessment_json("answer", "Completed with both results.")

        def fake_sql(query):
            if "missing" in query:
                return "Rejected: no such column: missing"
            return "revenue\n100" if "SUM" in query else "orders\n5"

        result = build_query_graph(fake_llm, fake_sql).invoke({"question": "Analyze orders"})

        self.assertEqual(result["final_answer"], "Completed with both results.")
        self.assertEqual(result["sql_tries"], 0)
        self.assertEqual(len(result["query_results"]), 2)
        self.assertEqual(result["agent_steps"], 5)

    def test_step_cap_returns_collected_evidence(self) -> None:
        def fake_llm(messages):
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps({"sql": "SELECT COUNT(*) AS orders FROM orders", "reason": "count"})
            return assessment_json("query_sql", goal="Get another metric")

        result = build_query_graph(fake_llm, lambda query: "orders\n5000").invoke(
            {"question": "Complex analysis", "agent_steps": MAX_AGENT_STEPS - 1}
        )

        self.assertIn("evidence collected so far", result["final_answer"])
        self.assertIn("orders\n5000", result["final_answer"])

    def test_three_invalid_queries_give_up(self) -> None:
        planner_calls = 0

        def fake_llm(messages):
            nonlocal planner_calls
            if messages[0]["content"] == GUARDRAIL_SYSTEM_PROMPT:
                return guardrail_json()
            planner_calls += 1
            return json.dumps({"sql": "DELETE FROM orders", "reason": "invalid"})

        result = build_query_graph(
            fake_llm,
            lambda query: "Rejected: Only SELECT queries are allowed",
        ).invoke({"question": "Count orders"})

        self.assertEqual(planner_calls, MAX_SQL_TRIES)
        self.assertIn("rephrase", result["final_answer"])

    def test_invalid_guardrail_fails_closed(self) -> None:
        result = build_query_graph(
            lambda messages: json.dumps(
                {"is_in_scope": "false", "is_greeting": False, "reason": "bad"}
            ),
            lambda query: self.fail("SQL should not run"),
        ).invoke({"question": "Weather?"})

        self.assertIn("Could not classify question", result["final_answer"])

    def test_greeting_and_out_of_scope_skip_sql(self) -> None:
        greeting = build_query_graph(
            lambda messages: self.fail("LLM should not run for greeting"),
            lambda query: self.fail("SQL should not run"),
        ).invoke({"question": "Hello!"})
        self.assertIn("anonymized ecommerce sample", greeting["final_answer"])

        outside = build_query_graph(
            lambda messages: guardrail_json(False),
            lambda query: self.fail("SQL should not run"),
        ).invoke({"question": "What is the capital of France?"})
        self.assertIn("outside this ecommerce dataset", outside["final_answer"])


class ThreadMemoryTests(unittest.TestCase):
    def make_llm(self, guardrail_inputs=None, summary_counter=None):
        def fake_llm(messages):
            system = messages[0]["content"]
            if system == GUARDRAIL_SYSTEM_PROMPT:
                if guardrail_inputs is not None:
                    guardrail_inputs.append(messages[1]["content"])
                return guardrail_json()
            if system == SQL_SYSTEM_PROMPT:
                return json.dumps({"sql": "SELECT COUNT(*) AS n FROM orders", "reason": "count"})
            if system == MEMORY_SUMMARY_PROMPT:
                if summary_counter is not None:
                    summary_counter[0] += 1
                return "Earlier questions counted orders in the sample."
            return assessment_json("answer", "There are 5,000 orders.")

        return fake_llm

    def test_follow_up_gets_recent_history_and_last_sql(self) -> None:
        inputs = []
        graph = build_query_graph(
            self.make_llm(guardrail_inputs=inputs),
            lambda query: "n\n5000",
            MemorySaver(),
        )
        config = {"configurable": {"thread_id": "follow-up"}}

        graph.invoke({"question": "How many orders?"}, config=config)
        result = graph.invoke({"question": "Now break that down by state."}, config=config)

        self.assertIn("How many orders?", inputs[1])
        self.assertIn("Last successful SQL", inputs[1])
        self.assertEqual(len(result["history"]), 4)

    def test_older_messages_roll_into_one_summary(self) -> None:
        summary_counter = [0]
        graph = build_query_graph(
            self.make_llm(summary_counter=summary_counter),
            lambda query: "n\n5000",
            MemorySaver(),
        )
        config = {"configurable": {"thread_id": "summary"}}

        result = None
        for number in range(4):
            result = graph.invoke({"question": f"Order question {number}"}, config=config)

        self.assertEqual(summary_counter[0], 1)
        self.assertEqual(len(result["history"]), 6)
        self.assertIn("Earlier questions counted orders", result["history_summary"])

    def test_empty_question_returns_immediately(self) -> None:
        self.assertEqual(ask(" ")["final_answer"], "Question cannot be empty.")


if __name__ == "__main__":
    unittest.main()

