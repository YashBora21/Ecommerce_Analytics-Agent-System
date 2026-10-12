import json
import unittest

from langgraph.checkpoint.memory import MemorySaver

from backend.agent.agent import MAX_AGENT_STEPS, MAX_SQL_TRIES, ask
from backend.agent.graph import build_query_graph
from backend.agent.prompt import RESULT_ASSESSMENT_PROMPT


def action(name: str, **arguments) -> str:
    return json.dumps({"action": name, "reason": "test", "arguments": arguments})


class ScriptedLlm:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.messages = []

    def __call__(self, messages):
        self.messages.append(messages)
        if messages[0]["content"] != RESULT_ASSESSMENT_PROMPT:
            return "summary"
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        response = self.responses.pop(0)
        return response(messages) if callable(response) else response


class QueryGraphTests(unittest.TestCase):
    def test_question_reaches_grounded_answer(self) -> None:
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT COUNT(*) AS orders FROM orders"),
            action("answer", final_answer="There are 10 orders."),
        )
        result = build_query_graph(llm, lambda _sql: "orders\n10").invoke(
            {"question": "How many orders are there?"}
        )
        self.assertEqual(result["final_answer"], "There are 10 orders.")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["steps"], 2)

    def test_agent_can_stop_without_sql(self) -> None:
        llm = ScriptedLlm(action("stop", final_answer="That cannot be answered from this database."))
        result = build_query_graph(
            llm, lambda _sql: self.fail("SQL should not run")
        ).invoke({"question": "Tell me about Python"})
        self.assertIn("cannot be answered", result["final_answer"])

    def test_agent_can_run_multiple_queries(self) -> None:
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT COUNT(*) AS orders FROM orders"),
            action("query_sql", sql="SELECT SUM(revenue) AS revenue FROM orders"),
            action("answer", final_answer="There are 10 orders and revenue is 100."),
        )

        def run_sql(query):
            return "orders\n10" if "COUNT" in query else "revenue\n100"

        result = build_query_graph(llm, run_sql).invoke(
            {"question": "Give order count and revenue"}
        )
        self.assertEqual(len(result["evidence"]), 2)
        self.assertEqual(result["steps"], 3)

    def test_agent_can_calculate_statistics(self) -> None:
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT revenue FROM orders"),
            action("calculate_statistics", operation="mean", column="revenue"),
            action("answer", final_answer="Average revenue is 20."),
        )
        result = build_query_graph(llm, lambda _sql: "revenue\n10\n20\n30").invoke(
            {"question": "What is average revenue?"}
        )
        self.assertEqual(result["final_answer"], "Average revenue is 20.")
        self.assertEqual(result["evidence"][-1]["kind"], "statistics")

    def test_agent_can_create_a_chart(self) -> None:
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT month, revenue FROM monthly_sales"),
            action(
                "create_chart",
                chart_type="line",
                x_column="month",
                y_column="revenue",
                title="Monthly revenue",
                final_answer="The chart shows monthly revenue.",
            ),
        )
        result = build_query_graph(
            llm,
            lambda _sql: "month,revenue\n2024-01,10\n2024-02,20",
            chart_runner=lambda _request: '{"data":[],"layout":{}}',
        ).invoke({"question": "Chart monthly revenue"})
        self.assertTrue(result["chart"])
        self.assertEqual(result["final_answer"], "The chart shows monthly revenue.")

    def test_missing_chart_columns_stop_after_one_retry(self) -> None:
        chart_action = action(
            "create_chart",
            chart_type="bar",
            x_column="state",
            y_column="missing",
            final_answer="Revenue by state.",
        )
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT state, revenue FROM sales"),
            chart_action,
            chart_action,
        )
        result = build_query_graph(
            llm,
            lambda _sql: "state,revenue\nSP,100",
            chart_runner=lambda _request: self.fail("Chart tool should not run"),
        ).invoke({"question": "Chart revenue by state"})
        self.assertEqual(result["chart_tries"], 2)
        self.assertIn("could not build the chart", result["final_answer"].lower())

    def test_invalid_agent_json_is_corrected_once(self) -> None:
        llm = ScriptedLlm(
            "not json",
            action("query_sql", sql="SELECT COUNT(*) AS orders FROM orders"),
            action("answer", final_answer="There are 10 orders."),
        )
        result = build_query_graph(llm, lambda _sql: "orders\n10").invoke(
            {"question": "Count orders"}
        )
        self.assertEqual(result["final_answer"], "There are 10 orders.")

    def test_duplicate_sql_is_rejected_then_corrected(self) -> None:
        first = "SELECT COUNT(*) AS orders FROM orders"
        second = "SELECT COUNT(DISTINCT order_id) AS orders FROM orders"
        llm = ScriptedLlm(
            action("query_sql", sql=first),
            action("query_sql", sql=first),
            action("query_sql", sql=second),
            action("answer", final_answer="There are 10 orders."),
        )
        executed = []

        def run_sql(query):
            executed.append(query)
            return "orders\n10"

        result = build_query_graph(llm, run_sql).invoke({"question": "Count orders"})
        self.assertEqual(executed, [first, second])
        self.assertEqual(result["final_answer"], "There are 10 orders.")

    def test_invalid_sql_stops_at_retry_limit(self) -> None:
        llm = ScriptedLlm(*[
            action("query_sql", sql="DROP TABLE orders") for _ in range(MAX_SQL_TRIES)
        ])
        result = build_query_graph(
            llm, lambda _sql: self.fail("Unsafe SQL should not execute")
        ).invoke({"question": "Delete orders"})
        self.assertEqual(result["sql_tries"], MAX_SQL_TRIES)
        self.assertIn("could not run a safe", result["final_answer"].lower())

    def test_sql_can_recover_on_the_third_retry(self) -> None:
        llm = ScriptedLlm(
            *[action("query_sql", sql="DROP TABLE orders") for _ in range(3)],
            action("query_sql", sql="SELECT COUNT(*) AS orders FROM orders"),
            action("answer", final_answer="There are 10 orders."),
        )

        result = build_query_graph(llm, lambda _sql: "orders\n10").invoke(
            {"question": "Count orders"}
        )

        self.assertEqual(result["final_answer"], "There are 10 orders.")

    def test_step_limit_returns_incomplete_evidence(self) -> None:
        llm = ScriptedLlm(*[
            action("query_sql", sql=f"SELECT {index} AS value")
            for index in range(MAX_AGENT_STEPS)
        ])
        result = build_query_graph(llm, lambda query: f"value\n{query.split()[1]}").invoke(
            {"question": "Keep querying forever"}
        )
        self.assertEqual(result["steps"], MAX_AGENT_STEPS)
        self.assertIn("incomplete", result["final_answer"].lower())

    def test_answer_at_step_limit_is_not_discarded(self) -> None:
        responses = [
            action("query_sql", sql=f"SELECT {index} AS value")
            for index in range(MAX_AGENT_STEPS - 1)
        ]
        responses.append(action("answer", final_answer="Finished at the limit."))
        result = build_query_graph(
            ScriptedLlm(*responses),
            lambda query: f"value\n{query.split()[1]}",
        ).invoke({"question": "Use the full budget"})
        self.assertEqual(result["steps"], MAX_AGENT_STEPS)
        self.assertEqual(result["final_answer"], "Finished at the limit.")

    def test_follow_up_receives_recent_context(self) -> None:
        def follow_up(messages):
            context = "\n".join(message["content"] for message in messages)
            self.assertIn("There are 10 orders", context)
            return action("answer", final_answer="The previous result was 10 orders.")

        llm = ScriptedLlm(
            action("query_sql", sql="SELECT COUNT(*) AS orders FROM orders"),
            action("answer", final_answer="There are 10 orders."),
            follow_up,
        )
        graph = build_query_graph(
            llm,
            lambda _sql: "orders\n10",
            checkpointer=MemorySaver(),
        )
        config = {"configurable": {"thread_id": "same-thread"}}
        graph.invoke({"question": "Count orders"}, config=config)
        result = graph.invoke({"question": "What was that result?"}, config=config)
        self.assertIn("previous result", result["final_answer"])

    def test_empty_question_returns_immediately(self) -> None:
        result = ask("   ", llm_call=lambda _messages: self.fail("LLM should not run"))
        self.assertEqual(result["final_answer"], "Question cannot be empty.")


if __name__ == "__main__":
    unittest.main()
