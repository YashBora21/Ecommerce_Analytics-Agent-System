import json
import unittest

from backend.agent.graph import build_query_graph


def action(name: str, **arguments) -> str:
    return json.dumps({"action": name, "reason": "test", "arguments": arguments})


class AutonomousSafetyTests(unittest.TestCase):
    def test_expensive_query_is_revised_before_execution(self) -> None:
        responses = iter([
            action("query_sql", sql="SELECT * FROM orders"),
            action("query_sql", sql="SELECT COUNT(*) AS order_count FROM orders"),
            action("answer", final_answer="There are 10 orders."),
        ])
        analyzer_calls = 0
        executed = []

        def analyzer(_query, _state):
            nonlocal analyzer_calls
            analyzer_calls += 1
            if analyzer_calls == 1:
                return {
                    "status": "needs_decomposition",
                    "reasons": ["estimated rows exceed the limit"],
                }
            return {"status": "safe", "sql_hash": "safe", "planner": {"plan_rows": 1}}

        def run_sql(query):
            executed.append(query)
            return "order_count\n10"

        result = build_query_graph(
            lambda _messages: next(responses),
            run_sql,
            sql_analyzer=analyzer,
        ).invoke({"question": "Analyze all orders"})

        self.assertEqual(executed, ["SELECT COUNT(*) AS order_count FROM orders"])
        self.assertEqual(result["final_answer"], "There are 10 orders.")

    def test_large_result_is_kept_as_an_artifact_reference(self) -> None:
        artifact = {
            "artifact_id": "00000000-0000-0000-0000-000000000001",
            "relation": "agent_results.result_test",
            "row_count": 1000000,
            "byte_size": 100,
            "columns": [{"name": "order_id", "type_oid": 25}],
            "preview": [{"order_id": "A"}],
            "profile": [],
            "complete": True,
            "expires_at": "2030-01-01T00:00:00+00:00",
            "sql_hash": "hash",
        }
        calls = 0

        def llm(messages):
            nonlocal calls
            calls += 1
            if calls == 1:
                return action("query_sql", sql="SELECT order_id FROM orders")
            self.assertIn("artifact_id", messages[-1]["content"])
            return action("answer", final_answer="The result is stored for follow-up analysis.")

        result = build_query_graph(
            llm,
            lambda _query: "[Artifact]\n" + json.dumps(artifact) + "\nPreview:\norder_id\nA",
        ).invoke({"question": "Keep all order IDs"})

        self.assertEqual(result["artifacts"][0]["row_count"], 1000000)
        self.assertNotIn("Preview", result["evidence"][0]["result"])


if __name__ == "__main__":
    unittest.main()
