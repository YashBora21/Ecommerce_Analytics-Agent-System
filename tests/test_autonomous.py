import json
import unittest

from backend.agent.graph import build_query_graph


def action(name: str, **arguments) -> str:
    return json.dumps({"action": name, "reason": "test", "arguments": arguments})


class AutonomousSafetyTests(unittest.TestCase):
    def test_executes_the_analyzers_repaired_sql(self) -> None:
        responses = iter([
            action("query_sql", sql="SELECT value FROM source do"),
            action("answer", final_answer="Done."),
        ])
        executed = []

        result = build_query_graph(
            lambda _messages: next(responses),
            lambda query: executed.append(query) or "value\n1",
            sql_analyzer=lambda _query, _state: {
                "status": "safe",
                "sql_hash": "fixed",
                "has_limit": False,
                "executable_sql": 'SELECT value FROM source AS "do"',
            },
        ).invoke({"question": "Run it"})

        self.assertEqual(executed, ['SELECT value FROM source AS "do"'])
        self.assertEqual(result["final_answer"], "Done.")

    def test_complete_result_rejects_limit_then_retries(self) -> None:
        responses = iter([
            action(
                "query_sql",
                sql="SELECT seller_id FROM sellers LIMIT 50",
                result_scope="complete",
            ),
            action(
                "query_sql",
                sql="SELECT seller_id FROM sellers",
                result_scope="complete",
            ),
            action("answer", final_answer="The complete seller result is ready."),
        ])
        executed = []

        def analyzer(query, _state):
            return {
                "status": "safe",
                "sql_hash": query,
                "has_limit": "LIMIT" in query,
            }

        result = build_query_graph(
            lambda _messages: next(responses),
            lambda query: executed.append(query) or "seller_id\nA",
            sql_analyzer=analyzer,
        ).invoke({"question": "Return every seller"})

        self.assertEqual(executed, ["SELECT seller_id FROM sellers"])
        self.assertEqual(result["final_answer"], "The complete seller result is ready.")

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

    def test_statistics_uses_the_explicit_owned_artifact(self) -> None:
        artifact = {
            "artifact_id": "00000000-0000-0000-0000-000000000001",
            "relation": "agent_results.result_test",
            "row_count": 1000000,
            "byte_size": 100,
            "columns": [
                {"name": "payment_value", "type_oid": 701},
                {"name": "review_score", "type_oid": 701},
            ],
            "preview": [],
            "profile": [],
            "complete": True,
            "has_limit": False,
            "expires_at": "2030-01-01T00:00:00+00:00",
            "sql_hash": "hash",
        }
        responses = iter([
            action("query_sql", sql="SELECT payment_value, review_score FROM orders"),
            action(
                "calculate_statistics",
                operation="correlation",
                column="payment_value",
                second_column="review_score",
                artifact_id=artifact["artifact_id"],
            ),
            action("answer", final_answer="The correlation is 0.1."),
        ])
        requests = []

        result = build_query_graph(
            lambda _messages: next(responses),
            lambda _query: "[Artifact]\n" + json.dumps(artifact) + "\nPreview:\n",
            statistics_runner=lambda request: requests.append(request) or (
                '{"ok":true,"operation":"correlation","column":"payment_value",'
                '"second_column":"review_score","value":0.1,"count":1000000,"error":null}'
            ),
        ).invoke({"question": "Correlate the columns", "owner_id": "thread-1"})

        self.assertEqual(requests[0]["artifact_id"], artifact["artifact_id"])
        self.assertEqual(requests[0]["owner_id"], "thread-1")
        self.assertNotIn("csv_data", requests[0])
        self.assertEqual(result["final_answer"], "The correlation is 0.1.")

    def test_statistics_does_not_guess_an_artifact(self) -> None:
        artifact = {
            "artifact_id": "00000000-0000-0000-0000-000000000001",
            "relation": "agent_results.result_test",
            "row_count": 1000000,
            "byte_size": 100,
            "columns": [{"name": "revenue", "type_oid": 701}],
            "preview": [],
            "profile": [],
            "complete": True,
            "has_limit": False,
            "expires_at": "2030-01-01T00:00:00+00:00",
            "sql_hash": "hash",
        }
        responses = iter([
            action("query_sql", sql="SELECT revenue FROM orders"),
            action("calculate_statistics", operation="mean", column="revenue"),
            action(
                "calculate_statistics",
                operation="mean",
                column="revenue",
                artifact_id=artifact["artifact_id"],
            ),
            action("answer", final_answer="The mean is 10."),
        ])
        requests = []

        result = build_query_graph(
            lambda _messages: next(responses),
            lambda _query: "[Artifact]\n" + json.dumps(artifact) + "\nPreview:\n",
            statistics_runner=lambda request: requests.append(request) or (
                '{"ok":true,"operation":"mean","column":"revenue",'
                '"second_column":"","value":10.0,"count":1000000,"error":null}'
            ),
        ).invoke({"question": "Calculate the mean", "owner_id": "thread-1"})

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["artifact_id"], artifact["artifact_id"])
        self.assertEqual(result["final_answer"], "The mean is 10.")


if __name__ == "__main__":
    unittest.main()
