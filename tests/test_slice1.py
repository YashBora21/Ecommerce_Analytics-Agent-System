import json
import unittest
from unittest.mock import MagicMock, patch

from backend.agent.graph import build_query_graph
from backend.agent.llm import call_groq
from backend.agent.nodes.assess import make_assess_result
from backend.tools.chart import create_chart, run_chart
from backend.tools.sql_analysis import analyze_sql, validate_sql
from backend.tools.statistics import calculate_statistics, run_statistics


def action(name: str, **arguments) -> str:
    return json.dumps({"action": name, "reason": "test", "arguments": arguments})


class ScriptedLlm:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        resp = self.responses.pop(0)
        return resp(messages) if callable(resp) else resp


class Slice1Tests(unittest.TestCase):
    def test_top_level_limit_detected_in_validate_sql(self) -> None:
        q_limit = "SELECT seller_id, SUM(price) AS revenue FROM order_items GROUP BY seller_id LIMIT 50"
        res_limit = validate_sql(q_limit, {"order_items"})
        self.assertTrue(res_limit["has_limit"])

        q_no_limit = "SELECT seller_id, SUM(price) AS revenue FROM order_items GROUP BY seller_id"
        res_no_limit = validate_sql(q_no_limit, {"order_items"})
        self.assertFalse(res_no_limit["has_limit"])

        q_cte_limit = "WITH top_sellers AS (SELECT seller_id FROM order_items LIMIT 50) SELECT * FROM top_sellers"
        res_cte = validate_sql(q_cte_limit, {"order_items"})
        self.assertFalse(res_cte["has_limit"])

    def test_cross_join_one_row_cte_allowed_and_huge_join_rejected(self) -> None:
        query_one_row = (
            "WITH cutoff AS (SELECT 1000 AS min_rev) "
            "SELECT s.seller_id, s.rev FROM sellers s CROSS JOIN cutoff c WHERE s.rev > c.min_rev"
        )
        validated = validate_sql(query_one_row, {"sellers"})
        self.assertIn("sellers", validated["tables"])

        with patch("backend.tools.sql_analysis.explain_postgres") as explain_mock:
            explain_mock.return_value = {
                "total_cost": 250.0,
                "plan_rows": 100,
                "plan_width": 32,
                "estimated_bytes": 3200,
            }
            res_safe = analyze_sql(query_one_row, {"sellers"}, database_url="postgres://fake")
            self.assertEqual(res_safe["status"], "safe")

        with patch("backend.tools.sql_analysis.explain_postgres") as explain_mock:
            explain_mock.return_value = {
                "total_cost": 8_800_000.0,
                "plan_rows": 10_000_000,
                "plan_width": 64,
                "estimated_bytes": 640_000_000,
            }
            res_huge = analyze_sql(
                "SELECT * FROM sellers s1 CROSS JOIN sellers s2",
                {"sellers"},
                database_url="postgres://fake",
            )
            self.assertEqual(res_huge["status"], "needs_decomposition")
            self.assertTrue(any("exceeds" in r for r in res_huge["reasons"]))

    def test_statistics_rejection_retries_twice_then_becomes_partial(self) -> None:
        stat_action = action("calculate_statistics", operation="mean", column="missing_col")
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT revenue FROM orders"),
            stat_action,
            stat_action,
        )
        result = build_query_graph(
            llm,
            sql_runner=lambda _sql: "revenue\n10\n20",
        ).invoke({"question": "Calculate stat"})

        self.assertEqual(result["statistics_tries"], 2)
        self.assertIn("incomplete", result["final_answer"].lower())

    def test_chart_rejection_retries_twice_then_becomes_partial(self) -> None:
        chart_action = action(
            "create_chart",
            chart_type="bar",
            x_column="state",
            y_column="missing_col",
            final_answer="Chart of revenue",
        )
        llm = ScriptedLlm(
            action("query_sql", sql="SELECT state FROM orders"),
            chart_action,
            chart_action,
        )
        result = build_query_graph(
            llm,
            sql_runner=lambda _sql: "state\nSP\nRJ",
        ).invoke({"question": "Build chart"})

        self.assertEqual(result["chart_tries"], 2)
        self.assertIn("could not build the chart", result["final_answer"].lower())

    def test_statistics_reads_artifact_via_sql(self) -> None:
        from datetime import datetime, timezone, timedelta
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(
                fetchone=lambda: {
                    "artifact_id": "00000000-0000-0000-0000-000000000001",
                    "relation_name": "result_test123",
                    "row_count": 3,
                    "byte_size": 100,
                    "columns_json": [{"name": "revenue"}],
                    "has_limit": False,
                    "expires_at": expires,
                }
            ),
            MagicMock(fetchone=lambda: {"value": 20.0, "count": 3}),
        ]

        with patch("backend.db.postgres.postgres_connection") as conn_mock:
            conn_mock.return_value.__enter__.return_value = mock_conn
            res = calculate_statistics(
                operation="mean",
                column="revenue",
                artifact_id="00000000-0000-0000-0000-000000000001",
                owner_id="owner1",
                database_url="postgres://fake",
            )
            self.assertTrue(res["ok"])
            self.assertEqual(res["value"], 20.0)

    def test_chart_reads_artifact_via_sql(self) -> None:
        from datetime import datetime, timezone, timedelta
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        mock_conn = MagicMock()
        mock_conn.execute.side_effect = [
            MagicMock(
                fetchone=lambda: {
                    "artifact_id": "00000000-0000-0000-0000-000000000001",
                    "relation_name": "result_test123",
                    "row_count": 2,
                    "byte_size": 100,
                    "columns_json": [{"name": "seller_id"}, {"name": "revenue"}],
                    "has_limit": False,
                    "expires_at": expires,
                }
            ),
            MagicMock(
                fetchall=lambda: [
                    {"seller_id": "s1", "revenue": 100.0},
                    {"seller_id": "s2", "revenue": 200.0},
                ]
            ),
        ]

        with patch("backend.db.postgres.postgres_connection") as conn_mock:
            conn_mock.return_value.__enter__.return_value = mock_conn
            res = create_chart(
                chart_type="bar",
                x_column="seller_id",
                y_column="revenue",
                title="Revenue by seller",
                artifact_id="00000000-0000-0000-0000-000000000001",
                owner_id="owner1",
                database_url="postgres://fake",
            )
            self.assertIn('"type":"bar"', res)

    def test_structured_flag_and_corrective_retry_on_markdown(self) -> None:
        calls = []

        def fake_llm(messages, structured_output=False, max_tokens=None, reasoning_effort=None):
            calls.append({
                "structured_output": structured_output,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            })
            if len(calls) == 1:
                return '```json\n{"action": "answer", "arguments": {"final_answer": "from markdown"}}\n```'
            return json.dumps({
                "decision": {
                    "action": "answer",
                    "reason": "done",
                    "arguments": {"final_answer": "Valid JSON answer."},
                }
            })

        node = make_assess_result(fake_llm, lambda state, ans: {"final_answer": ans})
        state = {
            "question": "What is total revenue?",
            "steps": 0,
            "evidence": [{"kind": "sql", "sql": "SELECT 1", "result": "total\n1"}],
            "artifacts": [],
        }
        res = node(state)
        self.assertEqual(res["action"]["name"], "answer")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["structured_output"])
        self.assertEqual(calls[0]["max_tokens"], 1600)
        self.assertEqual(calls[1]["max_tokens"], 3200)
        self.assertEqual(calls[0]["reasoning_effort"], "low")

    def test_retries_when_structured_output_hits_token_limit(self) -> None:
        limits = []

        def fake_llm(messages, structured_output=False, max_tokens=None, reasoning_effort=None):
            limits.append(max_tokens)
            if len(limits) == 1:
                raise RuntimeError(
                    "max completion tokens reached before generating a valid document"
                )
            return json.dumps(
                {
                    "decision": {
                        "action": "answer",
                        "reason": "done",
                        "arguments": {"final_answer": "Completed after retry."},
                    }
                }
            )

        node = make_assess_result(
            fake_llm,
            lambda state, answer: {"final_answer": answer},
        )
        result = node(
            {
                "question": "Run a complex analysis",
                "steps": 0,
                "evidence": [{"kind": "sql", "sql": "SELECT 1", "result": "value\n1"}],
                "artifacts": [],
            }
        )

        self.assertEqual(result["final_answer"], "Completed after retry.")
        self.assertEqual(limits, [1600, 3200])

    def test_uses_plain_answer_fallback_when_completed_evidence_has_invalid_json(self) -> None:
        llm = ScriptedLlm("not json", "still not json", "Grounded final answer.")
        node = make_assess_result(
            llm,
            lambda state, answer: {"final_answer": answer},
        )

        result = node(
            {
                "question": "Summarize the result",
                "steps": 0,
                "evidence": [{"kind": "sql", "sql": "SELECT 1", "result": "value\n1"}],
                "artifacts": [],
            }
        )

        self.assertEqual(result["final_answer"], "Grounded final answer.")
        self.assertEqual(result["action"]["name"], "answer")

    def test_statistical_answer_includes_tool_sample_count(self) -> None:
        llm = ScriptedLlm(action("answer", final_answer="Correlation is -0.33."))
        node = make_assess_result(
            llm,
            lambda state, answer: {"final_answer": answer},
        )

        result = node(
            {
                "question": "How strong is the relationship?",
                "steps": 0,
                "evidence": [
                    {
                        "kind": "statistics",
                        "sql": "STATISTICS correlation(x)",
                        "result": json.dumps({"count": 95824}),
                    }
                ],
                "artifacts": [],
            }
        )

        self.assertIn("Eligible rows used: 95,824.", result["final_answer"])

    def test_rejects_stale_answer_without_current_evidence(self) -> None:
        responses = iter([
            json.dumps({
                "decision": {
                    "action": "answer",
                    "reason": "reuse memory",
                    "arguments": {"final_answer": "The old answer."},
                }
            }),
            json.dumps({
                "decision": {
                    "action": "query_sql",
                    "reason": "get current evidence",
                    "arguments": {"sql": "SELECT COUNT(*) AS total FROM orders"},
                }
            }),
        ])

        node = make_assess_result(
            lambda *args, **kwargs: next(responses),
            lambda state, answer: {"final_answer": answer},
        )
        result = node({
            "question": "How many orders are there?",
            "steps": 0,
            "evidence": [],
            "history": [{"role": "assistant", "content": "There were 10."}],
        })

        self.assertEqual(result["action"]["name"], "query_sql")

if __name__ == "__main__":
    unittest.main()
