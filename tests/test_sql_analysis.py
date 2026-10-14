import unittest
from unittest.mock import patch

import psycopg

from backend.tools.sql_analysis import analyze_sql, validate_sql


class SqlAnalysisTests(unittest.TestCase):
    def test_rejects_injection_and_write_statements(self) -> None:
        for query in (
            "SELECT * FROM orders; DROP TABLE orders",
            "DELETE FROM orders",
            "SELECT pg_sleep(10)",
            "SELECT * FROM orders JOIN customers",
        ):
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_sql(query, {"orders", "customers"})

    def test_allows_comments_inside_a_safe_query(self) -> None:
        result = validate_sql(
            "SELECT order_id -- useful label\nFROM orders /* source table */",
            {"orders"},
        )

        self.assertEqual(result["tables"], ["orders"])

    def test_routes_cross_join_for_decomposition_and_checks_limit(self) -> None:
        cross_join = analyze_sql(
            "SELECT * FROM orders CROSS JOIN customers",
            {"orders", "customers"},
        )
        self.assertEqual(cross_join["status"], "needs_decomposition")
        self.assertIn("explicit join key", cross_join["reasons"][0])

        limited = validate_sql("SELECT * FROM orders LIMIT 10", {"orders"})
        self.assertTrue(limited["has_limit"])

        cte_limited = validate_sql(
            "WITH sub AS (SELECT * FROM orders LIMIT 10) SELECT * FROM sub",
            {"orders"},
        )
        self.assertFalse(cte_limited["has_limit"])

    def test_allows_scalar_threshold_cte_without_join_predicate(self) -> None:
        query = (
            "WITH seller_metrics AS ("
            "SELECT seller_id, SUM(price) AS revenue FROM order_items GROUP BY seller_id"
            "), threshold AS ("
            "SELECT percentile_cont(0.9) WITHIN GROUP (ORDER BY revenue) AS p90 "
            "FROM seller_metrics"
            ") SELECT sm.seller_id FROM seller_metrics sm, threshold t "
            "WHERE sm.revenue >= t.p90"
        )

        result = analyze_sql(query, {"order_items"})

        self.assertEqual(result["status"], "safe")
        self.assertFalse(result["risky_cross_join"])

    def test_allows_scalar_date_cte_joined_on_true(self) -> None:
        query = (
            "WITH latest AS (SELECT MAX(order_date) AS max_date FROM orders) "
            "SELECT COUNT(*) FROM orders o JOIN latest l ON TRUE "
            "WHERE o.order_date >= date_trunc('month', l.max_date)"
        )

        result = analyze_sql(query, {"orders"})

        self.assertEqual(result["status"], "safe")

    def test_repairs_a_server_rejected_table_alias(self) -> None:
        query = (
            "WITH delivered AS (SELECT order_id FROM orders) "
            "SELECT oi.order_id FROM order_items oi "
            "JOIN delivered do ON oi.order_id = do.order_id"
        )
        syntax_error = psycopg.errors.SyntaxError('syntax error at or near "do"')

        with patch(
            "backend.tools.sql_analysis.explain_postgres",
            side_effect=[syntax_error, {"total_cost": 1, "plan_rows": 1, "plan_width": 8}],
        ):
            result = analyze_sql(
                query,
                {"orders", "order_items"},
                database_url="postgresql://test",
            )

        self.assertEqual(result["status"], "safe")
        self.assertIn('"do"', result["executable_sql"])

    def test_enforces_discovered_tables_and_artifact_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "discovered schema"):
            validate_sql("SELECT * FROM secrets", {"orders"})
        with self.assertRaisesRegex(ValueError, "not owned"):
            validate_sql("SELECT * FROM agent_results.result_other", {"orders"}, set())

        result = validate_sql(
            "SELECT * FROM agent_results.result_mine",
            {"orders"},
            {"agent_results.result_mine"},
        )
        self.assertEqual(result["tables"], ["agent_results.result_mine"])

    def test_ctes_and_canonical_hash_are_supported(self) -> None:
        first = validate_sql(
            "WITH totals AS (SELECT order_id FROM orders) SELECT * FROM totals",
            {"orders"},
        )
        second = validate_sql(
            "with totals as(select order_id from orders) select * from totals",
            {"orders"},
        )
        self.assertEqual(first["sql_hash"], second["sql_hash"])

    def test_metric_only_limit_one_requires_tie_handling(self) -> None:
        unstable = analyze_sql(
            "SELECT category, AVG(score) AS average_score FROM reviews "
            "GROUP BY category ORDER BY average_score DESC LIMIT 1",
            {"reviews"},
        )
        stable = analyze_sql(
            "SELECT category, AVG(score) AS average_score, COUNT(*) AS sample_count "
            "FROM reviews GROUP BY category "
            "ORDER BY average_score DESC, sample_count DESC, category LIMIT 1",
            {"reviews"},
        )
        with_ties = analyze_sql(
            "SELECT category, AVG(score) AS average_score FROM reviews "
            "GROUP BY category ORDER BY average_score DESC "
            "FETCH FIRST 1 ROW WITH TIES",
            {"reviews"},
        )

        self.assertEqual(unstable["status"], "needs_revision")
        self.assertIn("tie handling", unstable["reasons"][0])
        self.assertEqual(stable["status"], "safe")
        self.assertEqual(with_ties["status"], "safe")

    def test_tie_error_names_every_unstable_cte(self) -> None:
        result = analyze_sql(
            "WITH top_state AS ("
            "SELECT state, COUNT(*) AS order_count FROM orders "
            "GROUP BY state ORDER BY order_count DESC LIMIT 1), "
            "top_seller AS ("
            "SELECT seller_id, SUM(total) AS revenue FROM orders "
            "GROUP BY seller_id ORDER BY revenue DESC LIMIT 1) "
            "SELECT * FROM top_state CROSS JOIN top_seller",
            {"orders"},
        )

        self.assertEqual(result["status"], "needs_revision")
        self.assertIn("top_state", result["reasons"][0])
        self.assertIn("top_seller", result["reasons"][0])

    def test_returns_only_analysis_used_by_runtime(self) -> None:
        result = validate_sql(
            "SELECT o.order_id FROM orders o JOIN customers c ON c.id = o.customer_id",
            {"orders", "customers"},
        )

        self.assertFalse(result["risky_cross_join"])
        self.assertNotIn("structural_score", result)
        self.assertNotIn("structural", result)


if __name__ == "__main__":
    unittest.main()
