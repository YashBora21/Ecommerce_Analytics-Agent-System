import unittest
from unittest.mock import patch

from backend.tools.sql_analysis import analyze_sql, validate_sql


class SqlAnalysisTests(unittest.TestCase):
    def test_rejects_injection_and_write_statements(self) -> None:
        for query in (
            "SELECT * FROM orders; DROP TABLE orders",
            "DELETE FROM orders",
            "SELECT pg_sleep(10)",
            "SELECT * FROM orders CROSS JOIN customers",
            "SELECT * FROM orders JOIN customers",
        ):
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_sql(query, {"orders", "customers"})

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

    def test_reports_correlated_and_predicate_complexity(self) -> None:
        result = validate_sql(
            "SELECT * FROM orders o WHERE o.total > 0 AND EXISTS "
            "(SELECT 1 FROM customers c WHERE c.id = o.customer_id)",
            {"orders", "customers"},
        )
        self.assertEqual(result["structural"]["correlated_subqueries"], 1)
        self.assertGreater(result["structural"]["predicate_depth"], 0)

    def test_structural_complexity_alone_does_not_reject_sql(self) -> None:
        with patch("backend.tools.sql_analysis.DEFAULT_MAX_PLAN_ROWS", 1):
            result = analyze_sql(
                "SELECT o.order_id FROM orders o JOIN customers c ON c.id = o.customer_id",
                {"orders", "customers"},
            )
        self.assertEqual(result["status"], "safe")
        self.assertGreater(result["structural_score"], 1)


if __name__ == "__main__":
    unittest.main()
