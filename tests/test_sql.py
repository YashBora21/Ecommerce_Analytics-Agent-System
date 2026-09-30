import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from backend.tools.sql import run_sql
from backend.tools.sql_core import execute_sql


class SqlCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(__file__).with_name("_test_sql.db")
        self.db_path.unlink(missing_ok=True)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("CREATE TABLE orders (order_id TEXT, order_value REAL)")
            connection.executemany(
                "INSERT INTO orders VALUES (?, ?)",
                [("A", 10.0), ("B", 20.0), ("C", 30.0)],
            )

    def tearDown(self) -> None:
        self.db_path.unlink(missing_ok=True)

    def test_runs_select_query(self) -> None:
        result = execute_sql(
            "SELECT COUNT(*) AS orders, SUM(order_value) AS revenue FROM orders",
            self.db_path,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], [{"orders": 3, "revenue": 60.0}])
        self.assertIsNone(result["error"])

    def test_runs_cte_query(self) -> None:
        result = execute_sql(
            "WITH totals AS (SELECT SUM(order_value) AS revenue FROM orders) "
            "SELECT revenue FROM totals",
            self.db_path,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["rows"], [{"revenue": 60.0}])

    def test_rejects_write_query(self) -> None:
        result = execute_sql("DELETE FROM orders", self.db_path)
        self.assertFalse(result["ok"])
        self.assertIn("Only SELECT", result["error"])

    def test_rejects_multiple_statements(self) -> None:
        result = execute_sql("SELECT 1; SELECT 2", self.db_path)
        self.assertFalse(result["ok"])
        self.assertIn("one SQL statement", result["error"])

    def test_caps_result_rows(self) -> None:
        result = execute_sql(
            "SELECT * FROM orders ORDER BY order_id",
            self.db_path,
            limit=2,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["row_count"], 2)
        self.assertTrue(result["truncated"])

    def test_rejects_non_positive_timeout(self) -> None:
        result = execute_sql("SELECT 1", self.db_path, timeout=0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout must be positive")

    def test_error_result_has_consistent_shape(self) -> None:
        result = execute_sql("DROP TABLE orders", self.db_path)
        self.assertEqual(
            set(result),
            {"ok", "columns", "rows", "row_count", "truncated", "error"},
        )


class SqlToolTests(unittest.TestCase):
    @patch("backend.tools.sql.execute_sql")
    def test_formats_csv_safely_for_the_agent(self, execute_sql_mock) -> None:
        execute_sql_mock.return_value = {
            "ok": True,
            "columns": ["state", "revenue"],
            "rows": [{"state": "Rio, Metro", "revenue": 100.5}],
            "row_count": 1,
            "truncated": False,
            "error": None,
        }

        output = run_sql.invoke({"query": "SELECT state, revenue FROM orders"})

        self.assertEqual(output, 'state,revenue\n"Rio, Metro",100.5')

    @patch("backend.tools.sql.execute_sql")
    def test_returns_validation_error_to_the_agent(self, execute_sql_mock) -> None:
        execute_sql_mock.return_value = {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": "Only SELECT queries are allowed",
        }

        output = run_sql.invoke({"query": "DELETE FROM orders"})

        self.assertEqual(output, "Rejected: Only SELECT queries are allowed")


if __name__ == "__main__":
    unittest.main()
