import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from backend.agent.nodes.sql import make_execute_query
from backend.tools.artifacts import artifact_csv
from backend.tools.sql import run_managed_sql, run_sql
from backend.tools.sql_core import execute_sql


class SqlCoreTests(unittest.TestCase):
    def test_artifact_csv_streams_rows_without_loading_everything(self) -> None:
        cursor = MagicMock()
        cursor.fetchmany.side_effect = [
            [{"seller_id": "s1", "revenue": 10}, {"seller_id": "s2", "revenue": 20}],
            [],
        ]
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor
        connection = MagicMock()
        connection.cursor.return_value = cursor_context

        @contextmanager
        def connection_context(_database_url):
            yield connection

        artifact = {
            "relation_name": "result_owned",
            "columns": [{"name": "seller_id"}, {"name": "revenue"}],
        }
        with patch("backend.tools.artifacts.owned_artifact", return_value=artifact), patch(
            "backend.tools.artifacts.postgres.postgres_connection",
            side_effect=connection_context,
        ):
            output = "".join(
                artifact_csv(
                    "00000000-0000-0000-0000-000000000001",
                    "thread-1",
                    "postgresql://test",
                )
            )

        self.assertEqual(output, "seller_id,revenue\ns1,10\ns2,20\n")
        cursor.fetchmany.assert_called_with(1000)

    @patch("backend.tools.sql_core.execute_postgres")
    def test_runs_postgres_query(self, execute_postgres_mock) -> None:
        expected = {
            "ok": True,
            "columns": ["orders"],
            "rows": [{"orders": 3}],
            "row_count": 1,
            "truncated": False,
            "error": None,
        }
        execute_postgres_mock.return_value = expected

        with patch.dict(
            "backend.tools.sql_core.os.environ",
            {"DATABASE_URL": "postgresql://test"},
            clear=True,
        ):
            result = execute_sql("SELECT COUNT(*) AS orders FROM orders", limit=2)

        self.assertEqual(result, expected)
        execute_postgres_mock.assert_called_once_with(
            "SELECT COUNT(*) AS orders FROM orders",
            "postgresql://test",
            2,
            15.0,
        )

    def test_rejects_write_query(self) -> None:
        result = execute_sql("DELETE FROM orders")
        self.assertFalse(result["ok"])
        self.assertIn("Only SELECT", result["error"])

    def test_rejects_multiple_statements(self) -> None:
        result = execute_sql("SELECT 1; SELECT 2")
        self.assertFalse(result["ok"])
        self.assertIn("one SQL statement", result["error"])

    def test_rejects_non_positive_timeout(self) -> None:
        result = execute_sql("SELECT 1", timeout=0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout must be positive")

    def test_requires_database_url(self) -> None:
        with patch.dict("backend.tools.sql_core.os.environ", {}, clear=True):
            result = execute_sql("SELECT 1")
        self.assertEqual(result["error"], "DATABASE_URL is not set")

    def test_error_result_has_consistent_shape(self) -> None:
        result = execute_sql("DROP TABLE orders")
        self.assertEqual(
            set(result),
            {"ok", "columns", "rows", "row_count", "truncated", "error"},
        )


class SqlToolTests(unittest.TestCase):
    def test_does_not_execute_the_same_failed_query_twice(self) -> None:
        sql_runner = MagicMock()
        sql_analyzer = MagicMock(
            return_value={
                "status": "safe",
                "sql_hash": "same-query",
                "executable_sql": "SELECT 1",
            }
        )
        execute_query = make_execute_query(sql_runner, sql_analyzer)

        result = execute_query(
            {
                "action": {
                    "args": {"sql": "SELECT 1", "result_scope": "complete"}
                },
                "analysis": {"sql_hash": "same-query"},
                "sql_tries": 1,
            }
        )

        sql_runner.assert_not_called()
        self.assertIn("already failed", result["error"])

    @patch("backend.tools.sql.materialize_result")
    @patch("backend.tools.sql.execute_sql")
    def test_small_results_stay_inline(self, execute_sql_mock, materialize_mock) -> None:
        execute_sql_mock.return_value = {
            "ok": True,
            "columns": ["state", "revenue"],
            "rows": [{"state": "SP", "revenue": 100}],
            "row_count": 1,
            "truncated": False,
            "error": None,
        }

        with patch.dict("backend.tools.sql.os.environ", {"DATABASE_URL": "postgresql://test"}):
            output = run_managed_sql("SELECT state, revenue FROM orders", owner_id="thread")

        self.assertEqual(output, "state,revenue\nSP,100")
        materialize_mock.assert_not_called()

    @patch("backend.tools.sql.materialize_result")
    @patch("backend.tools.sql.execute_sql")
    def test_truncated_results_are_materialized(
        self,
        execute_sql_mock,
        materialize_mock,
    ) -> None:
        execute_sql_mock.return_value = {
            "ok": True,
            "columns": ["order_id"],
            "rows": [{"order_id": "A"}],
            "row_count": 200,
            "truncated": True,
            "error": None,
        }
        materialize_mock.return_value = {
            "artifact_id": "artifact-1",
            "columns": [{"name": "order_id"}],
            "preview": [{"order_id": "A"}],
        }

        with patch.dict("backend.tools.sql.os.environ", {"DATABASE_URL": "postgresql://test"}):
            output = run_managed_sql("SELECT order_id FROM orders", owner_id="thread")

        self.assertTrue(output.startswith("[Artifact]\n"))
        materialize_mock.assert_called_once()

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
    def test_passes_artifact_ownership_to_execution(self, execute_sql_mock) -> None:
        execute_sql_mock.return_value = {
            "ok": True,
            "columns": ["value"],
            "rows": [{"value": 1}],
            "row_count": 1,
            "truncated": False,
            "error": None,
        }
        allowed_tables = {"orders"}
        allowed_artifacts = {"agent_results.result_owned"}

        run_managed_sql(
            "SELECT value FROM agent_results.result_owned",
            allowed_tables=allowed_tables,
            allowed_artifacts=allowed_artifacts,
        )

        execute_sql_mock.assert_called_once_with(
            "SELECT value FROM agent_results.result_owned",
            allowed_tables=allowed_tables,
            allowed_artifacts=allowed_artifacts,
        )

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
