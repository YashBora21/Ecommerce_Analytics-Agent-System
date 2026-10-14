import unittest
from unittest.mock import MagicMock, patch

from backend.agent.nodes.schema import make_inspect_schema
from backend.agent.routing import after_inspection
from backend.tools.schema import _SELECT_CACHE, _postgres_catalog, discover_schema


def table(name, columns, **metadata):
    return {
        "schema": "public",
        "table": name,
        "columns": [
            {"name": column, "type": "text", "nullable": False}
            for column in columns
        ],
        "primary_key": [],
        "foreign_keys": [],
        **metadata,
    }


class SchemaDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        _SELECT_CACHE.clear()

    def test_postgres_catalog_only_reads_configured_data_schemas(self) -> None:
        connection = MagicMock()
        connection.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=[])),
            MagicMock(fetchall=MagicMock(return_value=[])),
            MagicMock(fetchall=MagicMock(return_value=[])),
            MagicMock(fetchall=MagicMock(return_value=[])),
        ]
        context = MagicMock()
        context.__enter__.return_value = connection

        with patch("backend.tools.schema.postgres_connection", return_value=context):
            _postgres_catalog("postgresql://example")

        self.assertEqual(connection.execute.call_args_list[0].args[1], (["public"],))
        self.assertEqual(
            connection.execute.call_args_list[1].args[1],
            (["public"],),
        )
        self.assertEqual(
            connection.execute.call_args_list[2].args[1],
            (["public"], ["public"]),
        )
        self.assertEqual(connection.execute.call_args_list[3].args[1], (["public"],))

    def test_returns_relevant_tables_and_join_keys(self) -> None:
        catalog = {
            "dialect": "postgres",
            "tables": {
                "public.customers": {
                    "schema": "public",
                    "table": "customers",
                    "columns": [
                        {"name": "customer_id", "type": "text", "nullable": False},
                        {"name": "region", "type": "text", "nullable": True},
                    ],
                    "foreign_keys": [],
                },
                "public.orders": {
                    "schema": "public",
                    "table": "orders",
                    "columns": [
                        {"name": "order_id", "type": "text", "nullable": False},
                        {"name": "customer_id", "type": "text", "nullable": False},
                        {"name": "revenue", "type": "double precision", "nullable": False},
                    ],
                    "foreign_keys": [
                        {
                            "column": "customer_id",
                            "target": "public.customers.customer_id",
                        }
                    ],
                },
            },
        }

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema("revenue by customer region", refresh=True)

        self.assertEqual(
            set(result["allowed_tables"]),
            {"public.orders", "public.customers"},
        )
        self.assertIn(
            "FK customer_id -> public.customers.customer_id",
            result["context"],
        )

    def test_llm_selects_semantic_metric_and_code_adds_shared_id_neighbor(self) -> None:
        catalog = {
            "dialect": "postgres",
            "fingerprint": "semantic",
            "tables": {
                "public.audit_log": table("audit_log", ["event_id"]),
                "public.customers": table("customers", ["customer_id"]),
                "public.order_items": table("order_items", ["order_id", "price"]),
                "public.orders": table(
                    "orders", ["order_id", "customer_id", "purchased_at"]
                ),
            },
        }

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema(
                "revenue by customer",
                llm_call=lambda _prompt: (
                    '{"tables": ["public.order_items", "public.customers", "made_up"]}'
                ),
            )

        self.assertEqual(
            [item["table"] for item in result["tables"]],
            ["order_items", "customers", "orders"],
        )
        self.assertEqual(
            result["allowed_tables"],
            ["public.customers", "public.order_items", "public.orders"],
        )

    def test_relevant_table_is_not_lost_when_neighbors_exceed_limit(self) -> None:
        catalog = {
            "dialect": "postgres",
            "fingerprint": "ranking",
            "tables": {
                "public.a_audit": table("a_audit", ["order_id"]),
                "public.m_items": table("m_items", ["order_id"]),
                "public.z_orders": table("z_orders", ["order_id"]),
            },
        }

        with (
            patch("backend.tools.schema.load_catalog", return_value=catalog),
            patch("backend.tools.schema.MAX_SCHEMA_TABLES", 2),
        ):
            result = discover_schema("orders")

        self.assertEqual(result["tables"][0]["table"], "z_orders")
        self.assertEqual(len(result["tables"]), 2)

    def test_no_match_does_not_return_alphabetical_tables(self) -> None:
        catalog = {
            "dialect": "postgres",
            "fingerprint": "empty",
            "tables": {
                "public.audit_log": table("audit_log", ["event_id"]),
                "public.customers": table("customers", ["customer_id"]),
            },
        }

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema("weather tomorrow")

        self.assertEqual(result["tables"], [])
        self.assertEqual(result["allowed_tables"], [])
        self.assertTrue(result["selection_failed"])

    def test_invalid_selector_json_uses_token_fallback(self) -> None:
        catalog = {
            "dialect": "postgres",
            "fingerprint": "invalid-json",
            "tables": {"public.orders": table("orders", ["order_id"])},
        }

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema(
                "orders",
                llm_call=lambda _prompt: '{"tables": "public.orders"}',
            )

        self.assertEqual(result["allowed_tables"], ["public.orders"])
        self.assertFalse(result["selection_failed"])

    def test_llm_failure_uses_token_matches(self) -> None:
        catalog = {
            "dialect": "postgres",
            "fingerprint": "fallback",
            "tables": {"public.orders": table("orders", ["order_id"])},
        }

        def timeout(_prompt):
            raise TimeoutError

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema("orders", llm_call=timeout)

        self.assertEqual(result["allowed_tables"], ["public.orders"])

    def test_empty_selection_routes_to_explicit_failure(self) -> None:
        inspect_schema = make_inspect_schema(
            lambda _question, _state: {
                "context": "",
                "allowed_tables": [],
                "artifacts": [],
                "selection_failed": True,
            }
        )

        result = inspect_schema({"question": "unknown metric"})

        self.assertIn("No relevant database tables", result["error"])
        self.assertEqual(after_inspection(result), "partial")

    def test_empty_selection_can_continue_with_owned_artifacts(self) -> None:
        inspect_schema = make_inspect_schema(
            lambda _question, _state: {
                "context": "",
                "allowed_tables": [],
                "artifacts": [{"artifact_id": "owned"}],
                "selection_failed": True,
            }
        )

        result = inspect_schema({"question": "use the previous result"})

        self.assertEqual(result["error"], "")
        self.assertEqual(after_inspection(result), "agent")

    def test_context_includes_grain_cardinality_values_and_date_range(self) -> None:
        orders = table(
            "orders",
            ["order_id", "status", "purchased_at"],
            primary_key=["order_id"],
            estimated_rows=100,
        )
        orders["columns"][1].update(
            {"distinct": 3, "common_values": ["paid", "delivered", "canceled"]}
        )
        orders["columns"][2].update(
            {
                "type": "timestamp without time zone",
                "distinct": -0.9,
                "range": ["2020-01-01", "2020-12-31"],
            }
        )
        catalog = {
            "dialect": "postgres",
            "fingerprint": "metadata",
            "tables": {"public.orders": orders},
        }

        with patch("backend.tools.schema.load_catalog", return_value=catalog):
            result = discover_schema("orders")

        self.assertIn("grain: one row per (order_id)", result["context"])
        self.assertIn("estimated rows: 100", result["context"])
        self.assertIn("examples: ['paid', 'delivered', 'canceled']", result["context"])
        self.assertIn("approx range: 2020-01-01 to 2020-12-31", result["context"])


if __name__ == "__main__":
    unittest.main()
