import unittest
from unittest.mock import patch

from backend.tools.schema import discover_schema


class SchemaDiscoveryTests(unittest.TestCase):
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

        self.assertIn("orders", result["allowed_tables"])
        self.assertIn("customers", result["allowed_tables"])
        self.assertIn(
            "FK customer_id -> public.customers.customer_id",
            result["context"],
        )


if __name__ == "__main__":
    unittest.main()
