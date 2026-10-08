import sqlite3
import unittest
from pathlib import Path

from backend.tools.schema import discover_schema


class SchemaDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).with_name("_schema_test.db")
        self.path.unlink(missing_ok=True)
        connection = sqlite3.connect(self.path)
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE customers (customer_id TEXT PRIMARY KEY, region TEXT);
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT REFERENCES customers(customer_id),
                revenue REAL
            );
            CREATE TABLE order_items (
                order_id TEXT REFERENCES orders(order_id),
                price REAL
            );
            CREATE TABLE unrelated (value TEXT);
            """
        )
        connection.close()

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_returns_relevant_live_schema_and_join_keys(self) -> None:
        result = discover_schema("revenue by customer region", self.path, refresh=True)
        self.assertIn("orders", result["allowed_tables"])
        self.assertIn("customers", result["allowed_tables"])
        self.assertIn("order_items", result["allowed_tables"])
        self.assertIn("FK customer_id -> customers.customer_id", result["context"])


if __name__ == "__main__":
    unittest.main()
