import sqlite3
import unittest

from backend.tools.sql_core import DB_FILE


class DatabaseContractTests(unittest.TestCase):
    def test_complete_olist_schema_and_relationships(self) -> None:
        connection = sqlite3.connect(DB_FILE)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            expected = {
                "orders", "customers", "order_items", "payments", "reviews",
                "products", "sellers", "geolocation", "category_translation",
            }
            self.assertTrue(expected.issubset(tables))
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertGreater(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
