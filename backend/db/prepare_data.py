import csv
import sqlite3
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "ecommerce_sales_analytics_5000.csv"
DEFAULT_DB_PATH = Path(__file__).with_name("ecommerce.db")

ORDER_COLUMNS = [
    "order_id",
    "order_date",
    "order_status",
    "customer_id",
    "customer_city",
    "customer_state",
    "product_category",
    "seller_id",
    "seller_state",
    "quantity",
    "distinct_products",
    "item_value",
    "freight_value",
    "order_value",
    "payment_method",
    "payment_installments",
    "payment_value",
    "review_score",
    "estimated_delivery_date",
    "delivered_date",
    "delivery_days",
]

CREATE_DATABASE_SQL = """
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    order_date TEXT NOT NULL,
    order_status TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    customer_city TEXT NOT NULL,
    customer_state TEXT NOT NULL,
    product_category TEXT NOT NULL,
    seller_id TEXT NOT NULL,
    seller_state TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    distinct_products INTEGER NOT NULL CHECK (distinct_products >= 0),
    item_value REAL NOT NULL CHECK (item_value >= 0),
    freight_value REAL NOT NULL CHECK (freight_value >= 0),
    order_value REAL NOT NULL CHECK (order_value >= 0),
    payment_method TEXT NOT NULL,
    payment_installments INTEGER NOT NULL CHECK (payment_installments >= 0),
    payment_value REAL NOT NULL CHECK (payment_value >= 0),
    review_score REAL NOT NULL CHECK (review_score BETWEEN 0 AND 5),
    estimated_delivery_date TEXT NOT NULL,
    delivered_date TEXT NOT NULL,
    delivery_days REAL NOT NULL CHECK (delivery_days >= -1)
) STRICT;

CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_orders_category ON orders(product_category);
CREATE INDEX idx_orders_customer_state ON orders(customer_state);
CREATE INDEX idx_orders_seller_state ON orders(seller_state);
"""


def read_orders(csv_path: Path) -> list[dict[str, str]]:
    """Read the CSV and fail early if its expected shape has changed."""
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != ORDER_COLUMNS:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        orders = list(reader)

    if not orders:
        raise ValueError("CSV contains no orders")
    if any(not value for order in orders for value in order.values()):
        raise ValueError("CSV contains empty values")
    if len({order["order_id"] for order in orders}) != len(orders):
        raise ValueError("CSV contains duplicate order IDs")

    return orders


def insert_orders(connection: sqlite3.Connection, orders: list[dict[str, str]]) -> None:
    """Create the schema and insert every order in one transaction."""
    connection.executescript(CREATE_DATABASE_SQL)
    placeholders = ", ".join("?" for _ in ORDER_COLUMNS)
    connection.executemany(
        f"INSERT INTO orders VALUES ({placeholders})",
        ([order[column] for column in ORDER_COLUMNS] for order in orders),
    )


def validate_database(connection: sqlite3.Connection, expected_count: int) -> None:
    order_count, incorrect_totals = connection.execute(
        """
        SELECT
            COUNT(*),
            SUM(ABS(order_value - item_value - freight_value) > 0.011)
        FROM orders
        """
    ).fetchone()

    if order_count != expected_count or incorrect_totals:
        raise ValueError(
            "Database validation failed: "
            f"orders={order_count}, incorrect_totals={incorrect_totals}"
        )


def prepare_database(
    csv_path: Path = DEFAULT_CSV_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Build a validated database without risking the current working copy."""
    temporary_db_path = db_path.with_suffix(".db.tmp")
    temporary_db_path.unlink(missing_ok=True)
    orders = read_orders(csv_path)

    with closing(sqlite3.connect(temporary_db_path)) as connection, connection:
        insert_orders(connection, orders)
        validate_database(connection, len(orders))

    temporary_db_path.replace(db_path)
    print(f"Created {db_path} with {len(orders)} validated orders")


if __name__ == "__main__":
    prepare_database()
