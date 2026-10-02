import csv
import sqlite3
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = Path(__file__).with_name("ecommerce.db")

FILES = {
    "category_translation": ("product_category_name_translation.csv", ["product_category_name", "product_category_name_english"]),
    "customers": ("olist_customers_dataset.csv", ["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]),
    "geolocation": ("olist_geolocation_dataset.csv", ["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"]),
    "products": ("olist_products_dataset.csv", ["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]),
    "sellers": ("olist_sellers_dataset.csv", ["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]),
    "orders": ("olist_orders_dataset.csv", ["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"]),
    "order_items": ("olist_order_items_dataset.csv", ["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price", "freight_value"]),
    "payments": ("olist_order_payments_dataset.csv", ["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"]),
    "reviews": ("olist_order_reviews_dataset.csv", ["review_id", "order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"]),
}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE category_translation (product_category_name TEXT PRIMARY KEY, product_category_name_english TEXT NOT NULL) STRICT;
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, customer_unique_id TEXT NOT NULL, customer_zip_code_prefix INTEGER NOT NULL, customer_city TEXT NOT NULL, customer_state TEXT NOT NULL) STRICT;
CREATE TABLE geolocation (geolocation_zip_code_prefix INTEGER NOT NULL, geolocation_lat REAL NOT NULL, geolocation_lng REAL NOT NULL, geolocation_city TEXT NOT NULL, geolocation_state TEXT NOT NULL) STRICT;
CREATE TABLE products (product_id TEXT PRIMARY KEY, product_category_name TEXT, product_name_length INTEGER, product_description_length INTEGER, product_photos_qty INTEGER, product_weight_g REAL, product_length_cm REAL, product_height_cm REAL, product_width_cm REAL) STRICT;
CREATE TABLE sellers (seller_id TEXT PRIMARY KEY, seller_zip_code_prefix INTEGER NOT NULL, seller_city TEXT NOT NULL, seller_state TEXT NOT NULL) STRICT;
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id), order_status TEXT NOT NULL, order_purchase_timestamp TEXT NOT NULL, order_approved_at TEXT, order_delivered_carrier_date TEXT, order_delivered_customer_date TEXT, order_estimated_delivery_date TEXT NOT NULL) STRICT;
CREATE TABLE order_items (order_id TEXT NOT NULL REFERENCES orders(order_id), order_item_id INTEGER NOT NULL, product_id TEXT NOT NULL REFERENCES products(product_id), seller_id TEXT NOT NULL REFERENCES sellers(seller_id), shipping_limit_date TEXT NOT NULL, price REAL NOT NULL CHECK (price >= 0), freight_value REAL NOT NULL CHECK (freight_value >= 0), PRIMARY KEY (order_id, order_item_id)) STRICT;
CREATE TABLE payments (order_id TEXT NOT NULL REFERENCES orders(order_id), payment_sequential INTEGER NOT NULL, payment_type TEXT NOT NULL, payment_installments INTEGER NOT NULL CHECK (payment_installments >= 0), payment_value REAL NOT NULL CHECK (payment_value >= 0), PRIMARY KEY (order_id, payment_sequential)) STRICT;
CREATE TABLE reviews (review_id TEXT NOT NULL, order_id TEXT NOT NULL REFERENCES orders(order_id), review_score INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5), review_comment_title TEXT, review_comment_message TEXT, review_creation_date TEXT NOT NULL, review_answer_timestamp TEXT NOT NULL, PRIMARY KEY (review_id, order_id)) STRICT;
CREATE INDEX idx_customers_unique ON customers(customer_unique_id);
CREATE INDEX idx_customers_state ON customers(customer_state);
CREATE INDEX idx_geolocation_zip ON geolocation(geolocation_zip_code_prefix);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_purchase ON orders(order_purchase_timestamp);
CREATE INDEX idx_orders_status ON orders(order_status);
CREATE INDEX idx_items_product ON order_items(product_id);
CREATE INDEX idx_items_seller ON order_items(seller_id);
CREATE INDEX idx_payments_order ON payments(order_id);
CREATE INDEX idx_reviews_order ON reviews(order_id);
CREATE INDEX idx_products_category ON products(product_category_name);
"""

TARGET_COLUMNS = {
    "products": ["product_id", "product_category_name", "product_name_length", "product_description_length", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]
}


def import_csv(connection: sqlite3.Connection, table: str, data_dir: Path) -> int:
    filename, source_columns = FILES[table]
    target_columns = TARGET_COLUMNS.get(table, source_columns)
    placeholders = ", ".join("?" for _ in source_columns)
    sql = f"INSERT INTO {table} ({', '.join(target_columns)}) VALUES ({placeholders})"

    with (data_dir / filename).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header != source_columns:
            raise ValueError(f"Unexpected columns in {filename}: {header}")
        count = 0
        batch = []
        for row in reader:
            if len(row) != len(source_columns):
                raise ValueError(f"Malformed row in {filename} at line {count + 2}")
            batch.append([value if value != "" else None for value in row])
            count += 1
            if len(batch) == 10_000:
                connection.executemany(sql, batch)
                batch.clear()
        if batch:
            connection.executemany(sql, batch)
    if not count:
        raise ValueError(f"{filename} contains no rows")
    return count


def prepare_database(data_dir: Path = DATA_DIR, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Import the complete relational Olist dataset into a validated SQLite file."""
    temporary_path = db_path.with_suffix(".db.tmp")
    temporary_path.unlink(missing_ok=True)
    counts = {}
    try:
        with closing(sqlite3.connect(temporary_path)) as connection, connection:
            connection.executescript(SCHEMA)
            for table in FILES:
                counts[table] = import_csv(connection, table, data_dir)
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise ValueError(f"Foreign-key validation failed: {violations[:5]}")
            for table, expected in counts.items():
                actual = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if actual != expected:
                    raise ValueError(f"Row-count validation failed for {table}")
        temporary_path.replace(db_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    summary = ", ".join(f"{table}={count}" for table, count in counts.items())
    print(f"Created {db_path}: {summary}")


if __name__ == "__main__":
    prepare_database()
