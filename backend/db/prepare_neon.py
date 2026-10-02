import csv
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from backend.db.prepare_data import DATA_DIR, FILES, TARGET_COLUMNS


load_dotenv(Path(__file__).resolve().parents[2] / ".env")

SCHEMA = """
CREATE TABLE category_translation (product_category_name TEXT PRIMARY KEY, product_category_name_english TEXT NOT NULL);
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, customer_unique_id TEXT NOT NULL, customer_zip_code_prefix INTEGER NOT NULL, customer_city TEXT NOT NULL, customer_state TEXT NOT NULL);
CREATE TABLE geolocation (geolocation_zip_code_prefix INTEGER NOT NULL, geolocation_lat DOUBLE PRECISION NOT NULL, geolocation_lng DOUBLE PRECISION NOT NULL, geolocation_city TEXT NOT NULL, geolocation_state TEXT NOT NULL);
CREATE TABLE products (product_id TEXT PRIMARY KEY, product_category_name TEXT, product_name_length INTEGER, product_description_length INTEGER, product_photos_qty INTEGER, product_weight_g DOUBLE PRECISION, product_length_cm DOUBLE PRECISION, product_height_cm DOUBLE PRECISION, product_width_cm DOUBLE PRECISION);
CREATE TABLE sellers (seller_id TEXT PRIMARY KEY, seller_zip_code_prefix INTEGER NOT NULL, seller_city TEXT NOT NULL, seller_state TEXT NOT NULL);
CREATE TABLE orders (order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(customer_id), order_status TEXT NOT NULL, order_purchase_timestamp TIMESTAMP NOT NULL, order_approved_at TIMESTAMP, order_delivered_carrier_date TIMESTAMP, order_delivered_customer_date TIMESTAMP, order_estimated_delivery_date TIMESTAMP NOT NULL);
CREATE TABLE order_items (order_id TEXT NOT NULL REFERENCES orders(order_id), order_item_id INTEGER NOT NULL, product_id TEXT NOT NULL REFERENCES products(product_id), seller_id TEXT NOT NULL REFERENCES sellers(seller_id), shipping_limit_date TIMESTAMP NOT NULL, price DOUBLE PRECISION NOT NULL CHECK (price >= 0), freight_value DOUBLE PRECISION NOT NULL CHECK (freight_value >= 0), PRIMARY KEY (order_id, order_item_id));
CREATE TABLE payments (order_id TEXT NOT NULL REFERENCES orders(order_id), payment_sequential INTEGER NOT NULL, payment_type TEXT NOT NULL, payment_installments INTEGER NOT NULL CHECK (payment_installments >= 0), payment_value DOUBLE PRECISION NOT NULL CHECK (payment_value >= 0), PRIMARY KEY (order_id, payment_sequential));
CREATE TABLE reviews (review_id TEXT NOT NULL, order_id TEXT NOT NULL REFERENCES orders(order_id), review_score INTEGER NOT NULL CHECK (review_score BETWEEN 1 AND 5), review_comment_title TEXT, review_comment_message TEXT, review_creation_date TIMESTAMP NOT NULL, review_answer_timestamp TIMESTAMP NOT NULL, PRIMARY KEY (review_id, order_id));

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


def import_table(cursor, table: str, data_dir: Path) -> int:
    filename, source_columns = FILES[table]
    columns = TARGET_COLUMNS.get(table, source_columns)
    path = data_dir / filename

    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header != source_columns:
            raise ValueError(f"Unexpected columns in {filename}: {header}")
        count = sum(1 for _ in reader)

    with path.open(encoding="utf-8-sig", newline="") as source:
        source.readline()
        with cursor.copy(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN "
            "WITH (FORMAT CSV, NULL '')"
        ) as copy:
            for line in source:
                copy.write(line)

    if not count:
        raise ValueError(f"{filename} contains no rows")
    return count


def prepare_neon(data_dir: Path = DATA_DIR) -> None:
    """Import into an empty Neon database and validate of the Olist CSV files."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    counts = {}
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename = ANY(%s)",
                (list(FILES),),
            )
            existing = [row[0] for row in cursor.fetchall()]
            if existing:
                raise RuntimeError(
                    "Neon import requires an empty database; existing tables: "
                    + ", ".join(sorted(existing))
                )
            cursor.execute(SCHEMA)
            for table in FILES:
                counts[table] = import_table(cursor, table, data_dir)
            for table, expected in counts.items():
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                actual = cursor.fetchone()[0]
                if actual != expected:
                    raise ValueError(f"Row-count validation failed for {table}")

    summary = ", ".join(f"{table}={count}" for table, count in counts.items())
    print(f"Imported Neon database: {summary}")


if __name__ == "__main__":
    prepare_neon()




