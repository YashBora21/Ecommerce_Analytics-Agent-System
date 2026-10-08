import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

from backend.db.postgres import postgres_connection
from backend.tools.sql_analysis import validate_sql


DB_FILE = Path(__file__).resolve().parents[1] / "db" / "ecommerce.db"
load_dotenv(DB_FILE.parents[2] / ".env")

DEFAULT_LIMIT = 200
MAX_LIMIT = 1_000
DEFAULT_TIMEOUT = 15.0

ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def check_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
) -> str:
    """Validate one read-only query and return its normalized input text."""
    validate_sql(sql, allowed_tables, allowed_artifacts)
    query = sql.strip()
    return query[:-1].rstrip() if query.endswith(";") else query


def authorizer(
    action: int,
    first_argument: str | None,
    second_argument: str | None,
    _database_name: str | None,
    _trigger_name: str | None,
) -> int:
    """Allow only the SQLite operations needed to read query results."""
    if action not in ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (second_argument or first_argument or "").lower()
        if function_name == "load_extension":
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def open_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")

    database_uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(False)
    connection.execute("PRAGMA query_only = ON")
    connection.set_authorizer(authorizer)
    return connection


def execute_postgres(
    query: str,
    database_url: str,
    limit: int,
    timeout: float,
) -> dict[str, Any]:
    """Execute a query in a read-only PostgreSQL transaction."""
    with postgres_connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(timeout * 1_000)),))
            cursor.execute("SELECT set_config('lock_timeout', %s, true)", ("2000",))
            cursor.execute(query)
            fetched = cursor.fetchmany(limit + 1)
            columns = [description.name for description in cursor.description or []]

    rows = [dict(row) for row in fetched[:limit]]
    return {
        "ok": True,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(fetched) > limit,
        "error": None,
    }


def execute_sql(
    sql: str,
    path: Path | None = None,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
) -> dict[str, Any]:
    """Run one validated read-only query and always return the same shape."""
    try:
        query = check_sql(sql, allowed_tables, allowed_artifacts)
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        database_url = os.getenv("DATABASE_URL") if path is None else None
        if database_url:
            return execute_postgres(query, database_url, limit, timeout)

        connection = open_readonly(path or DB_FILE)
        try:
            deadline = time.monotonic() + timeout
            connection.set_progress_handler(
                lambda: int(time.monotonic() >= deadline),
                1_000,
            )
            connection.execute(f"EXPLAIN QUERY PLAN {query}").fetchall()
            cursor = connection.execute(query)
            fetched = cursor.fetchmany(limit + 1)
            columns = [description[0] for description in cursor.description or []]
        finally:
            connection.close()

        rows = [dict(row) for row in fetched[:limit]]
        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": len(fetched) > limit,
            "error": None,
        }
    except (ValueError, FileNotFoundError, sqlite3.Error, sqlite3.Warning, psycopg.Error) as error:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(error),
        }

