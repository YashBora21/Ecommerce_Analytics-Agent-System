import re
import sqlite3
import time
from pathlib import Path
from typing import Any


DB_FILE = Path(__file__).resolve().parents[1] / "db" / "ecommerce.db"
DEFAULT_LIMIT = 200
MAX_LIMIT = 1_000
DEFAULT_TIMEOUT = 2.0

ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def check_sql(sql: str) -> str:
    """Normalize one SELECT statement or raise a clear validation error."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")

    query = sql.strip()
    if "\x00" in query:
        raise ValueError("SQL cannot contain null bytes")
    if "--" in query or "/*" in query or "*/" in query:
        raise ValueError("SQL comments are not allowed")

    if query.endswith(";"):
        query = query[:-1].rstrip()
    if ";" in query:
        raise ValueError("Only one SQL statement is allowed")
    if not re.match(r"(?:SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed")

    return query


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


def execute_sql(
    sql: str,
    path: Path = DB_FILE,
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run one validated read-only query and always return the same shape."""
    try:
        query = check_sql(sql)
        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        connection = open_readonly(path)
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
    except (ValueError, FileNotFoundError, sqlite3.Error, sqlite3.Warning) as error:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(error),
        }
