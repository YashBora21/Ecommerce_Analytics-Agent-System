import os
from typing import Any

import psycopg
from dotenv import load_dotenv

from backend.db.postgres import postgres_connection
from backend.tools.sql_analysis import validate_sql

load_dotenv()
DEFAULT_LIMIT = 200
MAX_LIMIT = 1000
DEFAULT_TIMEOUT = 15.0
LOCK_TIMEOUT_MS = 2000


def check_sql(sql, allowed_tables=None, allowed_artifacts=None) -> str:
    validate_sql(sql, allowed_tables, allowed_artifacts)
    return sql.strip().removesuffix(";").rstrip()


def execute_postgres(query: str, database_url: str, limit: int, timeout: float) -> dict[str, Any]:
    with postgres_connection(database_url) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(int(timeout * 1000)),),
        )
        cur.execute(
            "SELECT set_config('lock_timeout', %s, true)",
            (str(LOCK_TIMEOUT_MS),),
        )

        cur.execute(query)

        fetched = cur.fetchmany(limit + 1)
        columns = [d.name for d in cur.description or []]

    rows = [dict(r) for r in fetched[:limit]]

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
    limit: int = DEFAULT_LIMIT,
    timeout: float = DEFAULT_TIMEOUT,
    allowed_tables=None,
    allowed_artifacts=None,
) -> dict[str, Any]:
    """Validate and run a single read-only query against Postgres."""
    try:
        query = check_sql(sql, allowed_tables, allowed_artifacts)

        if not 1 <= limit <= MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL is not set")
        return execute_postgres(query, database_url, limit, timeout)

    except (ValueError, psycopg.Error) as e:
        return {
            "ok": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(e),
        }
