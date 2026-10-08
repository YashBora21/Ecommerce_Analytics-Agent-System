import csv
import io
import json
import os

from langchain_core.tools import tool

from .artifacts import MATERIALIZE_ROWS, materialize_result
from .sql_core import execute_sql


def _csv(columns, rows) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    return output.getvalue().strip()


def run_managed_sql(
    query: str,
    owner_id: str = "",
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
    estimated_rows: int = 0,
) -> str:
    """Execute a small result or materialize a large PostgreSQL result."""
    can_materialize = bool(owner_id and os.getenv("DATABASE_URL"))
    if can_materialize and estimated_rows > MATERIALIZE_ROWS:
        artifact = materialize_result(
            query,
            owner_id,
            allowed_tables or set(),
            allowed_artifacts or set(),
        )
        preview = _csv(
            [column["name"] for column in artifact["columns"]],
            artifact["preview"],
        )
        return "[Artifact]\n" + json.dumps(artifact, default=str) + f"\nPreview:\n{preview}"

    result = execute_sql(
        query,
        allowed_tables=allowed_tables,
        allowed_artifacts=allowed_artifacts,
    )
    if not result["ok"]:
        return f"Rejected: {result['error']}"
    if not result["rows"]:
        return "Query ran successfully but returned no rows."
    if result["truncated"] and can_materialize:
        artifact = materialize_result(
            query,
            owner_id,
            allowed_tables or set(),
            allowed_artifacts or set(),
        )
        preview = _csv(
            [column["name"] for column in artifact["columns"]],
            artifact["preview"],
        )
        return "[Artifact]\n" + json.dumps(artifact, default=str) + f"\nPreview:\n{preview}"

    text = _csv(result["columns"], result["rows"])
    if result["truncated"]:
        text += "\n[Results truncated]"
    return text


@tool
def run_sql(query: str) -> str:
    """Run one validated, read-only SQL query against the connected database."""
    return run_managed_sql(query)
