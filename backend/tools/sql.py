import csv
import io
import json
import os

from langchain_core.tools import tool

from .artifacts import materialize_result
from .sql_core import execute_sql


def to_csv(columns, rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[c] for c in columns])
    return buf.getvalue().strip()


def run_managed_sql(
    query: str,
    owner_id: str = "",
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
    estimated_rows: int = 0,
) -> str:
    """Return small results inline; save big ones as an artifact."""
    result = execute_sql(
        query,
        allowed_tables=allowed_tables,
        allowed_artifacts=allowed_artifacts,
    )

    if not result["ok"]:
        return f"Rejected: {result['error']}"
    if not result["rows"]:
        return "Query ran successfully but returned no rows."

    truncated = result["truncated"]

    # only worth saving as an artifact if the result got cut off
    # and we actually have somewhere to store it
    if truncated and owner_id and os.getenv("DATABASE_URL"):
        try:
            artifact = materialize_result(
                query,
                owner_id,
                allowed_tables or set(),
                allowed_artifacts or set(),
            )
        except Exception as error:
            return f"Rejected: {type(error).__name__}: {error}"

        names = [c["name"] for c in artifact["columns"]]
        preview = to_csv(names, artifact["preview"])
        return f"[Artifact]\n{json.dumps(artifact, default=str)}\nPreview:\n{preview}"

    text = to_csv(result["columns"], result["rows"])
    if truncated:
        text += "\n[Results truncated]"
    return text


@tool
def run_sql(query: str) -> str:
    """Run one validated, read-only SQL query against the connected database."""
    return run_managed_sql(query)
