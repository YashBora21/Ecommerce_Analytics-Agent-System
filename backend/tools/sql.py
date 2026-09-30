import csv
import io

from langchain_core.tools import tool

from .sql_core import execute_sql


@tool
def run_sql(query: str) -> str:
    """Run one validated, read-only SQL query against the orders table."""
    result = execute_sql(query)
    if not result["ok"]:
        return f"Rejected: {result['error']}"
    if not result["rows"]:
        return "Query ran successfully but returned no rows."

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(result["columns"])
    for row in result["rows"]:
        writer.writerow([row[column] for column in result["columns"]])

    text = output.getvalue().strip()
    if result["truncated"]:
        text += "\n[Results truncated]"
    return text
