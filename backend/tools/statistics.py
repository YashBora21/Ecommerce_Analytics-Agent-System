import csv
import io
import json
import statistics
from typing import Any

from langchain_core.tools import tool


SUPPORTED_OPERATIONS = {"mean", "median", "min", "max", "stdev", "variance", "percentile", "correlation"}


def _numbers(rows: list[dict[str, str]], column: str) -> list[float]:
    if not rows or column not in rows[0]:
        raise ValueError(f"Column not found: {column}")

    values = []
    for row_number, row in enumerate(rows, start=1):
        value = row[column].strip()
        if not value:
            continue
        try:
            values.append(float(value))
        except ValueError as error:
            raise ValueError(
                f"Column {column} contains a nonnumeric value on row {row_number}"
            ) from error
    if not values:
        raise ValueError(f"Column has no numeric values: {column}")
    return values


def _percentile(values: list[float], percentile: float) -> float:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def calculate_statistics(
    csv_data: str,
    operation: str,
    column: str,
    second_column: str = "",
    percentile: float | None = None,
) -> dict[str, Any]:
    """Calculate one statistic from CSV-formatted SQL results."""
    try:
        if "[Results truncated]" in csv_data:
            raise ValueError("Statistics require complete SQL results")

        operation = operation.lower().strip()
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}")

        rows = list(csv.DictReader(io.StringIO(csv_data)))
        first_values = _numbers(rows, column)

        if operation == "mean":
            value = statistics.fmean(first_values)
        elif operation == "median":
            value = statistics.median(first_values)
        elif operation == "min":
            value = min(first_values)
        elif operation == "max":
            value = max(first_values)
        elif operation == "stdev":
            value = statistics.stdev(first_values)
        elif operation == "variance":
            value = statistics.variance(first_values)
        elif operation == "percentile":
            if percentile is None:
                raise ValueError("percentile value is required")
            value = _percentile(first_values, percentile)
        else:
            if not second_column:
                raise ValueError("second_column is required for correlation")
            second_values = _numbers(rows, second_column)
            if len(first_values) != len(second_values):
                raise ValueError("Correlation columns must have the same number of values")
            value = statistics.correlation(first_values, second_values)

        return {
            "ok": True,
            "operation": operation,
            "column": column,
            "second_column": second_column or None,
            "value": round(value, 6),
            "count": len(first_values),
            "error": None,
        }
    except (ValueError, statistics.StatisticsError) as error:
        return {
            "ok": False,
            "operation": operation,
            "column": column,
            "second_column": second_column or None,
            "value": None,
            "count": 0,
            "error": str(error),
        }


@tool
def run_statistics(
    csv_data: str,
    operation: str,
    column: str,
    second_column: str = "",
    percentile: float | None = None,
) -> str:
    """Calculate a statistic from actual CSV-formatted SQL query results."""
    result = calculate_statistics(
        csv_data,
        operation,
        column,
        second_column,
        percentile,
    )
    if not result["ok"]:
        return f"Rejected: {result['error']}"
    return json.dumps(result, separators=(",", ":"))




