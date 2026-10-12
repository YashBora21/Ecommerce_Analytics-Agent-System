import csv
import io
import json
import os
import statistics
from typing import Any

import psycopg
from langchain_core.tools import tool


SUPPORTED_OPERATIONS = {
    "mean", "median", "min", "max", "stdev", "variance", "percentile",
    "correlation", "linear_regression",
}
SIMPLE_OPERATIONS = {
    "mean": statistics.fmean,
    "median": statistics.median,
    "min": min,
    "max": max,
    "stdev": statistics.stdev,
    "variance": statistics.variance,
}


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
    percentile = _percentile_percent(percentile)
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _percentile_percent(percentile: float) -> float:
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    return percentile * 100 if 0 < percentile < 1 else percentile


def _artifact_statistic(
    artifact_id: str,
    owner_id: str,
    database_url: str,
    operation: str,
    column: str,
    second_column: str,
    percentile: float | None,
) -> dict[str, Any]:
    from psycopg import sql

    from backend.db.postgres import postgres_connection
    from backend.tools.artifacts import owned_artifact

    artifact = owned_artifact(artifact_id, owner_id, database_url)
    if artifact["has_limit"]:
        raise ValueError("Statistics require an artifact without LIMIT")
    column_names = {item["name"] for item in artifact["columns"]}
    if column not in column_names:
        raise ValueError(f"Column not found: {column}")
    if second_column and second_column not in column_names:
        raise ValueError(f"Column not found: {second_column}")

    first = sql.Identifier(column)
    second = sql.Identifier(second_column) if second_column else None
    relation = sql.SQL("{}.{}").format(
        sql.Identifier("agent_results"),
        sql.Identifier(artifact["relation_name"]),
    )
    where = sql.SQL("{} IS NOT NULL").format(first)
    if second is not None:
        where += sql.SQL(" AND {} IS NOT NULL").format(second)

    parameters = ()
    if operation == "mean":
        expression = sql.SQL("AVG({})::float8 AS value").format(first)
    elif operation == "median":
        expression = sql.SQL(
            "percentile_cont(0.5) WITHIN GROUP (ORDER BY {})::float8 AS value"
        ).format(first)
    elif operation == "min":
        expression = sql.SQL("MIN({})::float8 AS value").format(first)
    elif operation == "max":
        expression = sql.SQL("MAX({})::float8 AS value").format(first)
    elif operation == "stdev":
        expression = sql.SQL("STDDEV_SAMP({})::float8 AS value").format(first)
    elif operation == "variance":
        expression = sql.SQL("VAR_SAMP({})::float8 AS value").format(first)
    elif operation == "percentile":
        if percentile is None:
            raise ValueError("percentile value is required")
        percentile = _percentile_percent(percentile)
        expression = sql.SQL(
            "percentile_cont(%s) WITHIN GROUP (ORDER BY {})::float8 AS value"
        ).format(first)
        parameters = (percentile / 100,)
    elif operation == "correlation":
        if second is None:
            raise ValueError("second_column is required for correlation")
        expression = sql.SQL("CORR({}, {})::float8 AS value").format(first, second)
    else:
        if second is None:
            raise ValueError("second_column is required for linear_regression")
        expression = sql.SQL(
            "REGR_SLOPE({}, {})::float8 AS slope, "
            "REGR_INTERCEPT({}, {})::float8 AS intercept"
        ).format(second, first, second, first)

    query = sql.SQL("SELECT {}, COUNT(*) AS count FROM {} WHERE {}").format(
        expression,
        relation,
        where,
    )
    with postgres_connection(database_url) as connection:
        result = connection.execute(query, parameters).fetchone()
    if not result["count"]:
        raise ValueError(f"Column has no numeric values: {column}")

    if operation == "linear_regression":
        if result["slope"] is None:
            raise ValueError("Not enough values for linear_regression")
        value = {
            "slope": round(float(result["slope"]), 6),
            "intercept": round(float(result["intercept"]), 6),
        }
    else:
        if result["value"] is None:
            raise ValueError(f"Not enough values for {operation}")
        value = round(float(result["value"]), 6)
    return {
        "ok": True,
        "operation": operation,
        "column": column,
        "second_column": second_column or None,
        "value": value,
        "count": result["count"],
        "error": None,
    }


def calculate_statistics(
    csv_data: str = "",
    operation: str = "",
    column: str = "",
    second_column: str = "",
    percentile: float | None = None,
    artifact_id: str = "",
    owner_id: str = "",
    database_url: str | None = None,
) -> dict[str, Any]:
    """Calculate one statistic from an artifact via Postgres SQL or from CSV-formatted results."""
    try:
        operation = operation.lower().strip()
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}")

        if artifact_id:
            db_url = database_url or os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL is required to read artifact")
            return _artifact_statistic(
                artifact_id,
                owner_id,
                db_url,
                operation,
                column,
                second_column,
                percentile,
            )
        if "[Results truncated]" in csv_data:
            raise ValueError("Statistics require complete SQL results")
        rows = list(csv.DictReader(io.StringIO(csv_data)))
        first_values = _numbers(rows, column)
        second_values = _numbers(rows, second_column) if second_column else []

        if operation in SIMPLE_OPERATIONS:
            value = SIMPLE_OPERATIONS[operation](first_values)
        elif operation == "percentile":
            if percentile is None:
                raise ValueError("percentile value is required")
            value = _percentile(first_values, percentile)
        elif operation == "correlation":
            if not second_column:
                raise ValueError("second_column is required for correlation")
            if not second_values:
                raise ValueError(f"Column has no numeric values: {second_column}")
            if len(first_values) != len(second_values):
                raise ValueError("Correlation columns must have the same number of values")
            value = statistics.correlation(first_values, second_values)
        else:
            if not second_column:
                raise ValueError("second_column is required for linear_regression")
            if not second_values:
                raise ValueError(f"Column has no numeric values: {second_column}")
            if len(first_values) != len(second_values):
                raise ValueError("Regression columns must have the same number of values")
            regression = statistics.linear_regression(first_values, second_values)
            value = {
                "slope": round(regression.slope, 6),
                "intercept": round(regression.intercept, 6),
            }

        return {
            "ok": True,
            "operation": operation,
            "column": column,
            "second_column": second_column or None,
            "value": round(value, 6) if isinstance(value, float) else value,
            "count": len(first_values),
            "error": None,
        }
    except (ValueError, statistics.StatisticsError, psycopg.Error) as error:
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
    csv_data: str = "",
    operation: str = "",
    column: str = "",
    second_column: str = "",
    percentile: float | None = None,
    artifact_id: str = "",
    owner_id: str = "",
    database_url: str | None = None,
) -> str:
    """Calculate a statistic from actual CSV-formatted SQL query results or an artifact."""
    result = calculate_statistics(
        csv_data=csv_data,
        operation=operation,
        column=column,
        second_column=second_column,
        percentile=percentile,
        artifact_id=artifact_id,
        owner_id=owner_id,
        database_url=database_url,
    )
    if not result["ok"]:
        return f"Rejected: {result['error']}"
    return json.dumps(result, separators=(",", ":"))

