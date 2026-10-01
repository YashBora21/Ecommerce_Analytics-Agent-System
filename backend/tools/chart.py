import csv
import io

import plotly.graph_objects as go
from langchain_core.tools import tool


SUPPORTED_CHARTS = {"bar", "line", "scatter", "pie", "heatmap", "treemap"}
MAX_CHART_POINTS = 100
MAX_PIE_SLICES = 20
MAX_HEATMAP_CELLS = 400
MAX_TREEMAP_LEAVES = 100


def _require_columns(rows: list[dict[str, str]], columns: list[str]) -> None:
    for column in columns:
        if not column or column not in rows[0]:
            raise ValueError(f"Column not found: {column}")


def _numbers(rows: list[dict[str, str]], column: str) -> list[float]:
    _require_columns(rows, [column])
    try:
        return [float(row[column]) for row in rows]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Column must be numeric: {column}") from error


def _heatmap(rows, x_column, y_column, value_column):
    _require_columns(rows, [x_column, y_column, value_column])
    x_values = list(dict.fromkeys(row[x_column] for row in rows))
    y_values = list(dict.fromkeys(row[y_column] for row in rows))
    if len(x_values) * len(y_values) > MAX_HEATMAP_CELLS:
        raise ValueError(f"Heatmap exceeds {MAX_HEATMAP_CELLS} cells")

    totals = {}
    for row, value in zip(rows, _numbers(rows, value_column)):
        key = (row[x_column], row[y_column])
        totals[key] = totals.get(key, 0) + value
    grid = [[totals.get((x, y)) for x in x_values] for y in y_values]
    return go.Heatmap(x=x_values, y=y_values, z=grid)


def _treemap(rows, path_columns, value_column):
    if not path_columns:
        raise ValueError("Treemap requires path_columns")
    _require_columns(rows, [*path_columns, value_column])

    leaf_paths = {tuple(row[column] for column in path_columns) for row in rows}
    if len(leaf_paths) > MAX_TREEMAP_LEAVES:
        raise ValueError(f"Treemap exceeds {MAX_TREEMAP_LEAVES} leaves")

    totals = {}
    for row, value in zip(rows, _numbers(rows, value_column)):
        path = tuple(row[column] for column in path_columns)
        for depth in range(1, len(path) + 1):
            totals[path[:depth]] = totals.get(path[:depth], 0) + value

    paths = list(totals)
    ids = [" / ".join(path) for path in paths]
    parents = [" / ".join(path[:-1]) for path in paths]
    labels = [path[-1] for path in paths]
    return go.Treemap(
        ids=ids,
        parents=parents,
        labels=labels,
        values=[totals[path] for path in paths],
        branchvalues="total",
    )


def create_chart(
    csv_data: str,
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    value_column: str = "",
    path_columns: list[str] | None = None,
) -> str:
    """Create Plotly JSON from complete CSV-formatted SQL results."""
    if "[Results truncated]" in csv_data:
        raise ValueError("Charts require complete SQL results")

    chart_type = chart_type.lower().strip()
    if chart_type not in SUPPORTED_CHARTS:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    rows = list(csv.DictReader(io.StringIO(csv_data)))
    if not rows:
        raise ValueError("Chart requires at least one result row")

    if chart_type == "heatmap":
        trace = _heatmap(rows, x_column, y_column, value_column)
    elif chart_type == "treemap":
        trace = _treemap(rows, path_columns or [], value_column)
    else:
        _require_columns(rows, [x_column, y_column])
        if len(rows) > MAX_CHART_POINTS:
            raise ValueError(f"Chart exceeds {MAX_CHART_POINTS} points")
        x_values = [row[x_column] for row in rows]
        y_values = _numbers(rows, y_column)
        if chart_type == "pie" and len(set(x_values)) > MAX_PIE_SLICES:
            raise ValueError(f"Pie chart exceeds {MAX_PIE_SLICES} slices")
        if chart_type == "bar":
            trace = go.Bar(x=x_values, y=y_values)
        elif chart_type == "line":
            trace = go.Scatter(x=x_values, y=y_values, mode="lines+markers")
        elif chart_type == "scatter":
            trace = go.Scatter(x=x_values, y=y_values, mode="markers")
        else:
            trace = go.Pie(labels=x_values, values=y_values)

    figure = go.Figure(trace)
    figure.update_layout(title=title, xaxis_title=x_column, yaxis_title=y_column)
    return figure.to_json()


@tool
def run_chart(
    csv_data: str,
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    value_column: str = "",
    path_columns: list[str] | None = None,
) -> str:
    """Create a supported Plotly chart from actual SQL query results."""
    try:
        return create_chart(
            csv_data,
            chart_type,
            x_column,
            y_column,
            title,
            value_column,
            path_columns,
        )
    except (ValueError, csv.Error) as error:
        return f"Rejected: {error}"

