import csv
import io

import plotly.graph_objects as go
from langchain_core.tools import tool


CHART_TYPES = {"bar", "line", "scatter", "pie"}


def _values(rows: list[dict[str, str]], column: str) -> list[str | float]:
    if not rows or column not in rows[0]:
        raise ValueError(f"Column not found: {column}")

    values = []
    for row in rows:
        value = row[column]
        try:
            values.append(float(value))
        except ValueError:
            values.append(value)
    return values


def create_chart(
    csv_data: str,
    chart_type: str,
    x_column: str,
    y_column: str,
    title: str = "",
) -> str:
    """Create Plotly JSON from complete CSV-formatted SQL results."""
    if "[Results truncated]" in csv_data:
        raise ValueError("Charts require complete SQL results")

    chart_type = chart_type.lower().strip()
    if chart_type not in CHART_TYPES:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    rows = list(csv.DictReader(io.StringIO(csv_data)))
    x_values = _values(rows, x_column)
    y_values = _values(rows, y_column)

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
    x_column: str,
    y_column: str,
    title: str = "",
) -> str:
    """Create a Plotly chart from actual CSV-formatted SQL query results."""
    try:
        return create_chart(csv_data, chart_type, x_column, y_column, title)
    except ValueError as error:
        return f"Rejected: {error}"
