import csv
import io
import os

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
    csv_data: str = "",
    chart_type: str = "",
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    value_column: str = "",
    path_columns: list[str] | None = None,
    artifact_id: str = "",
    owner_id: str = "",
    database_url: str | None = None,
) -> str:
    """Create Plotly JSON from complete SQL results or artifact."""
    chart_type = chart_type.lower().strip()
    if chart_type not in SUPPORTED_CHARTS:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    if artifact_id:
        db_url = database_url or os.getenv("DATABASE_URL")
        if not db_url:
            raise ValueError("DATABASE_URL is required to read artifact")
        from backend.db.postgres import postgres_connection
        from backend.tools.artifacts import owned_artifact
        from psycopg import sql

        artifact = owned_artifact(artifact_id, owner_id, db_url)
        column_names = {item["name"] for item in artifact["columns"]}
        needed = list(dict.fromkeys(
            column for column in (
                x_column,
                y_column,
                value_column,
                *(path_columns or []),
            )
            if column
        ))
        for column in needed:
            if column not in column_names:
                raise ValueError(f"Column not found: {column}")
        row_limit = {
            "heatmap": MAX_HEATMAP_CELLS,
            "treemap": MAX_TREEMAP_LEAVES,
        }.get(chart_type, MAX_CHART_POINTS)
        relation = sql.SQL("{}.{}").format(
            sql.Identifier("agent_results"),
            sql.Identifier(artifact["relation_name"]),
        )
        query = sql.SQL("SELECT {} FROM {} LIMIT %s").format(
            sql.SQL(", ").join(sql.Identifier(column) for column in needed),
            relation,
        )
        with postgres_connection(db_url) as connection:
            database_rows = connection.execute(query, (row_limit + 1,)).fetchall()
        if len(database_rows) > row_limit:
            raise ValueError(f"Chart exceeds {row_limit} rows; aggregate the artifact first")
        rows = [
            {
                column: str(row[column]) if row[column] is not None else ""
                for column in needed
            }
            for row in database_rows
        ]
    else:
        if "[Results truncated]" in csv_data:
            raise ValueError("Charts require complete SQL results")
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
    csv_data: str = "",
    chart_type: str = "",
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    value_column: str = "",
    path_columns: list[str] | None = None,
    artifact_id: str = "",
    owner_id: str = "",
    database_url: str | None = None,
) -> str:
    """Create a supported Plotly chart from actual SQL query results or an artifact."""
    try:
        return create_chart(
            csv_data=csv_data,
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column,
            title=title,
            value_column=value_column,
            path_columns=path_columns,
            artifact_id=artifact_id,
            owner_id=owner_id,
            database_url=database_url,
        )
    except (ValueError, csv.Error) as error:
        return f"Rejected: {error}"
