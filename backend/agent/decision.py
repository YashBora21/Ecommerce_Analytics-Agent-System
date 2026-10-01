from backend.tools.chart import SUPPORTED_CHARTS
from backend.tools.statistics import SUPPORTED_OPERATIONS

ACTIONS = {"answer", "query_sql", "calculate_statistics", "create_chart"}


def _text(args, key, required=True):
    value = args.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    value = value.strip()
    if required and not value:
        raise ValueError(f"{key.replace('_', ' ')} is empty")
    return value


def _check_answer(args):
    return {"final_answer": _text(args, "final_answer")}


def _check_query(args):
    return {"next_query_goal": _text(args, "next_query_goal")}


def _check_statistics(args):
    operation = _text(args, "operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported statistics operation: {operation}")
    column = _text(args, "column")
    second_column = _text(args, "second_column", False)
    percentile = args.get("percentile")
    if operation == "correlation" and not second_column:
        raise ValueError("correlation requires second_column")
    if operation == "percentile" and not isinstance(percentile, (int, float)):
        raise ValueError("percentile requires a number")
    return {
        "operation": operation,
        "column": column,
        "second_column": second_column,
        "percentile": percentile,
    }


def _check_chart(args):
    chart_type = _text(args, "chart_type")
    if chart_type not in SUPPORTED_CHARTS:
        raise ValueError(f"unsupported chart type: {chart_type}")
    x_column = _text(args, "x_column", False)
    y_column = _text(args, "y_column", False)
    value_column = _text(args, "value_column", False)
    title = _text(args, "title", False)
    final_answer = _text(args, "final_answer")
    path_columns = args.get("path_columns", [])
    if not isinstance(path_columns, list) or not all(
        isinstance(column, str) and column.strip() for column in path_columns
    ):
        raise ValueError("path_columns must be a list of column names")
    if chart_type == "heatmap" and not all((x_column, y_column, value_column)):
        raise ValueError("heatmap requires x, y, and value columns")
    if chart_type == "treemap" and (not path_columns or not value_column):
        raise ValueError("treemap requires path and value columns")
    if chart_type not in {"heatmap", "treemap"} and not all((x_column, y_column)):
        raise ValueError("chart requires x and y columns")
    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
        "value_column": value_column,
        "path_columns": [column.strip() for column in path_columns],
        "title": title,
        "final_answer": final_answer,
    }


_VALIDATORS = {
    "answer": _check_answer,
    "query_sql": _check_query,
    "calculate_statistics": _check_statistics,
    "create_chart": _check_chart,
}


def parse_action(decision: dict) -> dict:
    if not isinstance(decision, dict):
        raise ValueError("decision must be an object")
    name = decision.get("action")
    if name not in ACTIONS:
        raise ValueError("invalid next action")
    if not isinstance(decision.get("reason"), str):
        raise ValueError("action reason must be text")
    args = decision.get("arguments")
    if not isinstance(args, dict):
        raise ValueError("arguments must be an object")
    return {"name": name, "args": _VALIDATORS[name](args)}



