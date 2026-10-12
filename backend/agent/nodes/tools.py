import json

from backend.agent.memory import sql_result
from backend.agent.state import QueryState


def _available_artifact_columns(state: QueryState) -> list[str]:
    columns = []
    for art in state.get("artifacts", []):
        for col in art.get("columns", []):
            name = col.get("name") if isinstance(col, dict) else str(col)
            if name and name not in columns:
                columns.append(name)
    return columns


def _owned_artifact(state: QueryState, artifact_id: str) -> dict | None:
    return next(
        (
            artifact
            for artifact in state.get("artifacts", [])
            if artifact.get("artifact_id") == artifact_id
            and artifact.get("complete")
        ),
        None,
    )


def _tool_error(
    state: QueryState,
    counter: str,
    tries: int,
    message: str,
) -> QueryState:
    columns = _available_artifact_columns(state)
    return {
        counter: tries,
        "action": {},
        "error": f"{message}. Available artifact columns: {columns}",
    }


def _statistics_answer(item: dict) -> str:
    result = json.loads(item["result"])
    operation = result["operation"].replace("_", " ").title()
    columns = result["column"].replace("_", " ")
    if result.get("second_column"):
        columns += f" and {result['second_column'].replace('_', ' ')}"
    value = result["value"]
    if isinstance(value, dict):
        value = ", ".join(f"{name}={number}" for name, number in value.items())
    return f"{operation} for {columns}: {value} (based on {result['count']} rows)."


def make_calculate_statistic(statistics_runner, remember):
    def calculate_statistic(state: QueryState) -> QueryState:
        tries = state.get("statistics_tries", 0) + 1
        args = dict(state["action"]["args"])
        required = {args["column"]}
        if args.get("second_column"):
            required.add(args["second_column"])

        artifact_id = args.get("artifact_id")
        source = artifact_id or next(
            (
                item.get("sql_hash") or item.get("sql")
                for item in reversed(state.get("evidence", []))
                if item.get("kind", "sql") == "sql"
            ),
            "",
        )
        request = {
            "operation": args["operation"],
            "column": args["column"],
            "second_column": args.get("second_column") or "",
            "percentile": args.get("percentile"),
            "source": source,
        }
        existing = next(
            (
                item
                for item in reversed(state.get("evidence", []))
                if item.get("kind") == "statistics"
                and item.get("request") == request
            ),
            None,
        )
        if existing:
            answer = _statistics_answer(existing)
            return {
                **remember(state, answer),
                "statistics_tries": 0,
                "action": {"name": "answer", "args": {"final_answer": answer}},
                "error": "",
            }

        if artifact_id:
            artifact = _owned_artifact(state, artifact_id)
            if not artifact:
                return _tool_error(
                    state,
                    "statistics_tries",
                    tries,
                    "Rejected: artifact is not owned by this thread",
                )
            if artifact.get("has_limit"):
                return _tool_error(
                    state,
                    "statistics_tries",
                    tries,
                    "Rejected: statistics require an artifact without LIMIT",
                )
            args["artifact_id"] = artifact_id
            args["owner_id"] = state.get("owner_id", "")
        else:
            try:
                args["csv_data"] = sql_result(state, required)
            except ValueError as error:
                return _tool_error(
                    state,
                    "statistics_tries",
                    tries,
                    f"Rejected: {error}",
                )

        result = statistics_runner(args)
        if result.startswith("Rejected:"):
            return _tool_error(state, "statistics_tries", tries, result)

        evidence = list(state.get("evidence", []))
        label = f"STATISTICS {args['operation']}({args['column']})"
        evidence.append({
            "kind": "statistics",
            "sql": label,
            "result": result,
            "request": request,
        })
        return {
            "evidence": evidence,
            "statistics_tries": 0,
            "action": {},
            "error": "",
        }

    return calculate_statistic


def make_create_plot(chart_runner, remember):
    def create_plot(state: QueryState) -> QueryState:
        tries = state.get("chart_tries", 0) + 1
        args = dict(state["action"]["args"])
        answer = args.pop("final_answer")
        required = {column for column in (
            args.get("x_column"),
            args.get("y_column"),
            args.get("value_column"),
            *(args.get("path_columns") or []),
        ) if column}

        artifact_id = args.get("artifact_id")
        if artifact_id:
            if not _owned_artifact(state, artifact_id):
                return _tool_error(
                    state,
                    "chart_tries",
                    tries,
                    "Rejected: artifact is not owned by this thread",
                )
            args["artifact_id"] = artifact_id
            args["owner_id"] = state.get("owner_id", "")
        else:
            try:
                args["csv_data"] = sql_result(state, required)
            except ValueError as error:
                return _tool_error(
                    state,
                    "chart_tries",
                    tries,
                    f"Rejected: {error}",
                )

        result = chart_runner(args)
        if result.startswith("Rejected:"):
            return _tool_error(state, "chart_tries", tries, result)

        return {
            **remember(state, answer),
            "chart": result,
            "chart_tries": 0,
            "action": {},
            "error": "",
        }

    return create_plot
