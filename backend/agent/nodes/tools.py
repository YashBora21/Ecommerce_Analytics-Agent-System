from backend.agent.memory import sql_result
from backend.agent.state import QueryState


def make_calculate_statistic(statistics_runner):
    def calculate_statistic(state: QueryState) -> QueryState:
        args = dict(state["action"]["args"])
        required = {args["column"]}
        if args.get("second_column"):
            required.add(args["second_column"])
        try:
            args["csv_data"] = sql_result(state, required)
        except ValueError as error:
            return {"error": f"Rejected: {error}"}
        result = statistics_runner(args)
        if result.startswith("Rejected:"):
            return {"error": result}
        evidence = list(state.get("evidence", []))
        label = f"STATISTICS {args['operation']}({args['column']})"
        evidence.append({"kind": "statistics", "sql": label, "result": result})
        return {"evidence": evidence, "action": {}, "error": ""}

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
        try:
            args["csv_data"] = sql_result(state, required)
        except ValueError as error:
            return {
                "chart_tries": tries,
                "action": {},
                "error": f"Rejected: {error}",
            }
        result = chart_runner(args)
        if result.startswith("Rejected:"):
            return {
                "chart_tries": tries,
                "action": {},
                "error": result,
            }
        return {
            **remember(state, answer),
            "chart": result,
            "chart_tries": tries,
            "action": {},
            "error": "",
        }

    return create_plot


