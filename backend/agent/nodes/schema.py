from backend.agent.state import QueryState


def make_inspect_schema(schema_runner):
    def inspect_schema(state: QueryState) -> QueryState:
        try:
            discovered = schema_runner(state["question"], state)
            return {
                "schema_context": discovered.get("context", ""),
                "allowed_tables": discovered.get("allowed_tables", []),
                "artifacts": discovered.get("artifacts", state.get("artifacts", [])),
                "error": "",
            }
        except Exception as error:
            return {"error": f"Could not inspect database schema: {error}"}

    return inspect_schema
