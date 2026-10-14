from backend.agent.state import QueryState


def make_inspect_schema(schema_runner):
    def inspect_schema(state: QueryState) -> QueryState:
        try:
            discovered = schema_runner(state["question"], state)
            artifacts = discovered.get("artifacts", state.get("artifacts", []))
            if discovered.get("selection_failed") and not artifacts:
                return {
                    "artifacts": artifacts,
                    "error": (
                        "No relevant database tables matched the question. "
                        "Please name the business entity or metric more specifically."
                    ),
                }
            return {
                "schema_context": discovered.get("context", ""),
                "allowed_tables": discovered.get("allowed_tables", []),
                "artifacts": artifacts,
                "error": "",
            }
        except Exception as error:
            return {"error": f"Could not inspect database schema: {error}"}

    return inspect_schema
