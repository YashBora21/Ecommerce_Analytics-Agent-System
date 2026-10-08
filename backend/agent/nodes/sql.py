import json

from backend.agent.state import QueryState, SqlRunner


def make_execute_query(sql_runner: SqlRunner, sql_analyzer):
    def execute_query(state: QueryState) -> QueryState:
        query = state["action"]["args"]["sql"]
        tries = state.get("sql_tries", 0) + 1

        analysis = sql_analyzer(query, state)
        if analysis.get("status") != "safe":
            reasons = "; ".join(analysis.get("reasons", ["query was rejected"]))
            return {
                "sql": query,
                "sql_tries": tries,
                "analysis": analysis,
                "action": {},
                "error": f"SQL {analysis.get('status', 'rejected')}: {reasons}",
            }

        sql_hash = analysis.get("sql_hash", "")
        if sql_hash and any(
            item.get("sql_hash") == sql_hash for item in state.get("evidence", [])
        ):
            return {
                "sql": query,
                "sql_tries": tries,
                "analysis": analysis,
                "action": {},
                "error": "SQL rejected: this query was already executed",
            }

        result = sql_runner(query, {**state, "analysis": analysis})
        if result.startswith("Rejected:"):
            return {
                "sql": query,
                "sql_tries": tries,
                "analysis": analysis,
                "action": {},
                "error": result,
            }

        evidence = list(state.get("evidence", []))
        item = {
            "kind": "sql",
            "sql": query,
            "sql_hash": sql_hash,
            "result": result,
            "complete": "[Results truncated]" not in result,
        }
        artifacts = list(state.get("artifacts", []))
        if result.startswith("[Artifact]\n"):
            artifact = json.loads(result.split("\n", 2)[1])
            artifacts.append(artifact)
            item["artifact_id"] = artifact["artifact_id"]
            item["result"] = (
                f"Artifact {artifact['artifact_id']} at {artifact['relation']}; "
                f"rows={artifact['row_count']}; columns={artifact['columns']}; "
                f"expires_at={artifact['expires_at']}"
            )
            item["complete"] = True
        evidence.append(item)
        return {
            "sql": query,
            "analysis": analysis,
            "evidence": evidence,
            "artifacts": artifacts,
            "sql_tries": 0,
            "action": {},
            "error": "",
        }

    return execute_query
