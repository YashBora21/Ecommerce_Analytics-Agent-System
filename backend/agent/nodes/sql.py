import json

from backend.agent.state import QueryState, SqlRunner


def _sql_error(query: str, tries: int, analysis: dict, error: str) -> QueryState:
    return {
        "sql": query,
        "sql_tries": tries,
        "analysis": analysis,
        "action": {},
        "error": error,
    }


def make_execute_query(sql_runner: SqlRunner, sql_analyzer):
    def execute_query(state: QueryState) -> QueryState:
        args = state["action"]["args"]
        query = args["sql"]
        tries = state.get("sql_tries", 0) + 1
        previous_hash = state.get("analysis", {}).get("sql_hash")

        analysis = sql_analyzer(query, state)
        if analysis.get("status") != "safe":
            reasons = "; ".join(analysis.get("reasons", ["query was rejected"]))
            return _sql_error(
                query,
                tries,
                analysis,
                f"SQL {analysis.get('status', 'rejected')}: {reasons}",
            )
        if args.get("result_scope", "complete") == "complete" and analysis.get("has_limit"):
            return _sql_error(
                query,
                tries,
                analysis,
                "SQL rejected: a complete result cannot contain LIMIT",
            )
        query = analysis.get("executable_sql", query)

        sql_hash = analysis.get("sql_hash", "")
        if tries > 1 and sql_hash and sql_hash == previous_hash:
            return _sql_error(
                query,
                tries,
                analysis,
                "SQL rejected: this exact query already failed; revise it before retrying",
            )
        if sql_hash and any(
            item.get("sql_hash") == sql_hash for item in state.get("evidence", [])
        ):
            return _sql_error(
                query,
                tries,
                analysis,
                "SQL rejected: this query was already executed",
            )

        result = sql_runner(query, {**state, "analysis": analysis})
        if result.startswith("Rejected:"):
            return _sql_error(query, tries, analysis, result)

        evidence = list(state.get("evidence", []))
        item = {
            "kind": "sql",
            "sql": query,
            "sql_hash": sql_hash,
            "result": result,
            "complete": "[Results truncated]" not in result,
            "result_scope": args.get("result_scope", "complete"),
        }
        artifacts = list(state.get("artifacts", []))
        if result.startswith("[Artifact]\n"):
            artifact = json.loads(result.split("\n", 2)[1])
            artifacts.append({
                "artifact_id": artifact["artifact_id"],
                "relation": artifact["relation"],
                "row_count": artifact["row_count"],
                "byte_size": artifact.get("byte_size", 0),
                "columns": artifact["columns"],
                "complete": artifact.get("complete", True),
                "has_limit": bool(artifact.get("has_limit", False)),
                "expires_at": artifact["expires_at"],
                "sql_hash": artifact.get("sql_hash", sql_hash),
            })
            has_limit = artifact.get("has_limit", False)
            item["artifact_id"] = artifact["artifact_id"]
            item["has_limit"] = has_limit
            item["result"] = (
                f"Artifact {artifact['artifact_id']} at {artifact['relation']}; "
                f"rows={artifact['row_count']}; has_limit={has_limit}; columns={artifact['columns']}; "
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
