import hashlib
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import tuple_row
from sqlglot import exp, parse
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import traverse_scope

from backend.db.postgres import postgres_connection


BLOCKED_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}
BLOCKED_FUNCTIONS = {
    "dblink",
    "dblink_exec",
    "lo_export",
    "lo_import",
    "pg_read_binary_file",
    "pg_read_file",
    "pg_sleep",
    "pg_terminate_backend",
    "pg_write_file",
}
DEFAULT_MAX_PLAN_COST = float(os.getenv("MAX_SQL_PLAN_COST", "1000000"))
DEFAULT_MAX_PLAN_ROWS = int(os.getenv("MAX_SQL_PLAN_ROWS", "5000000"))
DEFAULT_MAX_PLAN_BYTES = int(os.getenv("MAX_SQL_PLAN_BYTES", str(512 * 1024 * 1024)))


def _table_name(table: exp.Table) -> str:
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    return ".".join(parts)



def _predicate_depth(node: exp.Expression | None) -> int:
    if node is None:
        return 0
    children = [child for child in node.iter_expressions()]
    child_depth = max((_predicate_depth(child) for child in children), default=0)
    return child_depth + (1 if isinstance(node, (exp.And, exp.Or, exp.Not)) else 0)


def validate_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    if "\x00" in sql:
        raise ValueError("SQL cannot contain null bytes")
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("SQL comments are not allowed")
    try:
        statements = [statement for statement in parse(sql, read="postgres") if statement]
    except ParseError as error:
        raise ValueError(f"Invalid SQL: {error}") from error
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Only SELECT queries are allowed")
    blocked_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Command)
    if any(tree.find(node_type) for node_type in blocked_nodes) or tree.find(exp.Into):
        raise ValueError("Only SELECT queries are allowed")
    if tree.find(exp.Lock):
        raise ValueError("Row-locking SELECT queries are not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = []
    allowed = {name.lower() for name in allowed_tables or set()}
    artifacts = {name.lower() for name in allowed_artifacts or set()}
    for table in tree.find_all(exp.Table):
        name = _table_name(table)
        if table.name.lower() in cte_names:
            continue
        if table.db and table.db.lower() in BLOCKED_SCHEMAS:
            raise ValueError(f"System schema is not allowed: {table.db}")
        lower_name = name.lower()
        short_name = table.name.lower()
        if table.db and table.db.lower() == "agent_results":
            if lower_name not in artifacts:
                raise ValueError(f"Artifact is not owned by this thread: {name}")
        elif allowed and lower_name not in allowed and short_name not in allowed:
            raise ValueError(f"Table is not in the discovered schema: {name}")
        tables.append(name)

    joins = list(tree.find_all(exp.Join))
    for join in joins:
        if str(join.args.get("kind", "")).upper() == "CROSS":
            raise ValueError("CROSS JOIN is not allowed")
        if not join.args.get("on") and not join.args.get("using"):
            raise ValueError("Every JOIN requires an ON or USING condition")

    functions = set()
    for function in tree.find_all(exp.Func):
        name = function.name.lower() if function.name else function.sql_name().lower()
        functions.add(name)
        if name in BLOCKED_FUNCTIONS:
            raise ValueError(f"Function is not allowed: {name}")

    canonical = tree.sql(dialect="postgres", pretty=False, normalize=True)
    with_clause = tree.args.get("with_")
    metrics = {
        "tables": len(set(tables)),
        "joins": len(joins),
        "subqueries": sum(1 for _ in tree.find_all(exp.Subquery)),
        "correlated_subqueries": sum(
            1 for scope in traverse_scope(tree) if scope.is_correlated_subquery
        ),
        "ctes": sum(1 for _ in tree.find_all(exp.CTE)),
        "recursive": bool(with_clause and with_clause.args.get("recursive")),
        "windows": sum(1 for _ in tree.find_all(exp.Window)),
        "aggregates": sum(1 for _ in tree.find_all(exp.AggFunc)),
        "set_operations": sum(1 for _ in tree.find_all(exp.SetOperation)),
        "predicates": sum(1 for _ in tree.find_all(exp.Predicate)),
        "predicate_depth": _predicate_depth(tree.args.get("where")),
    }
    score = (
        metrics["tables"]
        + 3 * metrics["joins"]
        + 3 * metrics["subqueries"]
        + 4 * metrics["correlated_subqueries"]
        + 2 * metrics["ctes"]
        + 5 * int(metrics["recursive"])
        + 2 * metrics["windows"]
        + metrics["aggregates"]
        + 3 * metrics["set_operations"]
        + metrics["predicates"] // 3
        + metrics["predicate_depth"]
    )
    return {
        "canonical_sql": canonical,
        "sql_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "tables": sorted(set(tables)),
        "functions": sorted(functions),
        "structural": metrics,
        "structural_score": score,
    }


def _walk_plan(node: dict[str, Any], depth: int = 1) -> dict[str, Any]:
    children = node.get("Plans", [])
    child_metrics = [_walk_plan(child, depth + 1) for child in children]
    child_rows = [max(int(child.get("Plan Rows", 0)), 1) for child in children]
    multiplier = (
        float(node.get("Plan Rows", 0)) / max(child_rows)
        if len(child_rows) > 1
        else 1.0
    )
    node_type = node.get("Node Type", "")
    return {
        "depth": max([depth, *(item["depth"] for item in child_metrics)]),
        "sequential_scans": (1 if node_type == "Seq Scan" else 0)
        + sum(item["sequential_scans"] for item in child_metrics),
        "nested_loops": (1 if node_type == "Nested Loop" else 0)
        + sum(item["nested_loops"] for item in child_metrics),
        "joins": (1 if "Join" in node_type or node_type == "Nested Loop" else 0)
        + sum(item["joins"] for item in child_metrics),
        "sorts": (1 if node_type in {"Sort", "Incremental Sort"} else 0)
        + sum(item["sorts"] for item in child_metrics),
        "aggregations": (1 if node_type in {"Aggregate", "GroupAggregate", "HashAggregate"} else 0)
        + sum(item["aggregations"] for item in child_metrics),
        "max_join_multiplier": max(
            [multiplier, *(item["max_join_multiplier"] for item in child_metrics)]
        ),
    }


def explain_postgres(sql: str, database_url: str, timeout: float = 5.0) -> dict[str, Any]:
    with postgres_connection(database_url) as connection:
        with connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(timeout * 1000)),))
            cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            root = cursor.fetchone()[0][0]["Plan"]
    operations = _walk_plan(root)
    plan_rows = int(root.get("Plan Rows", 0))
    plan_width = int(root.get("Plan Width", 0))
    return {
        "total_cost": float(root.get("Total Cost", 0)),
        "plan_rows": plan_rows,
        "plan_width": plan_width,
        "estimated_bytes": plan_rows * plan_width,
        **operations,
    }


def analyze_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    try:
        analysis = validate_sql(sql, allowed_tables, allowed_artifacts)
        planner = explain_postgres(sql, database_url) if database_url else {}
        reasons = []
        if planner.get("total_cost", 0) > DEFAULT_MAX_PLAN_COST:
            reasons.append(f"estimated cost {planner['total_cost']:.0f} exceeds {DEFAULT_MAX_PLAN_COST:.0f}")
        if planner.get("plan_rows", 0) > DEFAULT_MAX_PLAN_ROWS:
            reasons.append(f"estimated rows {planner['plan_rows']} exceed {DEFAULT_MAX_PLAN_ROWS}")
        if planner.get("estimated_bytes", 0) > DEFAULT_MAX_PLAN_BYTES:
            reasons.append(
                f"estimated bytes {planner['estimated_bytes']} exceed {DEFAULT_MAX_PLAN_BYTES}"
            )
        return {
            "status": "needs_decomposition" if reasons else "safe",
            "reasons": reasons,
            "planner": planner,
            **analysis,
        }
    except ValueError as error:
        message = str(error)
        unsafe_markers = (
            "Only SELECT",
            "not allowed",
            "not owned",
            "CROSS JOIN",
            "requires an ON",
            "Row-locking",
        )
        status = "unsafe" if any(marker in message for marker in unsafe_markers) else "needs_revision"
        return {"status": status, "reasons": [message]}
    except psycopg.Error as error:
        return {"status": "needs_revision", "reasons": [str(error)]}
