import hashlib
import os
import re
from typing import Any

import psycopg
from psycopg.rows import tuple_row
from sqlglot import exp, parse
from sqlglot.errors import ParseError

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


def _one_row_ctes(tree: exp.Expression) -> set[str]:
    names = set()
    for cte in tree.find_all(exp.CTE):
        query = cte.this
        if not isinstance(query, exp.Select):
            continue
        limit = query.args.get("limit")
        limit_value = getattr(getattr(limit, "expression", None), "this", None)
        has_small_limit = str(limit_value).isdigit() and int(limit_value) <= 1
        scalar_aggregate = (
            query.find(exp.AggFunc) is not None
            and query.args.get("group") is None
            and query.find(exp.Window) is None
        )
        if query.args.get("from_") is None or has_small_limit or scalar_aggregate:
            names.add(cte.alias_or_name.lower())
    return names


def _has_unstable_limit_one(tree: exp.Expression) -> bool:
    for select in tree.find_all(exp.Select):
        limit = select.args.get("limit")
        limit_value = getattr(getattr(limit, "expression", None), "this", None)
        if str(limit_value) != "1":
            continue
        order = select.args.get("order")
        if not order or len(order.expressions) < 2:
            return True
    return False


def _repair_invalid_alias(sql: str, error: psycopg.Error) -> str | None:
    message = getattr(getattr(error, "diag", None), "message_primary", "") or str(error)
    match = re.search(r'syntax error at or near "([^"]+)"', message, re.IGNORECASE)
    if not match:
        return None
    invalid_alias = match.group(1).lower()
    statements = [statement for statement in parse(sql, read="postgres") if statement]
    if len(statements) != 1:
        return None
    tree = statements[0]
    repaired = False
    for alias in tree.find_all(exp.TableAlias):
        if alias.name.lower() == invalid_alias:
            alias.set("this", exp.to_identifier(alias.name, quoted=True))
            repaired = True
    if not repaired:
        return None
    for column in tree.find_all(exp.Column):
        if column.table.lower() == invalid_alias:
            column.set("table", exp.to_identifier(column.table, quoted=True))
    return tree.sql(dialect="postgres", pretty=False)


def validate_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a non-empty string")
    if "\x00" in sql:
        raise ValueError("SQL cannot contain null bytes")
    try:
        statements = [statement for statement in parse(sql, read="postgres") if statement]
    except ParseError as error:
        raise ValueError(f"Invalid SQL: {error}") from error
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Only SELECT queries are allowed")
    blocked_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Merge,
        exp.Create,
        exp.Drop,
        exp.Command,
    )
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
    one_row_ctes = _one_row_ctes(tree)
    risky_cross_join = False
    for join in joins:
        is_cross = str(join.args.get("kind", "")).upper() == "CROSS"
        joined_name = join.this.name.lower() if isinstance(join.this, exp.Table) else ""
        joins_one_row_cte = joined_name in one_row_ctes
        if is_cross:
            risky_cross_join = risky_cross_join or not joins_one_row_cte
            continue
        if not join.args.get("on") and not join.args.get("using"):
            if joins_one_row_cte:
                continue
            raise ValueError("Every JOIN requires an ON or USING condition")
        on_clause = join.args.get("on")
        if on_clause is not None and not any(on_clause.find_all(exp.Column)):
            raise ValueError("JOIN conditions must compare columns")

    for function in tree.find_all(exp.Func):
        name = function.name.lower() if function.name else function.sql_name().lower()
        if name in BLOCKED_FUNCTIONS:
            raise ValueError(f"Function is not allowed: {name}")

    canonical = tree.sql(dialect="postgres", pretty=False, normalize=True)
    has_limit = tree.args.get("limit") is not None
    return {
        "canonical_sql": canonical,
        "sql_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "tables": sorted(set(tables)),
        "has_limit": has_limit,
        "risky_cross_join": risky_cross_join,
        "unstable_limit_one": _has_unstable_limit_one(tree),
    }


def explain_postgres(sql: str, database_url: str, timeout: float = 5.0) -> dict[str, Any]:
    with postgres_connection(database_url) as connection:
        with connection.cursor(row_factory=tuple_row) as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (str(int(timeout * 1000)),),
            )
            cursor.execute(f"EXPLAIN (FORMAT JSON) {sql}")
            root = cursor.fetchone()[0][0]["Plan"]
    plan_rows = int(root.get("Plan Rows", 0))
    plan_width = int(root.get("Plan Width", 0))
    return {
        "total_cost": float(root.get("Total Cost", 0)),
        "plan_rows": plan_rows,
        "estimated_bytes": plan_rows * plan_width,
    }


def analyze_sql(
    sql: str,
    allowed_tables: set[str] | None = None,
    allowed_artifacts: set[str] | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    try:
        analysis = validate_sql(sql, allowed_tables, allowed_artifacts)
        if analysis["unstable_limit_one"]:
            return {
                "status": "needs_revision",
                "reasons": [
                    "LIMIT 1 has no deterministic tie handling. Return all ties, "
                    "or order by the metric, supporting sample count, and a stable identifier."
                ],
                "planner": {},
                "executable_sql": sql,
                **analysis,
            }
        executable_sql = sql
        try:
            planner = explain_postgres(executable_sql, database_url) if database_url else {}
        except psycopg.Error as error:
            repaired_sql = _repair_invalid_alias(executable_sql, error)
            if not repaired_sql:
                raise
            analysis = validate_sql(repaired_sql, allowed_tables, allowed_artifacts)
            planner = explain_postgres(repaired_sql, database_url)
            executable_sql = repaired_sql
        reasons = []
        if analysis["risky_cross_join"]:
            reasons.append("cross join requires decomposition or an explicit join key")
        if planner.get("total_cost", 0) > DEFAULT_MAX_PLAN_COST:
            reasons.append(
                f"estimated cost {planner['total_cost']:.0f} "
                f"exceeds {DEFAULT_MAX_PLAN_COST:.0f}"
            )
        if planner.get("plan_rows", 0) > DEFAULT_MAX_PLAN_ROWS:
            reasons.append(
                f"estimated rows {planner['plan_rows']} exceed {DEFAULT_MAX_PLAN_ROWS}"
            )
        if planner.get("estimated_bytes", 0) > DEFAULT_MAX_PLAN_BYTES:
            reasons.append(
                f"estimated bytes {planner['estimated_bytes']} exceed {DEFAULT_MAX_PLAN_BYTES}"
            )
        return {
            "status": "needs_decomposition" if reasons else "safe",
            "reasons": reasons,
            "planner": planner,
            "executable_sql": executable_sql,
            **analysis,
        }
    except ValueError as error:
        message = str(error)
        unsafe_markers = (
            "Only SELECT",
            "not allowed",
            "not owned",
            "requires an ON",
            "Row-locking",
        )
        status = "unsafe" if any(marker in message for marker in unsafe_markers) else "needs_revision"
        return {"status": status, "reasons": [message]}
    except psycopg.Error as error:
        return {"status": "needs_revision", "reasons": [str(error)]}
