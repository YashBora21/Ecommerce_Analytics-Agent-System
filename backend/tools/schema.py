import os
import re
import time
from typing import Any


from backend.db.postgres import postgres_connection


_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
CACHE_SECONDS = int(os.getenv("SCHEMA_CACHE_SECONDS", "300"))
MAX_SCHEMA_TABLES = int(os.getenv("MAX_SCHEMA_TABLES", "12"))


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if len(token) > 1
    }


def _postgres_catalog(database_url: str) -> dict[str, Any]:
    with postgres_connection(database_url) as connection:
        columns = connection.execute(
            """
            SELECT c.table_schema, c.table_name, c.column_name, c.data_type,
                   c.is_nullable = 'YES' AS nullable, c.ordinal_position
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema AND t.table_name = c.table_name
            WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema', 'agent_results')
              AND t.table_type IN ('BASE TABLE', 'VIEW')
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
            """
        ).fetchall()
        foreign_keys = connection.execute(
            """
            SELECT ns.nspname AS source_schema, src.relname AS source_table,
                   sa.attname AS source_column, tns.nspname AS target_schema,
                   target.relname AS target_table, ta.attname AS target_column
            FROM pg_constraint con
            JOIN pg_class src ON src.oid = con.conrelid
            JOIN pg_namespace ns ON ns.oid = src.relnamespace
            JOIN pg_class target ON target.oid = con.confrelid
            JOIN pg_namespace tns ON tns.oid = target.relnamespace
            JOIN LATERAL unnest(con.conkey) WITH ORDINALITY sk(attnum, ord) ON true
            JOIN LATERAL unnest(con.confkey) WITH ORDINALITY tk(attnum, ord)
              ON tk.ord = sk.ord
            JOIN pg_attribute sa ON sa.attrelid = src.oid AND sa.attnum = sk.attnum
            JOIN pg_attribute ta ON ta.attrelid = target.oid AND ta.attnum = tk.attnum
            WHERE con.contype = 'f'
              AND ns.nspname NOT IN ('pg_catalog', 'information_schema', 'agent_results')
            ORDER BY source_schema, source_table, source_column
            """
        ).fetchall()

    tables: dict[str, dict[str, Any]] = {}
    for row in columns:
        name = f"{row['table_schema']}.{row['table_name']}"
        table = tables.setdefault(
            name,
            {"schema": row["table_schema"], "table": row["table_name"], "columns": [], "foreign_keys": []},
        )
        table["columns"].append(
            {"name": row["column_name"], "type": row["data_type"], "nullable": row["nullable"]}
        )
    for row in foreign_keys:
        name = f"{row['source_schema']}.{row['source_table']}"
        if name in tables:
            tables[name]["foreign_keys"].append(
                {
                    "column": row["source_column"],
                    "target": f"{row['target_schema']}.{row['target_table']}.{row['target_column']}",
                }
            )
    return {"dialect": "postgres", "tables": tables}


def load_catalog(refresh: bool = False) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    cached = _CACHE.get(database_url)
    if cached and not refresh and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    catalog = _postgres_catalog(database_url)
    _CACHE[database_url] = (time.monotonic(), catalog)
    return catalog


def discover_schema(question: str, refresh: bool = False) -> dict[str, Any]:
    catalog = load_catalog(refresh)
    question_tokens = _tokens(question)
    scored = []
    for name, table in catalog["tables"].items():
        table_tokens = _tokens(name)
        column_tokens = set().union(*(_tokens(column["name"]) for column in table["columns"]))
        score = 4 * len(question_tokens & table_tokens) + len(question_tokens & column_tokens)
        scored.append((score, name, table))
    scored.sort(key=lambda item: (-item[0], item[1]))
    positive = [item for item in scored if item[0] > 0]
    selected = positive[:MAX_SCHEMA_TABLES] or scored[:MAX_SCHEMA_TABLES]

    adjacency = {name: set() for name in catalog["tables"]}
    for source_name, table in catalog["tables"].items():
        for foreign_key in table["foreign_keys"]:
            target = ".".join(foreign_key["target"].split(".")[:-1])
            if target in adjacency:
                adjacency[source_name].add(target)
                adjacency[target].add(source_name)

    selected_names = {name for _, name, _ in selected}
    frontier = set(selected_names)
    for _ in range(2):
        neighbors = set().union(*(adjacency[name] for name in frontier)) if frontier else set()
        frontier = neighbors - selected_names
        selected_names.update(frontier)
        if len(selected_names) >= MAX_SCHEMA_TABLES:
            break
    selected_tables = [
        catalog["tables"][name]
        for name in sorted(selected_names)[:MAX_SCHEMA_TABLES]
    ]

    lines = []
    allowed_tables = []
    for table in selected_tables:
        full_name = f"{table['schema']}.{table['table']}" if table["schema"] != "main" else table["table"]
        allowed_tables.extend({table["table"], full_name})
        columns = ", ".join(f"{column['name']} {column['type']}" for column in table["columns"])
        lines.append(f"- {full_name}: {columns}")
        for foreign_key in table["foreign_keys"]:
            lines.append(f"  FK {foreign_key['column']} -> {foreign_key['target']}")
    return {
        "dialect": catalog["dialect"],
        "tables": selected_tables,
        "allowed_tables": sorted(set(allowed_tables)),
        "context": "\n".join(lines),
    }
