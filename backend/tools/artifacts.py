import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg import sql

from backend.db.postgres import postgres_connection
from backend.tools.sql_analysis import validate_sql


ARTIFACT_SCHEMA = "agent_results"
ARTIFACT_TTL_HOURS = int(os.getenv("ARTIFACT_TTL_HOURS", "24"))
ARTIFACT_TIMEOUT = float(os.getenv("ARTIFACT_TIMEOUT", "120"))
MATERIALIZE_ROWS = int(os.getenv("MATERIALIZE_ROWS", "1000"))
MAX_ARTIFACT_ROWS = int(os.getenv("MAX_ARTIFACT_ROWS", "5000000"))
MAX_ARTIFACT_BYTES = int(os.getenv("MAX_ARTIFACT_BYTES", str(1024 * 1024 * 1024)))


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for durable result artifacts")
    return database_url


def setup_artifacts(connection: psycopg.Connection) -> None:
    connection.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(ARTIFACT_SCHEMA)))
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARTIFACT_SCHEMA}.artifacts (
            artifact_id uuid PRIMARY KEY,
            owner_id text NOT NULL,
            relation_name text NOT NULL UNIQUE,
            source_sql text NOT NULL,
            sql_hash text NOT NULL,
            row_count bigint NOT NULL,
            byte_size bigint NOT NULL,
            columns_json jsonb NOT NULL,
            preview_json jsonb NOT NULL,
            profile_json jsonb NOT NULL,
            complete boolean NOT NULL DEFAULT true,
            created_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL
        )
        """
    )
    connection.execute(
        f"CREATE INDEX IF NOT EXISTS artifacts_owner_expiry_idx "
        f"ON {ARTIFACT_SCHEMA}.artifacts(owner_id, expires_at)"
    )


def cleanup_artifacts(connection: psycopg.Connection) -> int:
    setup_artifacts(connection)
    expired = connection.execute(
        f"SELECT artifact_id, relation_name FROM {ARTIFACT_SCHEMA}.artifacts "
        "WHERE expires_at <= now() FOR UPDATE SKIP LOCKED"
    ).fetchall()
    for _, relation_name in expired:
        connection.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(ARTIFACT_SCHEMA), sql.Identifier(relation_name)
            )
        )
    if expired:
        connection.execute(
            f"DELETE FROM {ARTIFACT_SCHEMA}.artifacts WHERE artifact_id = ANY(%s)",
            ([row[0] for row in expired],),
        )
    return len(expired)


def owned_artifacts(owner_id: str) -> list[dict[str, Any]]:
    if not owner_id or not os.getenv("DATABASE_URL"):
        return []
    with postgres_connection(_database_url()) as connection:
        setup_artifacts(connection)
        rows = connection.execute(
            f"""
            SELECT artifact_id, relation_name, row_count, byte_size, columns_json,
                   preview_json, profile_json, complete, expires_at
            FROM {ARTIFACT_SCHEMA}.artifacts
            WHERE owner_id = %s AND complete AND expires_at > now()
            ORDER BY created_at
            """,
            (owner_id,),
        ).fetchall()
    return [
        {
            **dict(row),
            "artifact_id": str(row["artifact_id"]),
            "relation": f"{ARTIFACT_SCHEMA}.{row['relation_name']}",
            "columns": row["columns_json"],
            "preview": row["preview_json"],
            "profile": row["profile_json"],
            "expires_at": row["expires_at"].isoformat(),
        }
        for row in rows
    ]


def materialize_result(
    query: str,
    owner_id: str,
    allowed_tables: set[str],
    allowed_artifacts: set[str],
) -> dict[str, Any]:
    analysis = validate_sql(query, allowed_tables, allowed_artifacts)
    artifact_id = uuid.uuid4()
    relation_name = f"result_{artifact_id.hex}"
    database_url = _database_url()
    with postgres_connection(database_url) as connection:
        setup_artifacts(connection)
        cleanup_artifacts(connection)
        connection.execute("SELECT set_config('statement_timeout', %s, true)", (str(int(ARTIFACT_TIMEOUT * 1000)),))
        connection.execute("SELECT set_config('lock_timeout', %s, true)", ("2000",))
        relation = sql.SQL("{}.{}").format(sql.Identifier(ARTIFACT_SCHEMA), sql.Identifier(relation_name))
        try:
            connection.execute(sql.SQL("CREATE TABLE {} AS ").format(relation) + sql.SQL(query))
            row_count = connection.execute(sql.SQL("SELECT COUNT(*) AS n FROM {}").format(relation)).fetchone()["n"]
            byte_size = connection.execute(
                "SELECT pg_total_relation_size(%s)",
                (f"{ARTIFACT_SCHEMA}.{relation_name}",),
            ).fetchone()["pg_total_relation_size"]
            if row_count > MAX_ARTIFACT_ROWS or byte_size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"Artifact exceeds storage limits: rows={row_count}, bytes={byte_size}"
                )
            preview = [dict(row) for row in connection.execute(sql.SQL("SELECT * FROM {} LIMIT 5").format(relation)).fetchall()]
            cursor = connection.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(relation))
            columns = [
                {"name": description.name, "type_oid": description.type_code}
                for description in cursor.description or []
            ]
            connection.execute(sql.SQL("ANALYZE {}").format(relation))
            profile_rows = connection.execute(
                """
                SELECT attname AS column_name, null_frac, n_distinct,
                       most_common_vals::text AS most_common_values
                FROM pg_stats
                WHERE schemaname = %s AND tablename = %s
                ORDER BY attname
                """,
                (ARTIFACT_SCHEMA, relation_name),
            ).fetchall()
            profile = [dict(row) for row in profile_rows]
            expires_at = datetime.now(timezone.utc).timestamp() + ARTIFACT_TTL_HOURS * 3600
            connection.execute(
                f"""
                INSERT INTO {ARTIFACT_SCHEMA}.artifacts
                    (artifact_id, owner_id, relation_name, source_sql, sql_hash,
                     row_count, byte_size, columns_json, preview_json, profile_json, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb,
                        to_timestamp(%s))
                """,
                (
                    artifact_id,
                    owner_id,
                    relation_name,
                    query,
                    analysis["sql_hash"],
                    row_count,
                    byte_size,
                    json.dumps(columns, default=str),
                    json.dumps(preview, default=str),
                    json.dumps(profile, default=str),
                    expires_at,
                ),
            )
        except Exception:
            connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(relation))
            raise
    return {
        "artifact_id": str(artifact_id),
        "relation": f"{ARTIFACT_SCHEMA}.{relation_name}",
        "row_count": row_count,
        "byte_size": byte_size,
        "columns": columns,
        "preview": preview,
        "profile": profile,
        "complete": True,
        "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
        "sql_hash": analysis["sql_hash"],
    }
