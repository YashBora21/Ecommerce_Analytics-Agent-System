import csv
import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg import sql

from backend.db import postgres
from backend.tools.sql_analysis import validate_sql


ARTIFACT_SCHEMA = "agent_results"
ARTIFACT_TTL_HOURS = int(os.getenv("ARTIFACT_TTL_HOURS", "24"))
ARTIFACT_TIMEOUT = float(os.getenv("ARTIFACT_TIMEOUT", "120"))
MAX_ARTIFACT_ROWS = int(os.getenv("MAX_ARTIFACT_ROWS", "5000000"))
MAX_ARTIFACT_BYTES = int(os.getenv("MAX_ARTIFACT_BYTES", str(1024 * 1024 * 1024)))


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for durable result artifacts")
    return database_url


def setup_artifacts(connection: psycopg.Connection) -> None:
    connection.execute(
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
            sql.Identifier(ARTIFACT_SCHEMA)
        )
    )
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ARTIFACT_SCHEMA}.artifacts (
            artifact_id uuid PRIMARY KEY,
            owner_id text NOT NULL,
            relation_name text NOT NULL UNIQUE,
            row_count bigint NOT NULL,
            byte_size bigint NOT NULL,
            columns_json jsonb NOT NULL,
            has_limit boolean NOT NULL DEFAULT false,
            created_at timestamptz NOT NULL DEFAULT now(),
            expires_at timestamptz NOT NULL
        )
        """
    )
    connection.execute(
        f"""
        ALTER TABLE {ARTIFACT_SCHEMA}.artifacts
            ADD COLUMN IF NOT EXISTS has_limit boolean NOT NULL DEFAULT false,
            DROP COLUMN IF EXISTS source_sql,
            DROP COLUMN IF EXISTS profile_json,
            DROP COLUMN IF EXISTS sql_hash,
            DROP COLUMN IF EXISTS preview_json,
            DROP COLUMN IF EXISTS complete
        """
    )
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS artifacts_owner_expiry_idx
        ON {ARTIFACT_SCHEMA}.artifacts(owner_id, expires_at)
        """
    )


def _artifact_metadata(row: dict[str, Any]) -> dict[str, Any]:
    artifact = dict(row)
    artifact["artifact_id"] = str(row["artifact_id"])
    artifact["relation"] = f"{ARTIFACT_SCHEMA}.{row['relation_name']}"
    artifact["columns"] = row["columns_json"]
    artifact["has_limit"] = bool(row["has_limit"])
    artifact["expires_at"] = row["expires_at"].isoformat()
    return artifact


def cleanup_artifacts(connection: psycopg.Connection) -> int:
    setup_artifacts(connection)
    expired = connection.execute(
        f"SELECT artifact_id, relation_name FROM {ARTIFACT_SCHEMA}.artifacts "
        "WHERE expires_at <= now() FOR UPDATE SKIP LOCKED"
    ).fetchall()
    for row in expired:
        connection.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                sql.Identifier(ARTIFACT_SCHEMA), sql.Identifier(row["relation_name"])
            )
        )
    if expired:
        connection.execute(
            f"DELETE FROM {ARTIFACT_SCHEMA}.artifacts WHERE artifact_id = ANY(%s)",
            ([row["artifact_id"] for row in expired],),
        )
    return len(expired)


def owned_artifacts(owner_id: str) -> list[dict[str, Any]]:
    if not owner_id or not os.getenv("DATABASE_URL"):
        return []
    with postgres.postgres_connection(_database_url()) as connection:
        setup_artifacts(connection)
        rows = connection.execute(
            f"""
            SELECT artifact_id, relation_name, row_count, byte_size, columns_json,
                   has_limit, expires_at
            FROM {ARTIFACT_SCHEMA}.artifacts
            WHERE owner_id = %s AND expires_at > now()
            ORDER BY created_at
            """,
            (owner_id,),
        ).fetchall()
    return [_artifact_metadata(row) for row in rows]


def owned_artifact(
    artifact_id: str,
    owner_id: str,
    database_url: str | None = None,
) -> dict[str, Any]:
    if not artifact_id or not owner_id:
        raise ValueError("Artifact ID and owner are required")
    try:
        artifact_key = uuid.UUID(artifact_id)
    except ValueError as error:
        raise ValueError("Artifact is unavailable, expired, or owned by another thread") from error
    with postgres.postgres_connection(database_url or _database_url()) as connection:
        row = connection.execute(
            f"""
            SELECT artifact_id, relation_name, row_count, byte_size, columns_json,
                   has_limit, expires_at
            FROM {ARTIFACT_SCHEMA}.artifacts
            WHERE artifact_id = %s AND owner_id = %s
              AND expires_at > now()
            """,
            (artifact_key, owner_id),
        ).fetchone()
    if not row:
        raise ValueError("Artifact is unavailable, expired, or owned by another thread")
    return _artifact_metadata(row)


def artifact_csv(
    artifact_id: str,
    owner_id: str,
    database_url: str | None = None,
):
    artifact = owned_artifact(artifact_id, owner_id, database_url)
    columns = [column["name"] for column in artifact["columns"]]
    relation = sql.SQL("{}.{}").format(
        sql.Identifier(ARTIFACT_SCHEMA),
        sql.Identifier(artifact["relation_name"]),
    )

    def rows():
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(columns)
        yield output.getvalue()
        with postgres.postgres_connection(database_url or _database_url()) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            with connection.cursor(name=f"artifact_export_{uuid.uuid4().hex}") as cursor:
                cursor.execute(sql.SQL("SELECT * FROM {}").format(relation))
                while batch := cursor.fetchmany(1000):
                    output.seek(0)
                    output.truncate(0)
                    writer.writerows([row[column] for column in columns] for row in batch)
                    yield output.getvalue()

    return rows()


def materialize_result(
    query: str,
    owner_id: str,
    allowed_tables: set[str],
    allowed_artifacts: set[str],
) -> dict[str, Any]:
    analysis = validate_sql(query, allowed_tables, allowed_artifacts)
    has_limit = bool(analysis.get("has_limit", False))
    artifact_id = uuid.uuid4()
    relation_name = f"result_{artifact_id.hex}"
    database_url = _database_url()
    with postgres.postgres_connection(database_url) as connection:
        cleanup_artifacts(connection)
        connection.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(int(ARTIFACT_TIMEOUT * 1000)),),
        )
        connection.execute("SELECT set_config('lock_timeout', %s, true)", ("2000",))
        relation = sql.SQL("{}.{}").format(
            sql.Identifier(ARTIFACT_SCHEMA),
            sql.Identifier(relation_name),
        )
        try:
            connection.execute(sql.SQL("CREATE TABLE {} AS ").format(relation) + sql.SQL(query))
            row_count = connection.execute(
                sql.SQL("SELECT COUNT(*) AS n FROM {}").format(relation)
            ).fetchone()["n"]
            byte_size = connection.execute(
                "SELECT pg_total_relation_size(%s)",
                (f"{ARTIFACT_SCHEMA}.{relation_name}",),
            ).fetchone()["pg_total_relation_size"]
            if row_count > MAX_ARTIFACT_ROWS or byte_size > MAX_ARTIFACT_BYTES:
                raise ValueError(
                    f"Artifact exceeds storage limits: rows={row_count}, bytes={byte_size}"
                )
            preview = [
                dict(row)
                for row in connection.execute(
                    sql.SQL("SELECT * FROM {} LIMIT 5").format(relation)
                ).fetchall()
            ]
            cursor = connection.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(relation))
            columns = [
                {"name": description.name, "type_oid": description.type_code}
                for description in cursor.description or []
            ]
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=ARTIFACT_TTL_HOURS
            )
            connection.execute(
                f"""
                INSERT INTO {ARTIFACT_SCHEMA}.artifacts
                    (artifact_id, owner_id, relation_name, row_count, byte_size,
                     columns_json, has_limit, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                """,
                (
                    artifact_id,
                    owner_id,
                    relation_name,
                    row_count,
                    byte_size,
                    json.dumps(columns, default=str),
                    has_limit,
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
        "has_limit": has_limit,
        "expires_at": expires_at.isoformat(),
        "sql_hash": analysis["sql_hash"],
    }
