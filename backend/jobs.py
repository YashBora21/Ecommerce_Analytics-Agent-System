import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any


from backend.db.postgres import postgres_connection
from backend.agent.telemetry import (
    get_usage,
    reset_cancel_checker,
    reset_usage,
    set_cancel_checker,
    logger,
)


class RunStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.memory: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        if database_url:
            self._setup()

    def _setup(self) -> None:
        with postgres_connection(self.database_url) as connection:
            connection.execute("CREATE SCHEMA IF NOT EXISTS agent_results")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_results.query_runs (
                    run_id uuid PRIMARY KEY,
                    thread_id text NOT NULL,
                    question text NOT NULL,
                    status text NOT NULL,
                    result_json jsonb,
                    usage_json jsonb,
                    error text,
                    cancel_requested boolean NOT NULL DEFAULT false,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS query_runs_thread_idx "
                "ON agent_results.query_runs(thread_id, created_at DESC)"
            )

    def create(self, question: str, thread_id: str) -> str:
        run_id = str(uuid.uuid4())
        record = {
            "run_id": run_id,
            "thread_id": thread_id,
            "question": question,
            "status": "queued",
            "result": None,
            "usage": None,
            "error": None,
            "cancel_requested": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self.database_url:
            with postgres_connection(self.database_url) as connection:
                connection.execute(
                    "INSERT INTO agent_results.query_runs "
                    "(run_id, thread_id, question, status) VALUES (%s, %s, %s, 'queued')",
                    (run_id, thread_id, question),
                )
        else:
            with self.lock:
                self.memory[run_id] = record
        return run_id

    def update(self, run_id: str, **changes) -> None:
        if self.database_url:
            columns = {
                "status": changes.get("status"),
                "result_json": json.dumps(changes["result"], default=str) if "result" in changes else None,
                "usage_json": json.dumps(changes["usage"], default=str) if "usage" in changes else None,
                "error": changes.get("error"),
                "cancel_requested": changes.get("cancel_requested"),
            }
            assignments = []
            values = []
            for column, value in columns.items():
                if column.replace("_json", "") in changes or column in changes:
                    assignments.append(f"{column} = %s")
                    values.append(value)
            if not assignments:
                return
            values.append(run_id)
            with postgres_connection(self.database_url) as connection:
                connection.execute(
                    f"UPDATE agent_results.query_runs SET {', '.join(assignments)}, "
                    "updated_at = now() WHERE run_id = %s",
                    values,
                )
        else:
            with self.lock:
                self.memory[run_id].update(changes)

    def get(self, run_id: str) -> dict[str, Any] | None:
        if not self.database_url:
            with self.lock:
                record = self.memory.get(run_id)
                return dict(record) if record else None
        with postgres_connection(self.database_url) as connection:
            row = connection.execute(
                "SELECT * FROM agent_results.query_runs WHERE run_id = %s", (run_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "run_id": str(row["run_id"]),
            "thread_id": row["thread_id"],
            "question": row["question"],
            "status": row["status"],
            "result": row["result_json"],
            "usage": row["usage_json"],
            "error": row["error"],
            "cancel_requested": row["cancel_requested"],
            "created_at": row["created_at"].isoformat(),
        }

    def claim(self, run_id: str) -> bool:
        if not self.database_url:
            with self.lock:
                record = self.memory[run_id]
                if record["status"] != "queued":
                    return False
                record["status"] = "running"
                return True
        with postgres_connection(self.database_url) as connection:
            row = connection.execute(
                "UPDATE agent_results.query_runs SET status = 'running', updated_at = now() "
                "WHERE run_id = %s AND status = 'queued' RETURNING run_id",
                (run_id,),
            ).fetchone()
        return bool(row)

    def pending(self) -> list[dict[str, Any]]:
        if not self.database_url:
            return []
        with postgres_connection(self.database_url) as connection:
            connection.execute(
                "UPDATE agent_results.query_runs SET status = 'queued', updated_at = now() "
                "WHERE status = 'running' AND updated_at < now() - interval '5 minutes'"
            )
            rows = connection.execute(
                "SELECT run_id, thread_id, question FROM agent_results.query_runs "
                "WHERE status = 'queued' ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]


class RunManager:
    def __init__(self, store: RunStore | None = None):
        self.store = store or RunStore(os.getenv("JOB_DATABASE_URL"))
        self.executor = ThreadPoolExecutor(max_workers=int(os.getenv("JOB_WORKERS", "4")))
        self.futures: dict[str, Future] = {}
        self.lock = threading.Lock()

    def submit(self, question: str, thread_id: str, agent) -> tuple[str, Future]:
        run_id = self.store.create(question, thread_id)
        return run_id, self._submit_existing(run_id, question, thread_id, agent)

    def _submit_existing(self, run_id: str, question: str, thread_id: str, agent) -> Future:
        future = self.executor.submit(self._run, run_id, question, thread_id, agent)
        with self.lock:
            self.futures[run_id] = future
        return future

    def _run(self, run_id: str, question: str, thread_id: str, agent):
        if not self.store.claim(run_id):
            return None
        reset_usage()
        set_cancel_checker(
            lambda: bool((self.store.get(run_id) or {}).get("cancel_requested"))
        )
        try:
            result = agent(question, thread_id=thread_id)
            usage = get_usage()
            self.store.update(run_id, status="completed", result=result, usage=usage)
            logger.info(
                "event=run_completed run_id=%s thread_id=%s usage=%s",
                run_id,
                thread_id,
                usage,
            )
            return result, usage
        except Exception as error:
            cancelled = bool((self.store.get(run_id) or {}).get("cancel_requested"))
            status = "cancelled" if cancelled else "failed"
            self.store.update(run_id, status=status, error=str(error))
            logger.error(
                "event=run_%s run_id=%s thread_id=%s error=%s",
                status,
                run_id,
                thread_id,
                error,
            )
            raise
        finally:
            reset_cancel_checker()

    def recover(self, agent) -> None:
        for record in self.store.pending():
            run_id = str(record["run_id"])
            with self.lock:
                if run_id in self.futures:
                    continue
            self._submit_existing(run_id, record["question"], record["thread_id"], agent)

    def cancel(self, run_id: str) -> bool:
        record = self.store.get(run_id)
        if not record or record["status"] in {"completed", "failed", "cancelled"}:
            return False
        self.store.update(run_id, cancel_requested=True)
        with self.lock:
            future = self.futures.get(run_id)
        if future and future.cancel():
            self.store.update(run_id, status="cancelled")
        return True


run_manager = RunManager()
