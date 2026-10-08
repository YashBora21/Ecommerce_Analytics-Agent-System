import json
import os
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from backend.agent import ask
from backend.db.postgres import close_pools
from backend.agent.state import QueryState
from backend.agent.telemetry import logger
from backend.jobs import run_manager
from backend.tools.sql_core import execute_sql


@asynccontextmanager
async def lifespan(_app: FastAPI):
    run_manager.recover(ask)
    yield
    close_pools()


app = FastAPI(title="Ecommerce Analytics Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SYNC_QUERY_SECONDS = float(os.getenv("SYNC_QUERY_SECONDS", "15"))


class QueryRequest(BaseModel):
    question: Question
    thread_id: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    status: str = "completed"
    answer: str
    thread_id: str
    run_id: str
    chart: dict | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    usage: TokenUsage


class QueryAccepted(BaseModel):
    status: str = "queued"
    thread_id: str
    run_id: str
    poll_url: str


def get_agent():
    return ask


def _completed_response(result: QueryState, thread_id: str, run_id: str, usage: dict) -> QueryResponse:
    chart = json.loads(result["chart"]) if result.get("chart") else None
    return QueryResponse(
        answer=result["final_answer"],
        thread_id=thread_id,
        run_id=run_id,
        chart=chart,
        evidence=result.get("evidence", []),
        artifacts=result.get("artifacts", []),
        usage=TokenUsage(**usage),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    result = execute_sql("SELECT 1 AS ok")
    if not result["ok"]:
        raise HTTPException(status_code=503, detail=result["error"])
    return {"status": "ok", "database": "ok"}


@app.post("/api/query", response_model=QueryResponse | QueryAccepted)
def query(request: QueryRequest, agent=Depends(get_agent)):
    thread_id = request.thread_id or str(uuid.uuid4())
    logger.info("event=request_start thread_id=%s question=%s", thread_id, request.question[:160])
    run_id, future = run_manager.submit(request.question, thread_id, agent)
    try:
        completed = future.result(timeout=SYNC_QUERY_SECONDS)
        if completed is None:
            raise RuntimeError("Query run could not be claimed")
        result, usage = completed
        logger.info("event=request_end thread_id=%s run_id=%s usage=%s", thread_id, run_id, usage)
        return _completed_response(result, thread_id, run_id, usage)
    except FutureTimeout:
        accepted = QueryAccepted(
            thread_id=thread_id,
            run_id=run_id,
            poll_url=f"/api/runs/{run_id}",
        )
        return JSONResponse(status_code=202, content=accepted.model_dump())
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Agent returned an invalid response") from error


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    record = run_manager.store.get(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run not found")
    if record["status"] == "completed":
        return _completed_response(
            record["result"],
            record["thread_id"],
            run_id,
            record["usage"],
        )
    return {
        "status": record["status"],
        "thread_id": record["thread_id"],
        "run_id": run_id,
        "error": record.get("error"),
        "cancel_requested": record.get("cancel_requested", False),
    }


@app.delete("/api/runs/{run_id}")
def cancel_run(run_id: str) -> dict[str, Any]:
    if not run_manager.store.get(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    accepted = run_manager.cancel(run_id)
    return {"run_id": run_id, "cancel_requested": accepted}


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
