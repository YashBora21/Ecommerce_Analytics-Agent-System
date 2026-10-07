import json
import uuid
from typing import Annotated
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints

from backend.agent import ask
from backend.agent.state import QueryState
from backend.agent.telemetry import get_usage, logger, reset_usage
from backend.tools.sql_core import execute_sql


app = FastAPI(title="Ecommerce Analytics Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class QueryRequest(BaseModel):
    question: Question
    thread_id: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    chart: dict | None = None
    evidence: list[dict[str, str]] = Field(default_factory=list)
    usage: TokenUsage


def get_agent():
    return ask


@app.get("/api/health")
def health() -> dict[str, str]:
    result = execute_sql("SELECT 1 AS ok")
    if not result["ok"]:
        raise HTTPException(status_code=503, detail=result["error"])
    return {"status": "ok", "database": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query(request: QueryRequest, agent=Depends(get_agent)) -> QueryResponse:
    question = request.question
    thread_id = request.thread_id or str(uuid.uuid4())
    reset_usage()
    logger.info("event=request_start thread_id=%s question=%s", thread_id, question[:160])
    try:
        result: QueryState = agent(question, thread_id=thread_id)
        chart = json.loads(result["chart"]) if result.get("chart") else None
        usage = get_usage()
        logger.info("event=request_end thread_id=%s usage=%s", thread_id, usage)
        return QueryResponse(
            answer=result["final_answer"],
            thread_id=thread_id,
            chart=chart,
            evidence=result.get("evidence", []),
            usage=TokenUsage(**usage),
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Agent returned an invalid response") from error


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

