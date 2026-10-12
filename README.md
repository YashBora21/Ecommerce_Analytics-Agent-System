# Ecommerce Analytics Agent

A bounded LangGraph Query Agent for natural-language analysis over PostgreSQL.

## Flow

```text
question -> relevant live schema -> Query Agent -> validated SQL/tool
         -> Query Agent -> another tool or grounded answer
```

The same Query Agent chooses SQL, statistics, charts, clarification, or an
answer. Code—not prompts—enforces read-only SQL, discovered-table access,
artifact ownership, duplicate detection, join rules, planner limits, and
bounded retries, time, and token use.

Large or reusable PostgreSQL results are materialized under `agent_results`. LangGraph state keeps only bounded evidence and artifact metadata. Artifacts expire after 24 hours by default and can be queried by later steps without loading all rows into memory.

## Setup

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
npm run dev
```

Set `DATABASE_URL` to the PostgreSQL or Neon connection string. Set `CHECKPOINT_DATABASE_URL` and `JOB_DATABASE_URL` to enable durable conversation checkpoints and resumable queued runs; they may use the same database URL.

- App and API: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

## API

- `GET /api/health` checks database access.
- `POST /api/query` accepts `question` and optional `thread_id`.
- `GET /api/runs/{run_id}` polls a queued query.
- `DELETE /api/runs/{run_id}` requests cancellation.

A quick query returns immediately. A longer query returns HTTP 202 with a polling URL. Reuse `thread_id` for contextual follow-ups and artifact ownership.

## Safety

- SQLGlot parses exactly one PostgreSQL read-only query.
- DDL, DML, comments, system schemas, dangerous functions, cross joins, and joins without predicates are rejected.
- Only discovered source tables and unexpired artifacts owned by the current thread are accepted.
- PostgreSQL `EXPLAIN (FORMAT JSON)` routes genuinely expensive SQL back for revision; AST complexity is reported but does not reject valid SQL.
- Source reads use read-only transactions with statement and lock timeouts.
- Row, byte, retry, step, token, execution, chart, and artifact limits are configurable in `.env`.
- Incomplete or failed tool results are never presented as confirmed conclusions.

The PostgreSQL role should have `SELECT` on source schemas and write access only to `agent_results`; database grants remain the final security boundary.

## Tests

```powershell
python -m pytest -q
```

## Render

Create a Blueprint from `render.yaml`, then set `GROQ_API_KEY`, `DATABASE_URL`, `CHECKPOINT_DATABASE_URL`, and `JOB_DATABASE_URL`. The FastAPI service serves both the API and JavaScript frontend.
