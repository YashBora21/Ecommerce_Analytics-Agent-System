# Ecommerce Analytics Agent

Natural-language analytics over a 5,000-order sample derived from the Olist Brazilian ecommerce dataset.

## Milestone 1: CSV to SQLite

Build the database using only the Python standard library:

```powershell
python backend\db\prepare_data.py
```

This recreates `backend/db/ecommerce.db` atomically, validates the 21-column CSV contract and row count, verifies order totals, and creates the indexed `orders` table.

## Milestone 2: read-only SQL tool

`backend/tools/sql_core.py` validates and executes one `SELECT` or `WITH` query using a read-only SQLite connection. `backend/tools/sql.py` exposes that executor as the LangChain `run_sql` tool used by LangGraph.

Run its focused checks with:

```powershell
python -m unittest discover -s tests -v
```

## Milestone 3: minimal LangGraph query path

`backend/agent/agent.py` implements the first graph: question to Groq SQL planning, validated SQL execution, and a grounded answer. Schema and metric rules live in `backend/agent/prompt.py`.

Set `GROQ_API_KEY` and optionally `GROQ_MODEL` using `.env.example` as a reference. Environment files are not loaded automatically.

## Milestone 4: guardrails, retries, and thread memory

The LangGraph routes greetings and out-of-scope questions before SQL generation, retries rejected SQL at most three times, and tracks a five-step agent ceiling. When the caller reuses a `thread_id`, memory keeps the latest three conversation pairs, a rolling summary of older turns, and the last successful SQL for accurate follow-ups.







## Milestone 5: multi-query analysis

After each successful query, a structured assessment decides whether the collected SQL evidence is sufficient. If essential data is missing, the graph starts another SQL round while retaining prior results. Successful rounds reset the three-attempt SQL retry counter. The five-step ceiling gates new rounds; if another round cannot start, the response returns the evidence collected so far instead of discarding it.

## Milestone 6: statistics tool

`backend/tools/statistics.py` calculates mean, median, minimum, maximum, standard deviation, variance, percentile, and correlation from actual SQL result rows. The agent can route a completed query result through this tool and then assess the combined SQL and statistical evidence. Truncated SQL results are rejected rather than producing a misleading statistic.


## Milestone 7: Plotly charts

`backend/tools/chart.py` creates bar, line, scatter, pie, heatmap, or treemap charts from complete SQL results. The LLM chooses whether a chart is useful and supplies its type and columns; code validates that choice and returns Plotly JSON alongside the grounded text answer. Heatmaps use long-format x/y/value rows, treemaps use hierarchy/value rows, and a rejected chart gets one correction attempt before returning the collected text evidence with a clear note.


## Milestone 8: agent module split

The query agent is split by responsibility: `state.py` owns state and limits, `llm.py` owns Groq calls, `memory.py` owns conversation context, `decision.py` validates model-selected actions, `routing.py` owns graph routes, `nodes/` contains node factories, and `graph.py` only wires the workflow. `agent.py` remains a small compatibility entry point and `backend.agent` exports `ask()`. A `begin` node resets per-turn fields while checkpointed conversation memory remains available for follow-ups.


## Milestone 9: FastAPI

Run the backend with:

```powershell
uvicorn backend.app:app --reload
```

`GET /api/health` checks the API and read-only SQLite access. `POST /api/query` accepts `question` and an optional `thread_id`, then returns the grounded answer, Plotly chart JSON when present, the reusable thread ID, and supporting evidence.


## Request logging

Each graph node logs its start and end in the Uvicorn terminal, including the route, action, SQL, error, and cumulative Groq token usage. Every `/api/query` response includes `usage.input_tokens`, `usage.output_tokens`, and `usage.total_tokens`. Set `LOG_LEVEL` in `.env` to change verbosity.


## Frontend

Start both servers from one terminal:

```powershell
npm run dev
```

This uses Python's standard library launcher in `dev.py`; no npm packages are required. Press Ctrl+C to stop both servers.

Open `http://127.0.0.1:5500`. The frontend calls the API at `http://127.0.0.1:8000`, preserves its `thread_id` in session storage for follow-ups, renders Plotly responses, and displays per-request token usage.
