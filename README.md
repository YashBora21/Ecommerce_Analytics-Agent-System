# Olist Ecommerce Analytics Agent

A bounded LangGraph Query Agent for natural-language analysis over the complete relational Olist ecommerce dataset.

## Architecture

```text
JavaScript UI -> FastAPI -> guardrail -> Query Agent
                                      -> validated read-only SQL -> SQLite
                                      -> statistics tool
                                      -> Plotly chart tool
```

The LLM chooses structured actions (`query_sql`, `calculate_statistics`, `create_chart`, or `answer`). Python validates every action, SQL is executed on a read-only SQLite connection, and the graph limits SQL retries and total steps.

SQLite preserves all nine source datasets as relational tables: `orders`, `customers`, `order_items`, `payments`, `reviews`, `products`, `sellers`, `geolocation`, and `category_translation`. Prompt rules define grains, joins, metrics, missing values, and one-to-many fan-out protection.

## Setup

```powershell
python -m pip install -r requirements.txt
python backend\db\prepare_data.py
```

Create `.env` with `GROQ_API_KEY` and optionally `GROQ_MODEL`.

Start backend and frontend:

```powershell
npm run dev
```

- Frontend: `http://127.0.0.1:5500`
- API docs: `http://127.0.0.1:8000/docs`

## API and memory

- `GET /api/health` checks read-only database access.
- `POST /api/query` accepts `question` and optional `thread_id`; it returns the grounded answer, optional Plotly JSON, evidence, reusable thread ID, and token usage.

Reuse `thread_id` for follow-ups. Memory keeps recent turns, a rolling summary, and the last successful SQL.

## Safety and limits

- Only one `SELECT` or `WITH` statement is accepted.
- SQLite uses read-only/query-only mode and an operation allowlist.
- Queries time out after fifteen seconds and results are capped.
- Invalid SQL gets at most three attempts; new rounds are gated at five agent steps.
- Statistics and charts reject truncated results.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Deploy to Render

1. Import the local CSV files once with: python -m backend.db.prepare_neon
2. Push the repository to GitHub. The generated SQLite file and source CSV files are intentionally ignored.
3. In Render, create a Blueprint from the repository's render.yaml.
4. Enter GROQ_API_KEY and DATABASE_URL when Render requests the secret values.
5. Open the generated onrender.com URL. The same FastAPI service serves the frontend and API routes.

The Neon importer refuses to run when any target Olist table already exists, so it cannot overwrite an existing import.
