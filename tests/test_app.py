import json
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import app, get_agent
from backend.jobs import RunStore


class ApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    @patch("backend.app.execute_sql", return_value={"ok": True})
    def test_health_checks_database(self, _execute_sql) -> None:
        response = TestClient(app).get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "database": "ok"})

    def test_job_store_updates_only_supplied_fields(self) -> None:
        store = RunStore()
        store.database_url = "postgres://test"
        connection = MagicMock()

        with patch("backend.jobs.postgres_connection") as connect:
            connect.return_value.__enter__.return_value = connection
            store.update("run-1", status="completed", usage={"total_tokens": 3})

        query, values = connection.execute.call_args.args
        self.assertIn("status = %s", query)
        self.assertIn("usage_json = %s", query)
        self.assertNotIn("error = %s", query)
        self.assertEqual(values[-1], "run-1")

    def test_serves_frontend(self) -> None:
        response = TestClient(app).get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("<title>Ecommerce Analytics</title>", response.text)
    def test_allows_frontend_origin(self) -> None:
        response = TestClient(app).options(
            "/api/query",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://127.0.0.1:5500",
        )

    def test_query_returns_answer_and_reusable_thread(self) -> None:
        calls = []

        def fake_agent(question, thread_id=None):
            calls.append((question, thread_id))
            return {
                "final_answer": "There are 5,000 orders.",
                "evidence": [{"kind": "sql", "sql": "SELECT COUNT(*)", "result": "n\n5000"}],
            }

        app.dependency_overrides[get_agent] = lambda: fake_agent
        client = TestClient(app)
        first = client.post("/api/query", json={"question": "How many orders?"})
        thread_id = first.json()["thread_id"]
        second = client.post(
            "/api/query",
            json={"question": "Break that down by state", "thread_id": thread_id},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(calls[0][1], thread_id)
        self.assertEqual(calls[1][1], thread_id)
        self.assertEqual(first.json()["evidence"][0]["result"], "n\n5000")
        self.assertEqual(first.json()["usage"]["total_tokens"], 0)

    def test_query_returns_plotly_chart_as_json(self) -> None:
        def fake_agent(question, thread_id=None):
            return {
                "final_answer": "Revenue by state is shown.",
                "chart": json.dumps({"data": [{"type": "bar"}], "layout": {}}),
                "evidence": [],
            }

        app.dependency_overrides[get_agent] = lambda: fake_agent
        response = TestClient(app).post("/api/query", json={"question": "Chart revenue"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chart"]["data"][0]["type"], "bar")

    def test_query_exposes_only_downloadable_artifact_metadata(self) -> None:
        def fake_agent(question, thread_id=None):
            return {
                "final_answer": "The complete result is available as CSV.",
                "evidence": [{
                    "kind": "sql",
                    "sql": "SELECT seller_id, revenue FROM sellers",
                    "result": "stored",
                    "artifact_id": "00000000-0000-0000-0000-000000000001",
                }],
                "artifacts": [{
                    "artifact_id": "00000000-0000-0000-0000-000000000001",
                    "relation": "agent_results.secret_name",
                    "row_count": 3000,
                    "columns": [{"name": "seller_id"}, {"name": "revenue"}],
                    "preview": [{"seller_id": "secret"}],
                    "profile": [{"column_name": "revenue"}],
                    "complete": True,
                    "expires_at": "2030-01-01T00:00:00+00:00",
                }],
            }

        app.dependency_overrides[get_agent] = lambda: fake_agent
        body = TestClient(app).post("/api/query", json={"question": "Every seller"}).json()
        artifact = body["artifacts"][0]

        self.assertEqual(artifact["columns"], ["seller_id", "revenue"])
        self.assertNotIn("artifact_id", artifact)
        self.assertNotIn("relation", artifact)
        self.assertNotIn("preview", artifact)
        self.assertIn("/api/artifacts/", artifact["download_url"])

    def test_query_does_not_repeat_an_old_artifact_download(self) -> None:
        def fake_agent(question, thread_id=None):
            return {
                "final_answer": "The chart is ready.",
                "evidence": [{"kind": "sql", "sql": "SELECT month", "result": "month\n2020-01"}],
                "artifacts": [{
                    "artifact_id": "00000000-0000-0000-0000-000000000001",
                    "row_count": 96477,
                    "columns": [{"name": "order_id"}],
                    "complete": True,
                    "expires_at": "2030-01-01T00:00:00+00:00",
                }],
            }

        app.dependency_overrides[get_agent] = lambda: fake_agent
        body = TestClient(app).post("/api/query", json={"question": "Show a chart"}).json()

        self.assertEqual(body["artifacts"], [])

    def test_artifact_download_checks_thread_ownership(self) -> None:
        with patch("backend.app.artifact_csv", return_value=iter(["a,b\n", "1,2\n"])) as export:
            response = TestClient(app).get(
                "/api/artifacts/00000000-0000-0000-0000-000000000001",
                params={"thread_id": "thread-1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "a,b\n1,2\n")
        export.assert_called_once_with(
            "00000000-0000-0000-0000-000000000001",
            "thread-1",
        )

    def test_incomplete_agent_result_is_not_reported_as_completed(self) -> None:
        def incomplete_agent(question, thread_id=None):
            return {
                "final_answer": "The provider limit interrupted this analysis.",
                "evidence": [],
                "incomplete": True,
            }

        app.dependency_overrides[get_agent] = lambda: incomplete_agent
        response = TestClient(app).post(
            "/api/query",
            json={"question": "Run a large analysis"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "incomplete")

    def test_slow_query_returns_job_and_can_be_polled(self) -> None:
        def slow_agent(question, thread_id=None):
            time.sleep(0.05)
            return {"final_answer": "Completed later.", "evidence": []}

        app.dependency_overrides[get_agent] = lambda: slow_agent
        client = TestClient(app)
        with patch("backend.app.SYNC_QUERY_SECONDS", 0.001):
            queued = client.post("/api/query", json={"question": "Run slowly"})

        self.assertEqual(queued.status_code, 202)
        poll_url = queued.json()["poll_url"]
        for _ in range(20):
            completed = client.get(poll_url)
            if completed.json()["status"] == "completed":
                break
            time.sleep(0.01)
        self.assertEqual(completed.json()["answer"], "Completed later.")

    def test_rejects_blank_question(self) -> None:
        def should_not_run(question, thread_id=None):
            self.fail("Agent should not run")

        app.dependency_overrides[get_agent] = lambda: should_not_run
        response = TestClient(app).post("/api/query", json={"question": "   "})

        self.assertEqual(response.status_code, 422)

    def test_agent_runtime_error_returns_service_unavailable(self) -> None:
        def unavailable(question, thread_id=None):
            raise RuntimeError("GROQ_API_KEY is not set")

        app.dependency_overrides[get_agent] = lambda: unavailable
        response = TestClient(app).post("/api/query", json={"question": "Count orders"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("GROQ_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
