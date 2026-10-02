import json
import unittest

from fastapi.testclient import TestClient

from backend.app import app, get_agent


class ApiTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_checks_database(self) -> None:
        response = TestClient(app).get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "database": "ok"})

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

