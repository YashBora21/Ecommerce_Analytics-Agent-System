import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from groq import GroqError

from backend.agent.llm import call_groq
from backend.agent.prompt import RESULT_ASSESSMENT_PROMPT
from backend.agent.telemetry import add_usage, get_usage, logged_node, reset_usage


class TelemetryTests(unittest.TestCase):
    def test_usage_accumulates_for_the_request(self) -> None:
        reset_usage()
        add_usage(10, 4)
        add_usage(6, 2)

        self.assertEqual(
            get_usage(),
            {"input_tokens": 16, "output_tokens": 6, "total_tokens": 22},
        )

    def test_groq_usage_is_recorded(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )

        reset_usage()
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test", "GROQ_JSON_MODE": "1"},
        ), patch(
            "backend.agent.llm.Groq", return_value=client
        ):
            self.assertEqual(call_groq([{"role": "user", "content": "hello"}]), "ok")

        self.assertEqual(get_usage()["total_tokens"], 17)

    def test_assessment_uses_strict_pydantic_schema(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=(
                    '{"action":"answer","reason":"done",'
                    '"arguments":{"final_answer":"Complete."}}'
                )
            ))],
        )
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        messages = [
            {"role": "system", "content": RESULT_ASSESSMENT_PROMPT},
            {"role": "user", "content": "Assess this"},
        ]

        reset_usage()
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test", "GROQ_MODEL": "openai/gpt-oss-120b"},
        ), patch(
            "backend.agent.llm.Groq", return_value=client
        ):
            result = call_groq(messages)

        self.assertIn('"action":"answer"', result)
        response_format = requests[0]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("anyOf", schema["properties"]["decision"])
        self.assertEqual(get_usage()["total_tokens"], 11)

    def test_native_tool_call_failure_is_normalized_without_retry(self) -> None:
        requests = []
        error = GroqError("Tool choice is none, but model called a tool")
        error.body = {
            "error": {
                "failed_generation": (
                    '{"name":"query_sql","arguments":'
                    '{"sql":"SELECT review_score FROM reviews"}}'
                )
            }
        }

        def create(**kwargs):
            requests.append(kwargs)
            raise error

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        messages = [
            {"role": "system", "content": RESULT_ASSESSMENT_PROMPT},
            {"role": "user", "content": "Assess this"},
        ]

        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test", "GROQ_JSON_MODE": "0"},
        ), patch("backend.agent.llm.Groq", return_value=client):
            result = call_groq(messages)

        self.assertEqual(len(requests), 1)
        self.assertEqual(
            result,
            '{"action": "query_sql", "reason": "", '
            '"arguments": {"sql": "SELECT review_score FROM reviews"}}',
        )

    def test_rate_limit_is_retried_once(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )
        error = GroqError("Rate limit reached")
        error.status_code = 429
        error.response = SimpleNamespace(headers={"retry-after": "0"})
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise error
            return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.dict(
            os.environ,
            {"GROQ_API_KEY": "test", "GROQ_JSON_MODE": "0"},
        ), patch("backend.agent.llm.Groq", return_value=client), patch(
            "backend.agent.llm.time.sleep"
        ):
            result = call_groq([{"role": "user", "content": "hello"}])

        self.assertEqual(result, "ok")
        self.assertEqual(len(requests), 2)

    def test_node_wrapper_logs_start_and_end(self) -> None:
        node = logged_node(
            "demo", lambda state: {"steps": 1, "sql": "SELECT secret", "error": ""}
        )

        with self.assertLogs("ecommerce_agent", level="INFO") as logs:
            node({"steps": 0})

        output = "\n".join(logs.output)
        self.assertIn("node=demo event=start", output)
        self.assertIn("node=demo event=end", output)
        self.assertIn("*" * 70, output)
        self.assertNotIn("sql=", output)

    def test_sql_nodes_log_the_query(self) -> None:
        node = logged_node(
            "sql", lambda state: {"sql": "SELECT 1\nFROM orders", "error": ""}
        )

        with self.assertLogs("ecommerce_agent", level="INFO") as logs:
            node({"steps": 1})

        output = "\n".join(logs.output)
        self.assertIn("sql=SELECT 1 FROM orders", output)
        self.assertNotIn("SELECT 1\nFROM orders", output)


if __name__ == "__main__":
    unittest.main()
