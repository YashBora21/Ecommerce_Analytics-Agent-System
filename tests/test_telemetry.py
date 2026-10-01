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
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch(
            "backend.agent.llm.Groq", return_value=client
        ):
            self.assertEqual(call_groq([{"role": "user", "content": "hello"}]), "ok")

        self.assertEqual(get_usage()["total_tokens"], 17)

    def test_json_generation_failure_retries_without_json_mode(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"answer"}'))],
        )
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                raise GroqError("Failed to generate JSON: failed_generation")
            return response

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        messages = [
            {"role": "system", "content": RESULT_ASSESSMENT_PROMPT},
            {"role": "user", "content": "Assess this"},
        ]

        reset_usage()
        with patch.dict(os.environ, {"GROQ_API_KEY": "test"}), patch(
            "backend.agent.llm.Groq", return_value=client
        ):
            result = call_groq(messages)

        self.assertEqual(result, '{"action":"answer"}')
        self.assertIn("response_format", requests[0])
        self.assertNotIn("response_format", requests[1])
        self.assertEqual(get_usage()["total_tokens"], 11)

    def test_node_wrapper_logs_start_and_end(self) -> None:
        node = logged_node("demo", lambda state: {"steps": 1, "error": ""})

        with self.assertLogs("ecommerce_agent", level="INFO") as logs:
            node({"steps": 0})

        output = "\n".join(logs.output)
        self.assertIn("node=demo event=start", output)
        self.assertIn("node=demo event=end", output)


if __name__ == "__main__":
    unittest.main()
