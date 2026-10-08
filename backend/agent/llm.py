import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError, Groq, RateLimitError

from backend.agent.decision import decision_json_schema
from backend.agent.prompt import RESULT_ASSESSMENT_PROMPT
from backend.agent.state import Message
from backend.agent.telemetry import add_usage, logger

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "openai/gpt-oss-120b"


def _recover(error: BadRequestError) -> str | None:
    """Groq's 400 still carries the model's text in failed_generation. Use it."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    raw = details.get("failed_generation") if isinstance(details, dict) else None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if "action" in data:
        return json.dumps(data)
    if isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
        return json.dumps({"action": data["name"], "reason": "", "arguments": data["arguments"]})
    return None


def call_groq(messages: list[Message]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    options = {}
    if messages[0]["content"] == RESULT_ASSESSMENT_PROMPT:
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_agent_decision",
                "strict": True,
                "schema": decision_json_schema(),
            },
        }

    client = Groq(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, **options
            )
            break
        except BadRequestError as error:
            recovered = _recover(error)
            if not recovered:
                raise
            logger.warning("event=llm_tool_call_recovered model=%s", MODEL)
            return recovered
        except RateLimitError as error:
            if attempt == 2:
                raise
            try:
                delay = min(float(error.response.headers.get("retry-after", 3)), 10)
            except (TypeError, ValueError):
                delay = 3
            logger.warning("event=llm_rate_limit_retry delay=%s", delay)
            time.sleep(delay)

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    add_usage(input_tokens, output_tokens)
    logger.info("event=llm model=%s input_tokens=%s output_tokens=%s", MODEL, input_tokens, output_tokens)
    return response.choices[0].message.content or ""