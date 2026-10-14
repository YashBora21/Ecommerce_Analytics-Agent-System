import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import BadRequestError, Groq, GroqError, RateLimitError

from backend.agent.decision import decision_json_schema
from backend.agent.state import Message
from backend.agent.telemetry import add_usage, logger

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "openai/gpt-oss-120b"


def _recover(error: Exception) -> str | None:
    """Groq's 400 still carries the model's text in failed_generation. Use it."""
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    raw = details.get("failed_generation") if isinstance(details, dict) else None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return raw.strip() or None
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("decision"), dict):
        return json.dumps(data)
    if "action" in data:
        return json.dumps(data)
    if isinstance(data.get("name"), str) and isinstance(data.get("arguments"), dict):
        return json.dumps({"action": data["name"], "reason": "", "arguments": data["arguments"]})
    return None


def call_groq(
    messages: list[Message],
    structured_output: bool = False,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    options = {}
    if structured_output:
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_agent_decision",
                "strict": True,
                "schema": decision_json_schema(),
            },
        }
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        options["reasoning_effort"] = reasoning_effort

    response_format_applied = "response_format" in options
    client = Groq(api_key=api_key)
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0, **options
            )
            break
        except (BadRequestError, RateLimitError, GroqError) as error:
            is_rate_limit = (
                isinstance(error, RateLimitError)
                or getattr(error, "status_code", None) == 429
                or "rate limit" in str(error).lower()
            )
            if is_rate_limit:
                if attempt == 2:
                    raise
                try:
                    resp = getattr(error, "response", None)
                    delay = min(float(resp.headers.get("retry-after", 3)), 10) if resp else 3
                except (AttributeError, TypeError, ValueError):
                    delay = 3
                logger.warning("event=llm_rate_limit_retry delay=%s", delay)
                time.sleep(delay)
                continue

            recovered = _recover(error)
            if not recovered:
                raise
            logger.warning("event=llm_tool_call_recovered model=%s", MODEL)
            return recovered

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    add_usage(input_tokens, output_tokens)
    logger.info(
        "event=llm model=%s response_format_applied=%s input_tokens=%s output_tokens=%s",
        MODEL,
        response_format_applied,
        input_tokens,
        output_tokens,
    )
    return response.choices[0].message.content or ""
