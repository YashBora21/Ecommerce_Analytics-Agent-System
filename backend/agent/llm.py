import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, GroqError

from backend.agent.prompt import (
    RESULT_ASSESSMENT_PROMPT,
    SQL_SYSTEM_PROMPT,
)
from backend.agent.state import Message
from backend.agent.telemetry import add_usage, logger


load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def call_groq(messages: list[Message]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    options = {}
    if messages[0]["content"] in {
            RESULT_ASSESSMENT_PROMPT,
        SQL_SYSTEM_PROMPT,
    }:
        options["response_format"] = {"type": "json_object"}

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    completions = Groq(api_key=api_key).chat.completions
    request = {"model": model, "messages": messages, "temperature": 0}
    try:
        response = completions.create(**request, **options)
    except GroqError as error:
        message = str(error).lower()
        if "response_format" not in options or not any(
            marker in message for marker in ("failed to generate json", "failed_generation")
        ):
            raise
        logger.warning("event=llm_json_fallback model=%s error=%s", model, str(error)[:300])
        response = completions.create(**request)
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    add_usage(input_tokens, output_tokens)
    logger.info(
        "event=llm model=%s input_tokens=%s output_tokens=%s",
        model,
        input_tokens,
        output_tokens,
    )
    return response.choices[0].message.content or ""

