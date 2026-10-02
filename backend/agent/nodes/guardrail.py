import json
import re

from groq import GroqError

from backend.agent.memory import recent_context
from backend.agent.prompt import GUARDRAIL_SYSTEM_PROMPT
from backend.agent.state import GREETING_REPLY, OUT_OF_SCOPE_REPLY, LlmCall, QueryState


def _is_greeting(question: str) -> bool:
    normalized = re.sub(r"[^a-z ]", "", question.lower()).strip()
    return normalized in {
        "hi", "hello", "hey", "good morning", "good afternoon",
        "good evening", "how are you",
    }


def make_guardrail(llm_call: LlmCall):
    def guardrail(state: QueryState) -> QueryState:
        if _is_greeting(state["question"]):
            return {"route": "greeting", "error": ""}

        context = recent_context(state)
        content = state["question"]
        if context:
            content = f"{context}\n\nCurrent message: {content}"
        try:
            decision = json.loads(llm_call([
                {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]))
            in_scope = decision.get("is_in_scope")
            greeting = decision.get("is_greeting")
            if not isinstance(in_scope, bool) or not isinstance(greeting, bool):
                raise ValueError("guardrail decisions must be booleans")
            if not isinstance(decision.get("reason"), str):
                raise ValueError("guardrail reason must be text")
            route = "greeting" if greeting else "in_scope" if in_scope else "out_of_scope"
            return {"route": route, "error": ""}
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError, GroqError) as error:
            return {"error": f"Could not classify question: {error}"}

    return guardrail


def make_greeting(remember):
    return lambda state: remember(state, GREETING_REPLY)


def make_out_of_scope(remember):
    return lambda state: remember(state, OUT_OF_SCOPE_REPLY)
