from groq import GroqError
from pydantic import ValidationError

from backend.agent.decision import parse_action_json
from backend.agent.memory import format_evidence, recent_context
from backend.agent.prompt import RESULT_ASSESSMENT_PROMPT
from backend.agent.state import LlmCall, QueryState
from backend.agent.telemetry import logger


MAX_ASSESSMENT_ATTEMPTS = 2


def _artifact_context(state: QueryState) -> list[dict]:
    return [
        {
            "artifact_id": item.get("artifact_id"),
            "relation": item.get("relation"),
            "row_count": item.get("row_count"),
            "columns": item.get("columns"),
            "expires_at": item.get("expires_at"),
        }
        for item in state.get("artifacts", [])
        if item.get("complete")
    ]


def make_assess_result(llm_call: LlmCall, remember):
    def assess_result(state: QueryState) -> QueryState:
        steps = state.get("steps", 0) + 1
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Live schema:\n{state.get('schema_context', '')}\n\n"
            f"SQL evidence:\n{format_evidence(state)}\n\n"
            f"Available artifacts: {_artifact_context(state)}\n"
            f"Chart already generated: {'yes' if state.get('chart') else 'no'}\n"
            f"Previous tool error: {state.get('error', '')}"
        )
        messages = [{"role": "system", "content": RESULT_ASSESSMENT_PROMPT}]
        context = recent_context(state)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        try:
            for attempt in range(1, MAX_ASSESSMENT_ATTEMPTS + 1):
                response = llm_call(messages)
                try:
                    action = parse_action_json(response)
                    if action["name"] in {"answer", "clarify", "stop"}:
                        return {
                            **remember(state, action["args"]["final_answer"]),
                            "action": action,
                            "steps": steps,
                            "error": "",
                        }
                    return {"action": action, "steps": steps, "error": ""}
                except (ValidationError, ValueError) as error:
                    logger.warning(
                        "event=assessment_retry attempt=%s error=%s response=%s",
                        attempt,
                        error,
                        response[:300],
                    )
                    if attempt == MAX_ASSESSMENT_ATTEMPTS:
                        return {
                            "steps": steps,
                            "error": f"Could not assess query results: {error}",
                        }
                    messages.extend(
                        [
                            {"role": "assistant", "content": response},
                            {
                                "role": "user",
                                "content": (
                                    f"That response failed validation: {error}. "
                                    "Return corrected JSON matching the required action schema."
                                ),
                            },
                        ]
                    )
        except (RuntimeError, GroqError) as error:
            return {"steps": steps, "error": f"Could not assess query results: {error}"}

    return assess_result
