from groq import GroqError
from pydantic import ValidationError

from backend.agent.decision import parse_action_json
from backend.agent.memory import (
    asks_to_recall_previous_result,
    format_evidence,
    recent_context,
)
from backend.agent.prompt import RESULT_ASSESSMENT_PROMPT
from backend.agent.state import LlmCall, QueryState
from backend.agent.telemetry import logger


ASSESSMENT_TOKEN_LIMITS = (1600, 3200)


def _artifact_context(state: QueryState) -> list[dict]:
    return [
        {
            "artifact_id": item.get("artifact_id"),
            "relation": item.get("relation"),
            "row_count": item.get("row_count"),
            "columns": item.get("columns"),
            "has_limit": bool(item.get("has_limit", False)),
            "expires_at": item.get("expires_at"),
        }
        for item in state.get("artifacts", [])
        if item.get("complete")
    ]


def make_assess_result(llm_call: LlmCall, remember):
    def assess_result(state: QueryState) -> QueryState:
        steps = state.get("steps", 0) + 1
        downloadable_artifacts = [
            item["artifact_id"]
            for item in state.get("evidence", [])
            if item.get("artifact_id")
        ]
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Live schema:\n{state.get('schema_context', '')}\n\n"
            f"SQL evidence:\n{format_evidence(state)}\n\n"
            f"Available artifacts: {_artifact_context(state)}\n"
            f"Downloadable artifacts created this turn: {downloadable_artifacts}\n"
            f"Chart already generated: {'yes' if state.get('chart') else 'no'}\n"
            f"Previous tool error: {state.get('error', '')}"
        )
        messages = [{"role": "system", "content": RESULT_ASSESSMENT_PROMPT}]
        context = recent_context(state)
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})

        try:
            for attempt, token_limit in enumerate(ASSESSMENT_TOKEN_LIMITS, start=1):
                try:
                    try:
                        response = llm_call(
                            messages,
                            structured_output=True,
                            max_tokens=token_limit,
                            reasoning_effort="low",
                        )
                    except TypeError:
                        try:
                            response = llm_call(messages, structured_output=True)
                        except TypeError:
                            response = llm_call(messages)
                except (RuntimeError, GroqError) as error:
                    output_was_cut_off = "max completion tokens" in str(error).lower()
                    if output_was_cut_off and attempt < len(ASSESSMENT_TOKEN_LIMITS):
                        logger.warning(
                            "event=assessment_output_retry attempt=%s next_max_tokens=%s",
                            attempt,
                            ASSESSMENT_TOKEN_LIMITS[attempt],
                        )
                        continue
                    raise
                try:
                    action = parse_action_json(response)
                    if (
                        action["name"] == "answer"
                        and not state.get("evidence")
                        and not asks_to_recall_previous_result(state["question"])
                    ):
                        raise ValueError(
                            "A database answer requires SQL evidence from this turn; "
                            "choose query_sql, clarify, or stop"
                        )
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
                    if attempt == len(ASSESSMENT_TOKEN_LIMITS):
                        return {
                            "steps": steps,
                            "error": "The model returned an invalid action after correction.",
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
            logger.warning("event=assessment_failed error=%s", error)
            status = getattr(error, "status_code", None)
            message = (
                "The model service rate limit was reached."
                if status == 429
                else "The model could not produce a valid next action."
            )
            return {"steps": steps, "error": message}

    return assess_result
