from backend.agent.memory import format_evidence
from backend.agent.state import MAX_CHART_TRIES, QueryState


def make_partial_answer(remember):
    def partial_answer(state: QueryState) -> QueryState:
        evidence = format_evidence(state)[:3000]
        if evidence:
            if state.get("chart_tries", 0) >= MAX_CHART_TRIES:
                prefix = "I could not build the chart after one retry."
            else:
                prefix = "I could not fully complete the analysis within the allowed steps."
            reason = f" Reason: {state['error']}" if state.get("error") else ""
            answer = f"{prefix}{reason} Here is the evidence collected so far:\n{evidence}"
        else:
            reason = state.get("error", "Unknown query failure")
            answer = (
                "I could not run a valid query within the allowed attempts. "
                f"Reason: {reason} Please rephrase the question."
            )
        return remember(state, answer)

    return partial_answer

