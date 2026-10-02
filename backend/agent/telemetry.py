import logging
import os
from contextvars import ContextVar


_usage = ContextVar(
    "groq_usage",
    default={"input_tokens": 0, "output_tokens": 0},
)

logger = logging.getLogger("ecommerce_agent")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
logger.propagate = False


def reset_usage() -> None:
    _usage.set({"input_tokens": 0, "output_tokens": 0})


def add_usage(input_tokens: int, output_tokens: int) -> None:
    usage = _usage.get()
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens


def get_usage() -> dict[str, int]:
    usage = _usage.get()
    return {
        **usage,
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
    }


def logged_node(name, node):
    def run(state):
        logger.info("node=%s event=start step=%s", name, state.get("steps", 0))
        try:
            update = node(state)
        except Exception:
            logger.exception("node=%s event=failed", name)
            raise

        action = update["action"] if "action" in update else state.get("action", {})
        route = update["route"] if "route" in update else state.get("route", "")
        sql = update["sql"] if "sql" in update else state.get("sql", "")
        error = update["error"] if "error" in update else state.get("error", "")
        logger.info(
            "node=%s event=end step=%s route=%s action=%s sql=%s error=%s tokens=%s",
            name,
            update.get("steps", state.get("steps", 0)),
            route,
            action.get("name", ""),
            sql,
            error,
            get_usage(),
        )
        return update

    return run
