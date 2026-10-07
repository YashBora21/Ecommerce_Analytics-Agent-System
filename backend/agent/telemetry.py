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
        error = update["error"] if "error" in update else state.get("error", "")
        details = {
            "node": name,
            "step": update.get("steps", state.get("steps", 0)),
            "action": action.get("name", ""),
            "error": error,
            "tokens": get_usage(),
        }
        if name in {"generate_sql", "execute"}:
            sql = update.get("sql", state.get("sql", ""))
            details["sql"] = " ".join(sql.split())
            logger.info(
                "node=%(node)s event=end step=%(step)s action=%(action)s "
                "sql=%(sql)s error=%(error)s tokens=%(tokens)s",
                details,
            )
        else:
            logger.info(
                "node=%(node)s event=end step=%(step)s action=%(action)s "
                "error=%(error)s tokens=%(tokens)s",
                details,
            )
        logger.info("*" * 70)
        return update

    return run
