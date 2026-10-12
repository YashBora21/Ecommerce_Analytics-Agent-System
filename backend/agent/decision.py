from copy import deepcopy
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

from backend.tools.chart import SUPPORTED_CHARTS
from backend.tools.statistics import SUPPORTED_OPERATIONS


Text = Annotated[str, StringConstraints(strip_whitespace=True)]
RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FinalAnswer = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=5_000),
]


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerArgs(Arguments):
    final_answer: FinalAnswer


class QueryArgs(Arguments):
    sql: RequiredText
    result_scope: Literal["complete", "bounded"] = "complete"


class StatisticsArgs(Arguments):
    operation: RequiredText
    column: RequiredText
    second_column: Text | None = None
    percentile: float | None = None
    artifact_id: Text | None = None

    @field_validator("operation")
    @classmethod
    def supported_operation(cls, operation: str) -> str:
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported statistics operation: {operation}")
        return operation

    @model_validator(mode="after")
    def required_arguments(self):
        self.second_column = self.second_column or ""
        self.artifact_id = self.artifact_id or ""
        if self.operation in {"correlation", "linear_regression"} and not self.second_column:
            raise ValueError(f"{self.operation} requires second_column")
        if self.operation == "percentile" and self.percentile is None:
            raise ValueError("percentile requires a number")
        return self


class ChartArgs(Arguments):
    chart_type: RequiredText
    x_column: Text | None = None
    y_column: Text | None = None
    value_column: Text | None = None
    path_columns: list[RequiredText] | None = None
    title: Text | None = None
    artifact_id: Text | None = None
    final_answer: FinalAnswer

    @field_validator("chart_type")
    @classmethod
    def supported_chart(cls, chart_type: str) -> str:
        if chart_type not in SUPPORTED_CHARTS:
            raise ValueError(f"unsupported chart type: {chart_type}")
        return chart_type

    @model_validator(mode="after")
    def required_columns(self):
        self.x_column = self.x_column or ""
        self.y_column = self.y_column or ""
        self.value_column = self.value_column or ""
        self.path_columns = self.path_columns or []
        self.title = self.title or ""
        self.artifact_id = self.artifact_id or ""
        if self.chart_type == "heatmap" and not all((self.x_column, self.y_column, self.value_column)):
            raise ValueError("heatmap requires x, y, and value columns")
        if self.chart_type == "treemap" and (not self.path_columns or not self.value_column):
            raise ValueError("treemap requires path and value columns")
        if self.chart_type not in {"heatmap", "treemap"} and not all((self.x_column, self.y_column)):
            raise ValueError("chart requires x and y columns")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Text = ""


class AnswerDecision(Decision):
    action: Literal["answer", "clarify", "stop"]
    arguments: AnswerArgs


class QueryDecision(Decision):
    action: Literal["query_sql"]
    arguments: QueryArgs


class StatisticsDecision(Decision):
    action: Literal["calculate_statistics"]
    arguments: StatisticsArgs


class ChartDecision(Decision):
    action: Literal["create_chart"]
    arguments: ChartArgs


DecisionType = AnswerDecision | QueryDecision | StatisticsDecision | ChartDecision
DECISION_ADAPTER = TypeAdapter(DecisionType)


class DecisionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionType


ACTIONS = {
    "answer",
    "clarify",
    "stop",
    "query_sql",
    "calculate_statistics",
    "create_chart",
}


def _strict_schema(value):
    if isinstance(value, dict):
        schema = {key: _strict_schema(item) for key, item in value.items() if key != "default"}
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            schema["additionalProperties"] = False
            schema["required"] = list(properties)
        return schema
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    return value


def decision_json_schema() -> dict:
    """Return Groq's strict JSON schema generated from the Pydantic models."""
    return _strict_schema(deepcopy(DecisionEnvelope.model_json_schema()))


def _as_action(decision: DecisionType) -> dict:
    return {
        "name": decision.action,
        "args": decision.arguments.model_dump(),
    }


def parse_action(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("action") not in ACTIONS:
        raise ValueError("invalid next action")
    return _as_action(DECISION_ADAPTER.validate_python(value))


def parse_action_json(value: str) -> dict:
    try:
        return _as_action(DecisionEnvelope.model_validate_json(value).decision)
    except ValueError:
        return _as_action(DECISION_ADAPTER.validate_json(value))
