from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from backend.tools.chart import SUPPORTED_CHARTS
from backend.tools.statistics import SUPPORTED_OPERATIONS


Text = Annotated[str, StringConstraints(strip_whitespace=True)]
RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AnswerArgs(BaseModel):
    final_answer: RequiredText


class QueryArgs(BaseModel):
    next_query_goal: RequiredText


class StatisticsArgs(BaseModel):
    operation: RequiredText
    column: RequiredText
    second_column: Text = ""
    percentile: float | None = None

    @field_validator("operation")
    @classmethod
    def supported_operation(cls, operation: str) -> str:
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported statistics operation: {operation}")
        return operation

    @model_validator(mode="after")
    def required_arguments(self):
        if self.operation in {"correlation", "linear_regression"} and not self.second_column:
            raise ValueError(f"{self.operation} requires second_column")
        if self.operation == "percentile" and self.percentile is None:
            raise ValueError("percentile requires a number")
        return self


class ChartArgs(BaseModel):
    chart_type: RequiredText
    x_column: Text = ""
    y_column: Text = ""
    value_column: Text = ""
    path_columns: list[RequiredText] = Field(default_factory=list)
    title: Text = ""
    final_answer: RequiredText

    @field_validator("chart_type")
    @classmethod
    def supported_chart(cls, chart_type: str) -> str:
        if chart_type not in SUPPORTED_CHARTS:
            raise ValueError(f"unsupported chart type: {chart_type}")
        return chart_type

    @model_validator(mode="after")
    def required_columns(self):
        if self.chart_type == "heatmap" and not all(
            (self.x_column, self.y_column, self.value_column)
        ):
            raise ValueError("heatmap requires x, y, and value columns")
        if self.chart_type == "treemap" and (
            not self.path_columns or not self.value_column
        ):
            raise ValueError("treemap requires path and value columns")
        if self.chart_type not in {"heatmap", "treemap"} and not all(
            (self.x_column, self.y_column)
        ):
            raise ValueError("chart requires x and y columns")
        return self


ACTION_ARGUMENTS = {
    "answer": AnswerArgs,
    "query_sql": QueryArgs,
    "calculate_statistics": StatisticsArgs,
    "create_chart": ChartArgs,
}


class Decision(BaseModel):
    action: RequiredText
    reason: Text
    arguments: dict

    @field_validator("action")
    @classmethod
    def supported_action(cls, action: str) -> str:
        if action not in ACTION_ARGUMENTS:
            raise ValueError("invalid next action")
        return action





def parse_action_json(value: str) -> dict:
    decision = Decision.model_validate_json(value)
    arguments = ACTION_ARGUMENTS[decision.action].model_validate(decision.arguments)
    return {"name": decision.action, "args": arguments.model_dump()}


