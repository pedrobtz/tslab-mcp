"""Pydantic input models -- one per tool.

These schemas are the package's public contract *and* its documentation: the
field descriptions are what an agent reads when deciding how to call a tool, so
they carry an example value and real constraints rather than a restatement of
the field name. Validation lives here exclusively; tool bodies never
hand-check their inputs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResponseFormat(str, Enum):
    """Whether a response is shaped for a reader or for further computation."""

    MARKDOWN = "markdown"
    JSON = "json"


class ModelFamily(str, Enum):
    """Model cost classes. Statistical models are cheap and always installed."""

    ALL = "all"
    STATISTICAL = "statistical"
    FOUNDATION = "foundation"


class ReportFormat(str, Enum):
    """Output format for an exported report."""

    HTML = "html"
    MARKDOWN = "markdown"


class Metric(str, Enum):
    """Evaluation metrics available from ``utilsforecast.losses``."""

    MASE = "mase"
    SMAPE = "smape"
    MAE = "mae"
    RMSE = "rmse"
    MAPE = "mape"


class _Base(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=False,
    )


def _dedupe(values: list[str]) -> list[str]:
    """Drop repeats while preserving the caller's ordering."""
    seen: list[str] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


class LoadSeriesInput(_Base):
    path: str = Field(
        ...,
        description=(
            "Absolute path to a local CSV or Parquet file in Nixtla long "
            "format, with columns unique_id, ds, y. "
            "Example: '/data/monthly_deposits.parquet'."
        ),
        min_length=1,
    )
    freq: str | None = Field(
        default=None,
        description=(
            "Pandas offset alias, e.g. 'MS' (month start), 'D', 'W-MON', 'h'. "
            "Inferred from the longest series when omitted; pass it explicitly "
            "if inference failed. Gaps are reported but must be regularized in "
            "the source before analysis; the server never fills them silently."
        ),
        max_length=16,
    )
    handle: str | None = Field(
        default=None,
        description=(
            "Optional name to register this panel under, e.g. 'deposits'. "
            "Auto-generated when omitted. Reusing a name replaces that panel "
            "and discards its run log."
        ),
        min_length=1,
        max_length=64,
    )


class DescribeSeriesInput(_Base):
    handle: str = Field(
        ...,
        description="Handle returned by tsf_load_series, e.g. 'deposits'.",
        min_length=1,
    )
    max_series: int = Field(
        default=20,
        description=(
            "Cap on feature rows in the response, to bound context. The reply "
            "reports how many series were omitted."
        ),
        ge=1,
        le=200,
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description=(
            "'markdown' for a table you read directly; 'json' when you intend "
            "to compute on the numbers."
        ),
    )


class ListModelsInput(_Base):
    family: ModelFamily = Field(
        default=ModelFamily.ALL,
        description=(
            "Restrict the probe to 'statistical' (cheap, no weight downloads) "
            "or 'foundation' (pretrained, may download hundreds of MB)."
        ),
    )
    include_unavailable: bool = Field(
        default=True,
        description=(
            "Include models that failed to import, with the reason. Keep this "
            "true so you can tell 'not installed' from 'does not exist'."
        ),
    )


class CrossValidateInput(_Base):
    handle: str = Field(
        ..., description="Handle returned by tsf_load_series.", min_length=1
    )
    models: list[str] = Field(
        ...,
        description=(
            "Model names to compare, e.g. ['SeasonalNaive', 'AutoETS', "
            "'AutoARIMA']. Always include SeasonalNaive as the baseline -- a "
            "model that cannot beat it is not worth deploying. Names come from "
            "tsf_list_models; duplicates are dropped."
        ),
        min_length=1,
        max_length=12,
    )
    h: int = Field(
        ...,
        description=(
            "Forecast horizon in periods, matching the decision being made "
            "(e.g. 12 for a year of monthly data)."
        ),
        ge=1,
        le=5000,
    )
    n_windows: int = Field(
        default=5,
        description=(
            "Rolling-origin evaluation windows. Each window consumes h "
            "observations of history, so n_windows * h must fit inside the "
            "shortest series."
        ),
        ge=1,
        le=50,
    )
    step_size: int | None = Field(
        default=None,
        description=(
            "Observations between window origins. Defaults to h (non-"
            "overlapping windows)."
        ),
        ge=1,
        le=5000,
    )
    metrics: list[Metric] = Field(
        default_factory=lambda: [Metric.MASE, Metric.SMAPE],
        description=(
            "Metrics to aggregate. mase is scale-free and comparable across "
            "series; smape is bounded but unstable near zero."
        ),
        min_length=1,
        max_length=5,
    )

    @field_validator("models")
    @classmethod
    def _unique_models(cls, value: list[str]) -> list[str]:
        return _dedupe(value)

    @field_validator("metrics")
    @classmethod
    def _unique_metrics(cls, value: list[Metric]) -> list[Metric]:
        seen: list[Metric] = []
        for metric in value:
            if metric not in seen:
                seen.append(metric)
        return seen


class SelectModelInput(_Base):
    handle: str = Field(
        ..., description="Handle whose latest cross-validation run to evaluate.",
        min_length=1,
    )
    proposed_model: str = Field(
        ...,
        description=(
            "Model chosen by the agent from the latest CV evidence. The tool "
            "accepts it only when it is tied for the best qualifying score."
        ),
        min_length=1,
    )
    primary_metric: Metric = Field(
        default=Metric.MASE,
        description=(
            "Cross-validation metric used for selection. MASE is the default "
            "agent policy; every supported metric is lower-is-better."
        ),
    )
    min_relative_improvement: float = Field(
        default=0.0,
        description=(
            "Minimum fractional improvement over SeasonalNaive required for a "
            "challenger, e.g. 0.05 requires at least 5%. Leave at 0 to accept "
            "any strictly lower score."
        ),
        ge=0.0,
        le=1.0,
    )
    finalize_baseline: bool = Field(
        default=False,
        description=(
            "When no challenger qualifies, false requests one broader CV "
            "comparison; true validates SeasonalNaive as the final fallback."
        ),
    )


class ForecastInput(_Base):
    handle: str = Field(
        ..., description="Handle returned by tsf_load_series.", min_length=1
    )
    models: list[str] = Field(
        ...,
        description=(
            "Model names to fit. Normally pass the single selected_model "
            "validated by tsf_select_model; multiple or different models "
            "require an explicit selection override after a checkpoint."
        ),
        min_length=1,
        max_length=12,
    )
    h: int = Field(..., description="Forecast horizon in periods.", ge=1, le=5000)
    level: list[int] = Field(
        default_factory=lambda: [80, 95],
        description=(
            "Prediction interval levels as percentages, e.g. [80, 95]. Pass an "
            "empty list for point forecasts only."
        ),
        max_length=5,
    )
    max_preview_rows: int = Field(
        default=12,
        description=(
            "Rows of the forecast to inline in the response. The full frame is "
            "always written to parquet -- read that instead of raising this."
        ),
        ge=0,
        le=100,
    )
    allow_selection_override: bool = Field(
        default=False,
        description=(
            "Allow models that differ from the latest validated selection. "
            "Use only for an explicit user-requested override; otherwise the "
            "structured selection checkpoint is enforced."
        ),
    )

    @field_validator("models")
    @classmethod
    def _unique_models(cls, value: list[str]) -> list[str]:
        return _dedupe(value)

    @field_validator("level")
    @classmethod
    def _valid_levels(cls, value: list[int]) -> list[int]:
        for level in value:
            if not 1 <= level <= 99:
                raise ValueError(
                    f"Interval level {level} is out of range. Use percentages "
                    "strictly between 1 and 99, e.g. [80, 95]."
                )
        return sorted(set(value))


class DetectAnomaliesInput(_Base):
    handle: str = Field(
        ..., description="Handle returned by tsf_load_series.", min_length=1
    )
    model: str = Field(
        ...,
        description=(
            "Single model to use as the detector, e.g. 'SeasonalNaive'. The "
            "detector defines 'expected'; a weak model flags its own errors."
        ),
        min_length=1,
    )
    h: int = Field(
        default=1,
        description="Horizon of the rolling test. Keep at 1 for point anomalies.",
        ge=1,
        le=100,
    )
    n_windows: int | None = Field(
        default=None,
        description=(
            "Rolling-origin windows to test. Defaults to the maximum that "
            "preserves enough training and interval-calibration history for "
            "the chosen model. This can still be SLOW -- set this (e.g. 12) "
            "to bound the cost."
        ),
        ge=1,
        le=500,
    )
    level: int = Field(
        default=99,
        description=(
            "Two-sided interval confidence level as a percentage. Higher flags "
            "fewer, more extreme points."
        ),
        ge=50,
        le=99,
    )
    max_flagged: int = Field(
        default=100,
        description="Cap on flagged rows inlined in the response.",
        ge=0,
        le=500,
    )
    allow_selection_override: bool = Field(
        default=False,
        description=(
            "Allow a detector that differs from the latest validated model. "
            "Use only for an explicit user-directed override; complete analyses "
            "use the forecast selection for a consistent expectation model."
        ),
    )


class ExportRunInput(_Base):
    handle: str = Field(
        ...,
        description="Handle whose run log should be pinned to a manifest.",
        min_length=1,
    )
    note: str | None = Field(
        default=None,
        description=(
            "Your rationale, in prose: why this model was chosen, what the "
            "metric table showed, what you rejected. This is the only place a "
            "reasoning trace survives the session -- write it."
        ),
        max_length=4000,
    )


class ExportReportInput(_Base):
    handle: str | None = Field(
        default=None,
        description=(
            "Handle to report on, e.g. 'deposits'. Builds the report from this "
            "session's run log. Pass this or manifest_path, not both."
        ),
        min_length=1,
    )
    manifest_path: str | None = Field(
        default=None,
        description=(
            "Path to a manifest written earlier by tsf_export_run. Re-renders "
            "that run offline, without needing its handle to still be loaded."
        ),
        min_length=1,
    )
    format: ReportFormat = Field(
        default=ReportFormat.HTML,
        description=(
            "'html' for a self-contained page to open or send; 'markdown' to "
            "paste into a PR, issue, or wiki."
        ),
    )
    note: str | None = Field(
        default=None,
        description=(
            "Rationale to headline the report. Ignored when reporting from a "
            "manifest_path, which carries the note it was written with."
        ),
        max_length=4000,
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ExportReportInput:
        if bool(self.handle) == bool(self.manifest_path):
            raise ValueError(
                "Pass exactly one of handle or manifest_path: handle to report "
                "on the current session, manifest_path to re-render a manifest "
                "written earlier."
            )
        return self
