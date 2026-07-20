from __future__ import annotations

import csv
from dataclasses import fields
from enum import Enum
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation import EpisodeResult


EPISODE_METADATA_FIELDS = (
    "benchmark_name",
    "pair_id",
    "method_id",
    "external_state_source",
    "execution_index",
    "episode_fingerprint",
    "pair_valid",
    "program_error",
)
_EXPANDED_EPISODE_FIELDS = frozenset({"key_errors", "stage_durations"})
EPISODE_RESULT_FIELDS = tuple(
    field.name
    for field in fields(EpisodeResult)
    if field.name not in _EXPANDED_EPISODE_FIELDS
)

PAIRED_RESULT_FIELDS = (
    "pair_id",
    "seed",
    "pair_valid",
    "pair_error",
    "fingerprint",
    "oracle_ground_truth_success",
    "vision_ground_truth_success",
    "oracle_controller_reported_success",
    "vision_controller_reported_success",
    "oracle_failure_reason",
    "vision_failure_reason",
    "oracle_final_stage",
    "vision_final_stage",
    "oracle_simulation_time",
    "vision_simulation_time",
    "oracle_collision_count",
    "vision_collision_count",
    "vision_object_position_error",
    "vision_target_position_error",
    "outcome_category",
)

FAILURE_COUNT_FIELDS = ("method_id", "failure_reason", "count")


def validate_finite_json(value: Any) -> Any:
    """Return a JSON-compatible value, rejecting non-finite numeric data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return validate_finite_json(value.value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Non-finite numeric output is forbidden: {result}")
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): validate_finite_json(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [validate_finite_json(item) for item in value]
    raise TypeError(f"Output value is not JSON serializable: {type(value).__name__}")


def _csv_scalar(value: Any) -> Any:
    safe = validate_finite_json(value)
    if safe is None:
        return ""
    if isinstance(safe, (list, dict)):
        return json.dumps(
            safe,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            sort_keys=isinstance(safe, dict),
        )
    return safe


def episode_fieldnames(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    row_list = list(rows)
    standard = EPISODE_METADATA_FIELDS + EPISODE_RESULT_FIELDS
    standard_set = set(standard)
    dynamic = sorted(
        {
            str(key)
            for row in row_list
            for key in row
            if key not in standard_set
        }
    )
    return standard + tuple(dynamic)


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    output_path = Path(path)
    row_list = list(rows)
    allowed = set(fieldnames)
    for row in row_list:
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(f"CSV row contains fields absent from schema: {unknown}")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in row_list:
            writer.writerow(
                {name: _csv_scalar(row.get(name)) for name in fieldnames}
            )


def write_json(path: str | Path, value: Any) -> None:
    safe = validate_finite_json(value)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            safe,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        stream.write("\n")
