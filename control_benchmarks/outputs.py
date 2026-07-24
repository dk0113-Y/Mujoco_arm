from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TIMESERIES_FIELDS = (
    "episode_id",
    "experiment",
    "case",
    "control_cycle",
    "sim_time",
    "q",
    "dq",
    "q_target",
    "dq_target",
    "position_error",
    "velocity_error",
    "feedback_torque",
    "dynamics_compensation",
    "gravity",
    "coriolis_centrifugal",
    "passive_force",
    "raw_torque",
    "rate_limited_torque",
    "final_torque",
    "actuator_force",
    "saturation_mask",
    "rate_limit_mask",
    "joint_limit_mask",
    "velocity_limit_mask",
    "finite_value_status",
    "termination_reason",
)

EPISODE_FIELDS = (
    "episode_id",
    "experiment",
    "case",
    "position_rmse",
    "maximum_absolute_position_error",
    "velocity_rmse",
    "maximum_absolute_torque",
    "rms_torque",
    "torque_saturation_count",
    "torque_saturation_ratio",
    "torque_rate_limit_count",
    "torque_rate_limit_ratio",
    "maximum_joint_velocity",
    "final_position_error",
    "steady_state_error",
    "overshoot",
    "settling_time",
    "finite_value_status",
    "terminated",
    "termination_reason",
    "simulated_duration",
    "wall_clock_duration",
)


def strict_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return strict_json_value(value.tolist())
    if isinstance(value, np.generic):
        return strict_json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON output contains NaN or Infinity")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    raise TypeError(f"Unsupported JSON output type: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    normalized = strict_json_value(value)
    path.write_text(
        json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    normalized = strict_json_value(value)
    if isinstance(normalized, (list, dict)):
        return json.dumps(
            normalized,
            separators=(",", ":"),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    return normalized


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    rows_value = list(rows)
    allowed = set(fieldnames)
    for row in rows_value:
        unknown = set(row) - allowed
        missing = allowed - set(row)
        if unknown or missing:
            raise ValueError(
                f"CSV row schema mismatch; unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}"
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="raise"
        )
        writer.writeheader()
        for row in rows_value:
            writer.writerow(
                {field: _csv_value(row[field]) for field in fieldnames}
            )


def prepare_output_directory(path: str | Path, *, overwrite: bool) -> Path:
    output_path = Path(path).expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_path}")
    if output_path.is_dir() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_path}; use --overwrite"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        known_files = {
            "run_manifest.json",
            "episode_metrics.csv",
            "timeseries.csv",
            "summary.json",
            "config_snapshot.toml",
        }
        existing = list(output_path.iterdir())
        unknown = [
            item
            for item in existing
            if item.name not in known_files or not item.is_file()
        ]
        if unknown:
            raise FileExistsError(
                "Refusing overwrite because output directory contains unknown files: "
                + ", ".join(sorted(item.name for item in unknown))
            )
        for name in sorted(known_files):
            candidate = output_path / name
            if candidate.is_file():
                candidate.unlink()
    return output_path
