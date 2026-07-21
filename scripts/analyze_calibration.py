from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tomllib
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.pairing import EpisodeFingerprint


REQUIRED_INPUT_FILES = (
    "run_manifest.json",
    "config_snapshot.toml",
    "protocol_snapshot.toml",
    "seeds.json",
    "episodes.csv",
    "paired_results.csv",
    "failure_counts.csv",
    "summary.json",
    "production_metrics.json",
    "run.log",
)
OPTIONAL_INPUT_FILES = ("preflight_environment.json", "manual_assessment.json")
OUTPUT_FILES = (
    "calibration_analysis.json",
    "calibration_analysis.csv",
    "calibration_round_0_report.md",
)
METHOD_IDS = ("b0_oracle", "b1_vision")
PAIR_CATEGORIES = (
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
)
NONFINITE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:NaN|[-+]?Inf(?:inity)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


class CalibrationAnalysisError(ValueError):
    """Raised when a Round 0 archive is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise CalibrationAnalysisError(
            f"{path.name} contains a non-finite JSON value: {value}"
        )

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationAnalysisError(f"Cannot parse {path.name}: {exc}") from exc


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CalibrationAnalysisError(f"Cannot parse {path.name}: {exc}") from exc
    if not fieldnames:
        raise CalibrationAnalysisError(f"{path.name} has no header")
    return fieldnames, rows


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationAnalysisError(f"{label} must be a JSON object")
    return value


def _bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1", 1):
        return True
    if value in ("False", "false", "0", 0):
        return False
    raise CalibrationAnalysisError(f"{field} must be boolean, got {value!r}")


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise CalibrationAnalysisError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationAnalysisError(
            f"{field} must be an integer, got {value!r}"
        ) from exc
    if str(value).strip() not in {str(result), f"+{result}"}:
        try:
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError):
            raise CalibrationAnalysisError(
                f"{field} must be an integer, got {value!r}"
            ) from None
    return result


def _float(value: Any, field: str, *, optional: bool = False) -> float | None:
    if value in ("", None) and optional:
        return None
    if isinstance(value, bool):
        raise CalibrationAnalysisError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationAnalysisError(
            f"{field} must be numeric, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise CalibrationAnalysisError(f"{field} contains NaN or Inf")
    return result


def _json_cell(
    value: str,
    field: str,
    *,
    optional: bool = False,
) -> Any:
    if value == "" and optional:
        return None
    try:
        return json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CalibrationAnalysisError(
                    f"{field} contains a non-finite JSON value: {constant}"
                )
            ),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise CalibrationAnalysisError(
            f"{field} is not valid composite JSON: {value!r}"
        ) from exc


def _rate(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else count / denominator


def _numeric_summary(values: Iterable[float | None]) -> dict[str, Any]:
    selected = [float(value) for value in values if value is not None]
    return {
        "count": len(selected),
        "minimum": min(selected) if selected else None,
        "median": statistics.median(selected) if selected else None,
        "mean": statistics.fmean(selected) if selected else None,
        "maximum": max(selected) if selected else None,
    }


def _hash_seed_list(seeds: Sequence[int]) -> str:
    payload = ("\n".join(str(seed) for seed in seeds) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint(row: Mapping[str, str]) -> EpisodeFingerprint:
    return EpisodeFingerprint.from_episode_result(
        {
            "seed": _int(row.get("seed"), "episodes.seed"),
            "sampled_pick_position": _json_cell(
                row.get("sampled_pick_position", ""),
                "episodes.sampled_pick_position",
            ),
            "sampled_place_position": _json_cell(
                row.get("sampled_place_position", ""),
                "episodes.sampled_place_position",
            ),
            "pick_region": row.get("pick_region"),
            "place_region": row.get("place_region"),
            "sampled_mass": _float(
                row.get("sampled_mass"),
                "episodes.sampled_mass",
            ),
            "sampled_friction": _json_cell(
                row.get("sampled_friction", ""),
                "episodes.sampled_friction",
            ),
        }
    )


def _ensure_finite_text(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CalibrationAnalysisError(f"Cannot read {path.name}: {exc}") from exc
        if NONFINITE_PATTERN.search(text):
            raise CalibrationAnalysisError(
                f"{path.name} contains a NaN/Inf token"
            )


def _validate_archive(output_dir: Path) -> dict[str, Any]:
    missing = [
        name for name in REQUIRED_INPUT_FILES if not (output_dir / name).is_file()
    ]
    if missing:
        raise CalibrationAnalysisError(
            "Round 0 archive is missing required files: " + ", ".join(missing)
        )
    _ensure_finite_text(output_dir / name for name in REQUIRED_INPUT_FILES)

    manifest = _require_mapping(
        _strict_json(output_dir / "run_manifest.json"),
        "run_manifest.json",
    )
    seeds_doc = _require_mapping(
        _strict_json(output_dir / "seeds.json"),
        "seeds.json",
    )
    summary = _require_mapping(
        _strict_json(output_dir / "summary.json"),
        "summary.json",
    )
    production_metrics = _require_mapping(
        _strict_json(output_dir / "production_metrics.json"),
        "production_metrics.json",
    )
    optional: dict[str, Any] = {}
    for name in OPTIONAL_INPUT_FILES:
        path = output_dir / name
        if path.is_file():
            value = _strict_json(path)
            optional[name] = dict(_require_mapping(value, name))

    _, episodes = _csv_rows(output_dir / "episodes.csv")
    _, paired = _csv_rows(output_dir / "paired_results.csv")
    _, failure_counts = _csv_rows(output_dir / "failure_counts.csv")

    expected_manifest = {
        "total_requested_pairs": 30,
        "completed_pairs": 30,
        "invalid_pairs": 0,
        "unhandled_errors": 0,
        "protocol_id": "evaluation_protocol",
        "protocol_version": "1.0.1",
        "metrics_schema_version": "1.0.0",
        "split_id": "evaluation_protocol_v1",
        "split_name": "calibration",
        "calibration_run": True,
        "baseline_frozen": False,
        "automatic_parameter_search": False,
        "pilot": False,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise CalibrationAnalysisError(
                f"run_manifest.{field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if manifest.get("methods") != list(METHOD_IDS):
        raise CalibrationAnalysisError("Manifest method list is not the formal pair")
    if manifest.get("method_execution_order") != list(METHOD_IDS):
        raise CalibrationAnalysisError("Manifest method order is not Oracle then Vision")
    if manifest.get("effective_overrides") != {}:
        raise CalibrationAnalysisError("Round 0 has unexpected effective overrides")
    if manifest.get("git_dirty") is not False or manifest.get("git_status_short") != []:
        raise CalibrationAnalysisError("Round 0 did not start from a clean Git state")
    if manifest.get("unhandled_error_details") not in ([], None):
        raise CalibrationAnalysisError("Manifest contains unhandled error details")

    seeds_value = seeds_doc.get("seeds")
    if not isinstance(seeds_value, list):
        raise CalibrationAnalysisError("seeds.json.seeds must be a list")
    seeds = [_int(seed, "seeds.json.seeds") for seed in seeds_value]
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise CalibrationAnalysisError(
            "Calibration must contain exactly 30 unique seeds"
        )
    if seeds_doc.get("seed_count") != 30:
        raise CalibrationAnalysisError("seeds.json seed_count must be 30")
    if seeds_doc.get("duplicates_present") is not False:
        raise CalibrationAnalysisError("seeds.json reports duplicate seeds")
    if seeds_doc.get("pilot") is not False:
        raise CalibrationAnalysisError("Calibration seeds cannot be marked pilot")
    if _hash_seed_list(seeds) != manifest.get("seed_file_sha256"):
        raise CalibrationAnalysisError(
            "Seed order/content does not match manifest seed_file_sha256"
        )

    config_hash = _sha256(output_dir / "config_snapshot.toml")
    protocol_hash = _sha256(output_dir / "protocol_snapshot.toml")
    if config_hash != manifest.get("config_sha256"):
        raise CalibrationAnalysisError("Config snapshot SHA-256 mismatch")
    if protocol_hash != manifest.get("protocol_config_sha256"):
        raise CalibrationAnalysisError("Protocol snapshot SHA-256 mismatch")
    with (output_dir / "protocol_snapshot.toml").open("rb") as stream:
        protocol_toml = tomllib.load(stream)
    if protocol_toml.get("protocol", {}).get("protocol_version") != "1.0.1":
        raise CalibrationAnalysisError("Protocol snapshot version is not 1.0.1")
    if (
        protocol_toml.get("protocol", {}).get("metrics_schema_version")
        != "1.0.0"
    ):
        raise CalibrationAnalysisError("Metrics schema version is not 1.0.0")
    calibration_toml = protocol_toml.get("calibration", {})
    if calibration_toml.get("baseline_frozen") is not False:
        raise CalibrationAnalysisError("Protocol snapshot marks baseline frozen")
    if calibration_toml.get("automatic_parameter_search") is not False:
        raise CalibrationAnalysisError("Protocol snapshot enables automatic search")

    if len(episodes) != 60:
        raise CalibrationAnalysisError(
            f"episodes.csv contains {len(episodes)} rows, expected 60"
        )
    method_counts = Counter(row.get("method_id") for row in episodes)
    if method_counts != Counter({"b0_oracle": 30, "b1_vision": 30}):
        raise CalibrationAnalysisError(
            f"Episode method counts are invalid: {dict(method_counts)}"
        )
    pair_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(episodes):
        label = f"episodes row {index + 2}"
        if not _bool(row.get("pair_valid"), f"{label}.pair_valid"):
            raise CalibrationAnalysisError(f"{label} belongs to an invalid pair")
        if row.get("program_error") not in ("", None):
            raise CalibrationAnalysisError(f"{label} contains a program error")
        if row.get("result_fields_complete") not in ("True", "true", "1"):
            raise CalibrationAnalysisError(f"{label} has incomplete result fields")
        if row.get("protocol_version") != "1.0.1":
            raise CalibrationAnalysisError(f"{label} protocol version mismatch")
        if row.get("split_name") != "calibration":
            raise CalibrationAnalysisError(f"{label} split mismatch")
        if row.get("config_sha256") != config_hash:
            raise CalibrationAnalysisError(f"{label} config hash mismatch")
        digest = _fingerprint(row).digest
        if digest != row.get("episode_fingerprint"):
            raise CalibrationAnalysisError(f"{label} fingerprint mismatch")
        pair_id = row.get("pair_id")
        if not pair_id:
            raise CalibrationAnalysisError(f"{label} has no pair_id")
        pair_map[pair_id].append(row)

    if len(pair_map) != 30:
        raise CalibrationAnalysisError(
            f"episodes.csv contains {len(pair_map)} pair IDs, expected 30"
        )
    for pair_id, rows in pair_map.items():
        if len(rows) != 2 or {row["method_id"] for row in rows} != set(METHOD_IDS):
            raise CalibrationAnalysisError(
                f"{pair_id} does not contain exactly one B0 and one B1 episode"
            )
        if len({row["seed"] for row in rows}) != 1:
            raise CalibrationAnalysisError(f"{pair_id} seed mismatch")
        if len({row["episode_fingerprint"] for row in rows}) != 1:
            raise CalibrationAnalysisError(f"{pair_id} fingerprint mismatch")

    if len(paired) != 30:
        raise CalibrationAnalysisError(
            f"paired_results.csv contains {len(paired)} rows, expected 30"
        )
    paired_seed_order: list[int] = []
    paired_by_id: dict[str, dict[str, str]] = {}
    for index, row in enumerate(paired):
        label = f"paired_results row {index + 2}"
        seed = _int(row.get("seed"), f"{label}.seed")
        paired_seed_order.append(seed)
        if not _bool(row.get("pair_valid"), f"{label}.pair_valid"):
            raise CalibrationAnalysisError(f"{label} is invalid")
        if row.get("pair_error") not in ("", None):
            raise CalibrationAnalysisError(f"{label} contains pair_error")
        if row.get("outcome_category") not in PAIR_CATEGORIES:
            raise CalibrationAnalysisError(
                f"{label} has invalid outcome category"
            )
        pair_id = row.get("pair_id")
        if pair_id not in pair_map or pair_id in paired_by_id:
            raise CalibrationAnalysisError(f"{label} pair_id is missing or duplicate")
        episode_pair = pair_map[pair_id]
        if row.get("fingerprint") != episode_pair[0]["episode_fingerprint"]:
            raise CalibrationAnalysisError(f"{label} fingerprint mismatch")
        paired_by_id[pair_id] = row
    if paired_seed_order != seeds:
        raise CalibrationAnalysisError(
            "Paired seed order differs from the registered Calibration order"
        )

    expected_failure_counts: Counter[tuple[str, str]] = Counter()
    for row in episodes:
        reason = (
            "success"
            if _bool(
                row.get("controller_reported_success"),
                "episodes.controller_reported_success",
            )
            else (row.get("failure_reason") or "unknown_failure")
        )
        expected_failure_counts[(row["method_id"], reason)] += 1
    actual_failure_counts: Counter[tuple[str, str]] = Counter()
    for row in failure_counts:
        actual_failure_counts[
            (row.get("method_id", ""), row.get("failure_reason", ""))
        ] += _int(row.get("count"), "failure_counts.count")
    if actual_failure_counts != expected_failure_counts:
        raise CalibrationAnalysisError(
            "failure_counts.csv does not match episodes.csv"
        )

    summary_paired = _require_mapping(summary.get("paired"), "summary.paired")
    if (
        summary_paired.get("valid_pair_count") != 30
        or summary_paired.get("invalid_pair_count") != 0
        or summary_paired.get("program_error_pair_count") != 0
    ):
        raise CalibrationAnalysisError("summary.json reports an incomplete run")
    for method_id in METHOD_IDS:
        method_summary = _require_mapping(
            _require_mapping(summary.get("methods"), "summary.methods").get(method_id),
            f"summary.methods.{method_id}",
        )
        if (
            method_summary.get("requested_episodes") != 30
            or method_summary.get("completed_episodes") != 30
            or method_summary.get("program_errors") != 0
        ):
            raise CalibrationAnalysisError(
                f"summary.json reports incomplete {method_id} episodes"
            )
    methods_metrics = _require_mapping(
        production_metrics.get("methods"),
        "production_metrics.methods",
    )
    for method_id in METHOD_IDS:
        method_metrics = _require_mapping(
            methods_metrics.get(method_id),
            f"production_metrics.methods.{method_id}",
        )
        if (
            method_metrics.get("requested_episode_count") != 30
            or method_metrics.get("valid_episode_count") != 30
            or method_metrics.get("invalid_numeric_episode_count") != 0
        ):
            raise CalibrationAnalysisError(
                f"production_metrics.json reports invalid {method_id} coverage"
            )

    log_text = (output_dir / "run.log").read_text(encoding="utf-8")
    if log_text.count("episode_start") != 60 or log_text.count("episode_end") != 60:
        raise CalibrationAnalysisError("run.log does not contain 60 starts and ends")
    for marker in (" ERROR ", "Traceback", "pair_rejected", "program_error"):
        if marker in log_text:
            raise CalibrationAnalysisError(f"run.log contains {marker.strip()!r}")

    return {
        "manifest": dict(manifest),
        "seeds": seeds,
        "summary": dict(summary),
        "production_metrics": dict(production_metrics),
        "episodes": episodes,
        "paired": paired,
        "failure_counts": failure_counts,
        "protocol_toml": protocol_toml,
        "optional": optional,
        "hashes": {
            "baseline_config_sha256": config_hash,
            "protocol_config_sha256": protocol_hash,
            "calibration_split_sha256": manifest["seed_file_sha256"],
        },
    }


def _optional_number(row: Mapping[str, str], name: str) -> float | None:
    return _float(row.get(name), name, optional=True)


def _required_number(row: Mapping[str, str], name: str) -> float:
    value = _float(row.get(name), name)
    assert value is not None
    return value


def _seed_list(rows: Iterable[Mapping[str, str]]) -> list[int]:
    return [_int(row.get("seed"), "episodes.seed") for row in rows]


def _failure_name(row: Mapping[str, str]) -> str:
    return row.get("failure_reason") or "success"


def _group_summary(
    rows: Sequence[Mapping[str, str]],
    key: Callable[[Mapping[str, str]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    result: dict[str, dict[str, Any]] = {}
    for group in sorted(grouped):
        selected = grouped[group]
        safe = sum(_bool(row.get("safe_task_success"), "safe_task_success") for row in selected)
        placement = sum(
            _bool(row.get("placement_success"), "placement_success")
            for row in selected
        )
        result[group] = {
            "episode_count": len(selected),
            "safe_task_success_count": safe,
            "safe_task_success_rate": _rate(safe, len(selected)),
            "placement_success_count": placement,
            "placement_success_rate": _rate(placement, len(selected)),
            "failure_reason_counts": dict(
                sorted(Counter(_failure_name(row) for row in selected).items())
            ),
            "seeds": _seed_list(selected),
        }
    return result


def _equal_width_groups(
    rows: Sequence[Mapping[str, str]],
    *,
    name: str,
    value: Callable[[Mapping[str, str]], float],
    low: float,
    high: float,
) -> dict[str, Any]:
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise CalibrationAnalysisError(f"Invalid bin range for {name}")
    width = (high - low) / 4.0
    bins: list[list[Mapping[str, str]]] = [[] for _ in range(4)]
    for row in rows:
        current = value(row)
        if current < low - 1e-12 or current > high + 1e-12:
            raise CalibrationAnalysisError(
                f"{name} value {current} is outside [{low}, {high}]"
            )
        index = min(3, max(0, int((current - low) / width)))
        bins[index].append(row)
    groups: dict[str, dict[str, Any]] = {}
    for index, selected in enumerate(bins):
        lower = low + index * width
        upper = low + (index + 1) * width
        label = f"Q{index + 1} [{lower:.9g}, {upper:.9g}{']' if index == 3 else ')'}"
        safe = sum(_bool(row.get("safe_task_success"), "safe_task_success") for row in selected)
        groups[label] = {
            "lower": lower,
            "upper": upper,
            "upper_inclusive": index == 3,
            "episode_count": len(selected),
            "safe_task_success_count": safe,
            "safe_task_success_rate": _rate(safe, len(selected)),
            "failure_reason_counts": dict(
                sorted(Counter(_failure_name(row) for row in selected).items())
            ),
            "seeds": _seed_list(selected),
        }
    return {
        "binning": "four equal-width bins over the protocol range",
        "range": [low, high],
        "groups": groups,
    }


def _empirical_quartile_groups(
    rows: Sequence[Mapping[str, str]],
    *,
    name: str,
    value: Callable[[Mapping[str, str]], float],
) -> dict[str, Any]:
    values = sorted(value(row) for row in rows)
    if len(values) < 4:
        raise CalibrationAnalysisError(f"Not enough values to group {name}")
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    edges = [values[0], *cuts, values[-1]]
    bins: list[list[Mapping[str, str]]] = [[] for _ in range(4)]
    for row in rows:
        current = value(row)
        index = 0
        while index < 3 and current > cuts[index]:
            index += 1
        bins[index].append(row)
    groups: dict[str, dict[str, Any]] = {}
    for index, selected in enumerate(bins):
        label = (
            f"Q{index + 1} [{edges[index]:.9g}, "
            f"{edges[index + 1]:.9g}{']' if index == 3 else ']'}"
        )
        safe = sum(_bool(row.get("safe_task_success"), "safe_task_success") for row in selected)
        groups[label] = {
            "lower": edges[index],
            "upper": edges[index + 1],
            "episode_count": len(selected),
            "safe_task_success_count": safe,
            "safe_task_success_rate": _rate(safe, len(selected)),
            "failure_reason_counts": dict(
                sorted(Counter(_failure_name(row) for row in selected).items())
            ),
            "seeds": _seed_list(selected),
        }
    return {
        "binning": "Calibration-sample inclusive quartiles",
        "cut_points": cuts,
        "groups": groups,
    }


def _stats_for_field(
    rows: Iterable[Mapping[str, str]],
    field: str,
) -> dict[str, Any]:
    return _numeric_summary(_optional_number(row, field) for row in rows)


def _build_analysis(archive: Mapping[str, Any]) -> dict[str, Any]:
    episodes = list(archive["episodes"])
    paired = list(archive["paired"])
    b1 = [row for row in episodes if row["method_id"] == "b1_vision"]
    b0 = [row for row in episodes if row["method_id"] == "b0_oracle"]
    safe_rows = [
        row for row in b1 if _bool(row.get("safe_task_success"), "safe_task_success")
    ]
    failed_rows = [row for row in b1 if row not in safe_rows]

    safe_count = len(safe_rows)
    placement_count = sum(
        _bool(row.get("placement_success"), "placement_success") for row in b1
    )
    first_count = sum(
        _bool(
            row.get("first_attempt_placement_success"),
            "first_attempt_placement_success",
        )
        for row in b1
    )
    collision_rows = [
        row for row in b1 if _int(row.get("collision_count"), "collision_count") > 0
    ]
    unexplained_rows = [
        row
        for row in b1
        if _bool(row.get("unexplained_failure"), "unexplained_failure")
    ]
    controller_success_rows = [
        row
        for row in b1
        if _bool(row.get("controller_reported_success"), "controller_reported_success")
    ]
    ground_truth_success_rows = [
        row
        for row in b1
        if _bool(
            row.get("privileged_ground_truth_success"),
            "privileged_ground_truth_success",
        )
    ]
    false_positive_rows = [
        row for row in b1 if _bool(row.get("false_positive"), "false_positive")
    ]
    false_negative_rows = [
        row for row in b1 if _bool(row.get("false_negative"), "false_negative")
    ]
    safe_times = [_required_number(row, "simulation_time") for row in safe_rows]
    core = {
        "valid_episode_count": len(b1),
        "safe_task_success_count": safe_count,
        "safe_task_success_rate": _rate(safe_count, len(b1)),
        "first_attempt_placement_success_count": first_count,
        "first_attempt_placement_success_rate": _rate(first_count, len(b1)),
        "placement_success_count": placement_count,
        "placement_success_rate": _rate(placement_count, len(b1)),
        "collision_episode_count": len(collision_rows),
        "collision_episode_rate": _rate(len(collision_rows), len(b1)),
        "safe_successful_simulation_time": _numeric_summary(safe_times),
        "unexplained_failure_count": len(unexplained_rows),
        "unexplained_failure_rate": _rate(len(unexplained_rows), len(b1)),
        "controller_reported_success_count": len(controller_success_rows),
        "controller_reported_success_rate": _rate(
            len(controller_success_rows), len(b1)
        ),
        "privileged_ground_truth_success_count": len(ground_truth_success_rows),
        "privileged_ground_truth_success_rate": _rate(
            len(ground_truth_success_rows), len(b1)
        ),
        "false_positive_count": len(false_positive_rows),
        "false_positive_seeds": _seed_list(false_positive_rows),
        "false_negative_count": len(false_negative_rows),
        "false_negative_seeds": _seed_list(false_negative_rows),
        "collision_seeds": _seed_list(collision_rows),
        "unexplained_failure_seeds": _seed_list(unexplained_rows),
    }

    recorded_b1 = _require_mapping(
        _require_mapping(
            archive["production_metrics"].get("methods"),
            "production_metrics.methods",
        ).get("b1_vision"),
        "production_metrics.methods.b1_vision",
    )
    comparisons = {
        "safe_task_success_count": safe_count,
        "first_attempt_placement_success_count": first_count,
        "placement_success_count": placement_count,
        "collision_episode_count": len(collision_rows),
        "unexplained_failure_count": len(unexplained_rows),
    }
    for name, value in comparisons.items():
        if recorded_b1.get(name) != value:
            raise CalibrationAnalysisError(
                f"B1 {name} disagrees with production_metrics.json"
            )

    pair_counts = Counter(row["outcome_category"] for row in paired)
    pair_diagnostics = {
        category: {
            "count": pair_counts[category],
            "seeds": [
                _int(row.get("seed"), "paired_results.seed")
                for row in paired
                if row["outcome_category"] == category
            ],
        }
        for category in PAIR_CATEGORIES
    }
    pair_diagnostics["invalid_pair"] = {"count": 0, "seeds": []}
    pair_diagnostics["program_error"] = {"count": 0, "seeds": []}

    failures: dict[str, dict[str, Any]] = {}
    failure_groups: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in failed_rows:
        failure_groups[_failure_name(row)].append(row)
    for reason in sorted(failure_groups):
        selected = failure_groups[reason]
        failures[reason] = {
            "count": len(selected),
            "rate": _rate(len(selected), len(b1)),
            "final_stage_counts": dict(
                sorted(Counter(row["final_stage"] for row in selected).items())
            ),
            "seeds": _seed_list(selected),
            "pick_region_counts": dict(
                sorted(Counter(row["pick_region"] for row in selected).items())
            ),
            "place_region_counts": dict(
                sorted(Counter(row["place_region"] for row in selected).items())
            ),
            "region_pair_counts": dict(
                sorted(
                    Counter(row["pick_place_region_pair"] for row in selected).items()
                )
            ),
            "same_cross_counts": dict(
                sorted(
                    Counter(
                        "same_region"
                        if _bool(row.get("same_region"), "same_region")
                        else "cross_region"
                        for row in selected
                    ).items()
                )
            ),
            "mass": _stats_for_field(selected, "sampled_mass"),
            "pick_place_distance": _stats_for_field(
                selected, "pick_place_distance"
            ),
            "friction": [
                _numeric_summary(
                    [
                        float(
                            _json_cell(
                                row["sampled_friction"],
                                "sampled_friction",
                            )[index]
                        )
                        for row in selected
                    ]
                )
                for index in range(3)
            ],
        }

    final_stage = _group_summary(b1, lambda row: row["final_stage"])
    stage_duration_fields = sorted(
        {
            field
            for row in b1
            for field in row
            if field.startswith("stage_duration.")
        }
    )
    stage_durations: dict[str, dict[str, Any]] = {}
    for field in stage_duration_fields:
        all_stats = _stats_for_field(b1, field)
        if all_stats["count"] == 0:
            continue
        stage_durations[field.split(".", 1)[1]] = {
            "all_episodes_reaching_stage": all_stats,
            "safe_success_episodes": _stats_for_field(safe_rows, field),
        }

    region_analysis = {
        "pick_region": _group_summary(b1, lambda row: row["pick_region"]),
        "place_region": _group_summary(b1, lambda row: row["place_region"]),
        "region_pair": _group_summary(
            b1, lambda row: row["pick_place_region_pair"]
        ),
        "same_cross": _group_summary(
            b1,
            lambda row: (
                "same_region"
                if _bool(row.get("same_region"), "same_region")
                else "cross_region"
            ),
        ),
    }

    protocol_toml = archive["protocol_toml"]
    physics = protocol_toml["physics"]
    mass_range = [float(value) for value in physics["mass_range"]]
    friction_min = [float(value) for value in physics["friction_min"]]
    friction_max = [float(value) for value in physics["friction_max"]]
    physical_analysis = {
        "mass": _equal_width_groups(
            b1,
            name="mass",
            value=lambda row: _required_number(row, "sampled_mass"),
            low=mass_range[0],
            high=mass_range[1],
        ),
        "friction_sliding": _equal_width_groups(
            b1,
            name="friction_sliding",
            value=lambda row: float(
                _json_cell(row["sampled_friction"], "sampled_friction")[0]
            ),
            low=friction_min[0],
            high=friction_max[0],
        ),
        "friction_torsional": _equal_width_groups(
            b1,
            name="friction_torsional",
            value=lambda row: float(
                _json_cell(row["sampled_friction"], "sampled_friction")[1]
            ),
            low=friction_min[1],
            high=friction_max[1],
        ),
        "friction_rolling": _equal_width_groups(
            b1,
            name="friction_rolling",
            value=lambda row: float(
                _json_cell(row["sampled_friction"], "sampled_friction")[2]
            ),
            low=friction_min[2],
            high=friction_max[2],
        ),
        "pick_place_distance": _empirical_quartile_groups(
            b1,
            name="pick_place_distance",
            value=lambda row: _required_number(row, "pick_place_distance"),
        ),
    }

    outcome_by_seed = {
        _int(row["seed"], "paired_results.seed"): row["outcome_category"]
        for row in paired
    }
    perception_failures = [
        row
        for row in b1
        if row.get("failure_reason")
        in {
            "initial_perception_failed",
            "pregrasp_reacquisition_failed",
            "pregrasp_position_unstable",
            "final_object_not_found",
            "final_visual_place_xy_error",
            "final_visual_place_height_error",
        }
    ]
    oracle_only_b1 = [
        row
        for row in b1
        if outcome_by_seed[_int(row["seed"], "seed")] == "oracle_only_success"
    ]
    perception_analysis = {
        "initial_valid_frame_count_distribution": dict(
            sorted(
                Counter(
                    _int(row.get("initial_valid_frame_count"), "initial_valid_frame_count")
                    for row in b1
                ).items()
            )
        ),
        "pregrasp_valid_frame_count_distribution": dict(
            sorted(
                Counter(
                    _int(
                        row.get("pregrasp_valid_frame_count"),
                        "pregrasp_valid_frame_count",
                    )
                    for row in b1
                ).items()
            )
        ),
        "final_visual_valid_frame_count_distribution": dict(
            sorted(
                Counter(
                    _int(
                        row.get("final_visual_valid_frame_count"),
                        "final_visual_valid_frame_count",
                    )
                    for row in b1
                ).items()
            )
        ),
        "initial_object_position_error": {
            "all_available": _stats_for_field(b1, "object_position_error"),
            "safe_success": _stats_for_field(
                safe_rows, "object_position_error"
            ),
            "failed_available": _stats_for_field(
                failed_rows, "object_position_error"
            ),
            "oracle_only_available": _stats_for_field(
                oracle_only_b1, "object_position_error"
            ),
        },
        "initial_target_position_error": {
            "all_available": _stats_for_field(b1, "target_position_error"),
            "safe_success": _stats_for_field(
                safe_rows, "target_position_error"
            ),
            "failed_available": _stats_for_field(
                failed_rows, "target_position_error"
            ),
            "oracle_only_available": _stats_for_field(
                oracle_only_b1, "target_position_error"
            ),
        },
        "initial_object_confidence": {
            "safe_success": _stats_for_field(
                safe_rows, "initial_object_confidence"
            ),
            "failed_available": _stats_for_field(
                failed_rows, "initial_object_confidence"
            ),
        },
        "initial_target_confidence": {
            "safe_success": _stats_for_field(
                safe_rows, "initial_target_confidence"
            ),
            "failed_available": _stats_for_field(
                failed_rows, "initial_target_confidence"
            ),
        },
        "initial_position_spread": _stats_for_field(
            b1, "initial_position_spread"
        ),
        "initial_object_position_spread": _stats_for_field(
            b1, "initial_object_position_spread"
        ),
        "initial_target_position_spread": _stats_for_field(
            b1, "initial_target_position_spread"
        ),
        "pregrasp_correction_magnitude": {
            "all_available": _stats_for_field(
                b1, "pregrasp_correction_magnitude"
            ),
            "safe_success": _stats_for_field(
                safe_rows, "pregrasp_correction_magnitude"
            ),
            "failed_available": _stats_for_field(
                failed_rows, "pregrasp_correction_magnitude"
            ),
        },
        "pregrasp_position_spread": _stats_for_field(
            b1, "pregrasp_position_spread"
        ),
        "final_visual_xy_error": _stats_for_field(
            b1, "final_visual_xy_error"
        ),
        "final_visual_height_error": _stats_for_field(
            b1, "final_visual_height_error"
        ),
        "final_visual_position_spread": _stats_for_field(
            b1, "key_error.final_visual_position_spread"
        ),
        "perception_failure_count": len(perception_failures),
        "perception_failure_seeds": _seed_list(perception_failures),
        "oracle_only_seed_evidence": [
            {
                "seed": _int(row["seed"], "seed"),
                "failure_reason": row.get("failure_reason"),
                "initial_valid_frames": _int(
                    row.get("initial_valid_frame_count"),
                    "initial_valid_frame_count",
                ),
                "pregrasp_valid_frames": _int(
                    row.get("pregrasp_valid_frame_count"),
                    "pregrasp_valid_frame_count",
                ),
                "object_position_error": _optional_number(
                    row, "object_position_error"
                ),
                "target_position_error": _optional_number(
                    row, "target_position_error"
                ),
            }
            for row in oracle_only_b1
        ],
        "interpretation_boundary": (
            "object_position_error and target_position_error are first-provider-"
            "sample labels, not the robust multi-frame position used by control"
        ),
    }

    grasp_not_confirmed = [
        row for row in b1 if row.get("failure_reason") == "grasp_not_confirmed"
    ]
    transfer_lost = [
        row
        for row in b1
        if row.get("failure_reason") == "grasp_lost_during_transfer"
    ]
    contact_analysis = {
        "grasp_candidate_count": sum(
            _bool(row.get("grasp_candidate"), "grasp_candidate") for row in b1
        ),
        "trial_lift_completed_count": sum(
            _bool(row.get("trial_lift_completed"), "trial_lift_completed")
            for row in b1
        ),
        "grasp_confirmed_count": sum(
            _bool(row.get("grasp_confirmed"), "grasp_confirmed") for row in b1
        ),
        "grasp_lost_count": sum(
            _bool(row.get("grasp_lost"), "grasp_lost") for row in b1
        ),
        "aperture_after_close": {
            "safe_success": _stats_for_field(
                safe_rows, "gripper_aperture_after_close"
            ),
            "grasp_not_confirmed": _stats_for_field(
                grasp_not_confirmed, "gripper_aperture_after_close"
            ),
        },
        "bilateral_contact_duration": {
            "safe_success": _stats_for_field(
                safe_rows, "bilateral_contact_duration"
            ),
            "grasp_not_confirmed": _stats_for_field(
                grasp_not_confirmed, "bilateral_contact_duration"
            ),
        },
        "contact_loss_event_count": {
            "safe_success": _stats_for_field(
                safe_rows, "contact_loss_event_count"
            ),
            "grasp_not_confirmed": _stats_for_field(
                grasp_not_confirmed, "contact_loss_event_count"
            ),
        },
        "grasp_not_confirmed_seeds": _seed_list(grasp_not_confirmed),
        "transfer_loss": [
            {
                "seed": _int(row["seed"], "seed"),
                "mass": _required_number(row, "sampled_mass"),
                "aperture_drop": _optional_number(
                    row, "key_error.aperture_drop"
                ),
                "contact_loss_event_count": _int(
                    row.get("contact_loss_event_count"),
                    "contact_loss_event_count",
                ),
                "collision_count": _int(
                    row.get("collision_count"), "collision_count"
                ),
            }
            for row in transfer_lost
        ],
        "recording_boundary": (
            "The archive does not contain failure-time aperture, confirmation "
            "held steps, per-step bilateral state, or release aperture."
        ),
    }

    b1_config = protocol_toml["b1"]
    motion_specs = (
        ("position_error", float(b1_config["arrival_position_tolerance"])),
        (
            "orientation_error",
            float(b1_config["arrival_orientation_tolerance"]),
        ),
        (
            "joint_speed",
            float(b1_config["settled_joint_velocity_threshold"]),
        ),
    )
    motion_analysis: dict[str, Any] = {
        "failure_reason_counts": {
            reason: sum(row.get("failure_reason") == reason for row in b1)
            for reason in (
                "ik_not_converged",
                "motion_stage_timeout",
                "motion_not_settled",
            )
        }
    }
    for suffix, threshold in motion_specs:
        values: list[float] = []
        for row in b1:
            for field, cell in row.items():
                if not field.startswith("key_error.") or not field.endswith(suffix):
                    continue
                if field in {
                    "key_error.ik_position_error",
                    "key_error.ik_orientation_error",
                    "key_error.final_visual_xy_error",
                    "key_error.final_visual_height_error",
                }:
                    continue
                value = _float(cell, field, optional=True)
                if value is not None:
                    values.append(value)
        motion_analysis[suffix] = {
            "threshold": threshold,
            **_numeric_summary(values),
            "above_nominal_threshold_count": sum(
                current > threshold for current in values
            ),
            "above_hysteresis_threshold_count": sum(
                current > 1.25 * threshold for current in values
            ),
        }

    b0_success_count = sum(
        _bool(row.get("privileged_ground_truth_success"), "b0 ground truth")
        for row in b0
    )
    integrity = {
        "status": "PASS",
        "required_input_files": list(REQUIRED_INPUT_FILES),
        "optional_input_files_present": sorted(archive["optional"]),
        "requested_pair_count": 30,
        "completed_pair_count": 30,
        "b0_episode_count": len(b0),
        "b1_episode_count": len(b1),
        "unique_seed_count": len(set(archive["seeds"])),
        "invalid_pair_count": 0,
        "program_error_count": 0,
        "invalid_numeric_episode_count": 0,
        "fingerprints_recomputed": 60,
        "hashes": dict(archive["hashes"]),
        "manifest_flags": {
            name: archive["manifest"][name]
            for name in (
                "calibration_run",
                "baseline_frozen",
                "automatic_parameter_search",
            )
        },
        "effective_overrides": archive["manifest"]["effective_overrides"],
    }
    return {
        "analysis_schema_version": "1.0.0",
        "analysis_scope": (
            "Read-only descriptive statistics for B1-Vision Round 0; "
            "B0-Oracle is used only for paired diagnosis"
        ),
        "integrity": integrity,
        "run": {
            "git_branch": archive["manifest"].get("git_branch"),
            "git_commit": archive["manifest"].get("git_commit"),
            "git_dirty_at_start": archive["manifest"].get("git_dirty"),
            "submodule_status": archive["manifest"].get("submodule_status"),
            "protocol_id": archive["manifest"].get("protocol_id"),
            "protocol_version": archive["manifest"].get("protocol_version"),
            "metrics_schema_version": archive["manifest"].get(
                "metrics_schema_version"
            ),
            "split_id": archive["manifest"].get("split_id"),
            "split_name": archive["manifest"].get("split_name"),
            "command": archive["manifest"].get("command"),
            "start_time": archive["manifest"].get("start_time"),
            "end_time": archive["manifest"].get("end_time"),
        },
        "b1_core_metrics": core,
        "b0_diagnostic_ground_truth_success_count": b0_success_count,
        "pair_diagnostics": pair_diagnostics,
        "b1_failure_analysis": failures,
        "b1_final_stage_analysis": final_stage,
        "b1_stage_duration_analysis": stage_durations,
        "region_analysis": region_analysis,
        "physical_group_analysis": physical_analysis,
        "perception_analysis": perception_analysis,
        "motion_analysis": motion_analysis,
        "grasp_and_contact_analysis": contact_analysis,
        "preflight_environment": archive["optional"].get(
            "preflight_environment.json"
        ),
        "manual_assessment": archive["optional"].get(
            "manual_assessment.json",
            {
                "decision": {
                    "parameter_change_recommended_now": None,
                    "reason": "Manual evidence review has not been supplied.",
                }
            },
        ),
        "analysis_limits": [
            "No Development or Held-out Test results were read.",
            "No Jacobian singular values, manipulability, jerk, or unrecorded telemetry were inferred.",
            "B0-Oracle is diagnostic and is not treated as a deployable method.",
            "Grouped rates are descriptive for 30 Calibration seeds and are not causal estimates.",
        ],
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "—"
    return str(value)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        result.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return result


def _analysis_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        section: str,
        group: str,
        metric: str,
        *,
        count: Any = None,
        rate: Any = None,
        summary: Mapping[str, Any] | None = None,
        seeds: Sequence[int] = (),
        details: Any = None,
    ) -> None:
        numeric = dict(summary or {})
        rows.append(
            {
                "section": section,
                "group": group,
                "metric": metric,
                "count": count,
                "rate": rate,
                "minimum": numeric.get("minimum"),
                "median": numeric.get("median"),
                "mean": numeric.get("mean"),
                "maximum": numeric.get("maximum"),
                "seeds": json.dumps(list(seeds), separators=(",", ":")),
                "details_json": json.dumps(
                    details,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if details is not None
                else "",
            }
        )

    core = analysis["b1_core_metrics"]
    for metric in (
        "safe_task_success",
        "first_attempt_placement_success",
        "placement_success",
        "collision_episode",
        "unexplained_failure",
        "controller_reported_success",
        "privileged_ground_truth_success",
    ):
        add(
            "core_metrics",
            "b1_vision",
            metric,
            count=core[f"{metric}_count"],
            rate=core[f"{metric}_rate"],
        )
    add(
        "core_metrics",
        "b1_vision",
        "safe_successful_simulation_time",
        count=core["safe_successful_simulation_time"]["count"],
        summary=core["safe_successful_simulation_time"],
    )
    add(
        "diagnostics",
        "b1_vision",
        "false_positive",
        count=core["false_positive_count"],
        seeds=core["false_positive_seeds"],
    )
    add(
        "diagnostics",
        "b1_vision",
        "false_negative",
        count=core["false_negative_count"],
        seeds=core["false_negative_seeds"],
    )

    for category, item in analysis["pair_diagnostics"].items():
        add(
            "pair_diagnostics",
            "b0_oracle_vs_b1_vision",
            category,
            count=item["count"],
            seeds=item["seeds"],
        )
    for reason, item in analysis["b1_failure_analysis"].items():
        add(
            "failure_reason",
            reason,
            "episode_count",
            count=item["count"],
            rate=item["rate"],
            seeds=item["seeds"],
            details=item,
        )
    for stage, item in analysis["b1_final_stage_analysis"].items():
        add(
            "final_stage",
            stage,
            "episode_count",
            count=item["episode_count"],
            rate=item["safe_task_success_rate"],
            seeds=item["seeds"],
            details=item,
        )
    for stage, item in analysis["b1_stage_duration_analysis"].items():
        for population, summary in item.items():
            add(
                "stage_duration",
                stage,
                population,
                count=summary["count"],
                summary=summary,
            )
    for dimension, groups in analysis["region_analysis"].items():
        for group, item in groups.items():
            add(
                f"region_{dimension}",
                group,
                "safe_task_success",
                count=item["safe_task_success_count"],
                rate=item["safe_task_success_rate"],
                seeds=item["seeds"],
                details=item,
            )
    for dimension, section in analysis["physical_group_analysis"].items():
        for group, item in section["groups"].items():
            add(
                f"physical_{dimension}",
                group,
                "safe_task_success",
                count=item["safe_task_success_count"],
                rate=item["safe_task_success_rate"],
                seeds=item["seeds"],
                details=item,
            )
    for name, item in analysis["motion_analysis"].items():
        if name == "failure_reason_counts":
            add(
                "motion",
                "failures",
                name,
                count=sum(item.values()),
                details=item,
            )
        else:
            add(
                "motion",
                name,
                "observed_values",
                count=item["count"],
                summary=item,
                details={
                    "threshold": item["threshold"],
                    "above_nominal_threshold_count": item[
                        "above_nominal_threshold_count"
                    ],
                },
            )
    return rows


def _render_markdown(analysis: Mapping[str, Any]) -> str:
    integrity = analysis["integrity"]
    run = analysis["run"]
    core = analysis["b1_core_metrics"]
    lines = [
        "# B1-Vision Round 0 Calibration 报告",
        "",
        "> 本报告只做 Round 0 记录、完整性核验和诊断；B0-Oracle 仅用于外部视觉状态估计造成的系统级损失归因。",
        "",
        "## 1. 运行与完整性",
        "",
        f"- 分支 / commit：{run['git_branch']} / {run['git_commit']}",
        f"- 协议 / metrics schema：{run['protocol_version']} / {run['metrics_schema_version']}",
        f"- split：{run['split_id']} / {run['split_name']}",
        f"- pair：{integrity['completed_pair_count']}/{integrity['requested_pair_count']}；B0={integrity['b0_episode_count']}，B1={integrity['b1_episode_count']}",
        f"- invalid pair={integrity['invalid_pair_count']}；program error={integrity['program_error_count']}；invalid numeric={integrity['invalid_numeric_episode_count']}",
        f"- baseline config SHA-256：{integrity['hashes']['baseline_config_sha256']}",
        f"- protocol SHA-256：{integrity['hashes']['protocol_config_sha256']}",
        f"- Calibration split SHA-256：{integrity['hashes']['calibration_split_sha256']}",
        f"- flags：calibration_run={_fmt(integrity['manifest_flags']['calibration_run'])}，baseline_frozen={_fmt(integrity['manifest_flags']['baseline_frozen'])}，automatic_parameter_search={_fmt(integrity['manifest_flags']['automatic_parameter_search'])}",
        "",
    ]
    preflight = analysis.get("preflight_environment")
    if isinstance(preflight, Mapping):
        disk = preflight.get("disk_c", {})
        memory = preflight.get("memory", {})
        gpu = preflight.get("gpu", {})
        input_hashes = preflight.get("input_hashes", {})
        lines.extend(
            [
                "### 运行前资源快照",
                "",
                f"- OS：{preflight.get('operating_system', {}).get('caption')} {preflight.get('operating_system', {}).get('version')}",
                f"- Python / MuJoCo / NumPy：{preflight.get('runtime', {}).get('python')} / {preflight.get('runtime', {}).get('mujoco')} / {preflight.get('runtime', {}).get('numpy')}",
                f"- C 盘剩余：{disk.get('free_gib')} GiB",
                f"- 可用内存：{memory.get('free_gib')} GiB / {memory.get('total_gib')} GiB",
                f"- GPU：{gpu.get('name')}，显存 {gpu.get('memory_used_mib')}/{gpu.get('memory_total_mib')} MiB，利用率 {gpu.get('utilization_percent')}%",
                f"- 输出目录运行前状态：{preflight.get('output_directory_state_before_run')}",
            ]
        )
        if input_hashes:
            lines.append(
                f"- split manifest 原始 SHA-256：{input_hashes.get('split_manifest_sha256')}"
            )
        lines.append("")

    lines.extend(
        [
            "## 2. B1 核心生产型指标",
            "",
            *_table(
                ("指标", "数量", "分母", "比例"),
                (
                    (
                        "safe_task_success",
                        core["safe_task_success_count"],
                        core["valid_episode_count"],
                        _percent(core["safe_task_success_rate"]),
                    ),
                    (
                        "first_attempt_placement_success",
                        core["first_attempt_placement_success_count"],
                        core["valid_episode_count"],
                        _percent(core["first_attempt_placement_success_rate"]),
                    ),
                    (
                        "placement_success",
                        core["placement_success_count"],
                        core["valid_episode_count"],
                        _percent(core["placement_success_rate"]),
                    ),
                    (
                        "collision_episode",
                        core["collision_episode_count"],
                        core["valid_episode_count"],
                        _percent(core["collision_episode_rate"]),
                    ),
                    (
                        "unexplained_failure",
                        core["unexplained_failure_count"],
                        core["valid_episode_count"],
                        _percent(core["unexplained_failure_rate"]),
                    ),
                ),
            ),
            "",
            "安全成功仿真周期时间："
            f"count={core['safe_successful_simulation_time']['count']}，"
            f"median={_fmt(core['safe_successful_simulation_time']['median'])} s，"
            f"mean={_fmt(core['safe_successful_simulation_time']['mean'])} s，"
            f"min={_fmt(core['safe_successful_simulation_time']['minimum'])} s，"
            f"max={_fmt(core['safe_successful_simulation_time']['maximum'])} s。",
            "",
            f"Controller success={core['controller_reported_success_count']}/30；privileged GT success={core['privileged_ground_truth_success_count']}/30；false positive={core['false_positive_count']}；false negative={core['false_negative_count']}。",
            "",
            "## 3. B0/B1 成对诊断",
            "",
            *_table(
                ("类别", "数量", "seeds"),
                (
                    (name, item["count"], item["seeds"])
                    for name, item in analysis["pair_diagnostics"].items()
                ),
            ),
            "",
            "- oracle_only_success 表示外部视觉状态估计造成的系统级损失候选。",
            "- both_failed 说明问题不只来自外部视觉。",
            "- vision_only_success 不能解释为 Vision 优于 Oracle；应优先检查接触动力学、估计偏差、确定性和配对逻辑。",
            "",
            "## 4. B1 failure reason 与最终阶段",
            "",
            *_table(
                ("Failure", "数量", "比例", "最终阶段", "seeds"),
                (
                    (
                        reason,
                        item["count"],
                        _percent(item["rate"]),
                        ", ".join(item["final_stage_counts"]),
                        item["seeds"],
                    )
                    for reason, item in analysis["b1_failure_analysis"].items()
                ),
            ),
            "",
            *_table(
                ("最终阶段", "episode", "safe success", "比例"),
                (
                    (
                        stage,
                        item["episode_count"],
                        item["safe_task_success_count"],
                        _percent(item["safe_task_success_rate"]),
                    )
                    for stage, item in analysis["b1_final_stage_analysis"].items()
                ),
            ),
            "",
            "只报告了实际出现的失败；未出现的失败类型不据此虚构。",
            "",
            "## 5. 阶段耗时",
            "",
            *_table(
                (
                    "阶段",
                    "到达数",
                    "all median(s)",
                    "safe count",
                    "safe median(s)",
                    "safe mean(s)",
                ),
                (
                    (
                        stage,
                        item["all_episodes_reaching_stage"]["count"],
                        item["all_episodes_reaching_stage"]["median"],
                        item["safe_success_episodes"]["count"],
                        item["safe_success_episodes"]["median"],
                        item["safe_success_episodes"]["mean"],
                    )
                    for stage, item in analysis[
                        "b1_stage_duration_analysis"
                    ].items()
                ),
            ),
            "",
            "## 6. 区域分组",
            "",
        ]
    )
    for title, key in (
        ("Pick region", "pick_region"),
        ("Place region", "place_region"),
        ("Same / cross", "same_cross"),
        ("Region pair", "region_pair"),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                *_table(
                    ("组", "episode", "safe", "safe rate", "failure counts"),
                    (
                        (
                            group,
                            item["episode_count"],
                            item["safe_task_success_count"],
                            _percent(item["safe_task_success_rate"]),
                            json.dumps(
                                item["failure_reason_counts"],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                        for group, item in analysis["region_analysis"][key].items()
                    ),
                ),
                "",
            ]
        )

    lines.extend(["## 7. 质量、摩擦与距离分组", ""])
    for dimension, section in analysis["physical_group_analysis"].items():
        lines.extend(
            [
                f"### {dimension}",
                "",
                f"分箱：{section['binning']}",
                "",
                *_table(
                    ("分组", "episode", "safe", "safe rate", "failed seeds"),
                    (
                        (
                            group,
                            item["episode_count"],
                            item["safe_task_success_count"],
                            _percent(item["safe_task_success_rate"]),
                            [
                                seed
                                for seed in item["seeds"]
                                if seed
                                in {
                                    failed_seed
                                    for failure in analysis[
                                        "b1_failure_analysis"
                                    ].values()
                                    for failed_seed in failure["seeds"]
                                }
                            ],
                        )
                        for group, item in section["groups"].items()
                    ),
                ),
                "",
            ]
        )

    perception = analysis["perception_analysis"]
    motion = analysis["motion_analysis"]
    contact = analysis["grasp_and_contact_analysis"]
    lines.extend(
        [
            "## 8. 感知分析",
            "",
            f"- 初始有效帧分布：{json.dumps(perception['initial_valid_frame_count_distribution'], sort_keys=True)}。",
            f"- Pregrasp 有效帧分布：{json.dumps(perception['pregrasp_valid_frame_count_distribution'], sort_keys=True)}。",
            f"- 初始物体首帧 3-D 误差（可用样本）median={_fmt(perception['initial_object_position_error']['all_available']['median'])} m，mean={_fmt(perception['initial_object_position_error']['all_available']['mean'])} m，max={_fmt(perception['initial_object_position_error']['all_available']['maximum'])} m。",
            f"- 初始目标首帧 3-D 误差（可用样本）median={_fmt(perception['initial_target_position_error']['all_available']['median'])} m，mean={_fmt(perception['initial_target_position_error']['all_available']['mean'])} m，max={_fmt(perception['initial_target_position_error']['all_available']['maximum'])} m。",
            f"- Pregrasp correction max={_fmt(perception['pregrasp_correction_magnitude']['all_available']['maximum'])} m；spread max={_fmt(perception['pregrasp_position_spread']['maximum'])} m。",
            f"- 最终视觉 XY / height max={_fmt(perception['final_visual_xy_error']['maximum'])} / {_fmt(perception['final_visual_height_error']['maximum'])} m。",
            f"- 感知阶段失败 seeds：{_fmt(perception['perception_failure_seeds'])}。",
            f"- Oracle-only 逐 seed 证据：{json.dumps(perception['oracle_only_seed_evidence'], ensure_ascii=False, separators=(',', ':'))}。",
            f"- 口径边界：{perception['interpretation_boundary']}。",
            "",
            "不能只用平均误差判断感知是否可接受：本轮失败呈全帧不可用和明显阶段集中，必须同时看 paired outcome 与最终阶段。",
            "",
            "## 9. 控制、抓取与接触",
            "",
            f"- 运动失败计数：{json.dumps(motion['failure_reason_counts'], sort_keys=True)}。",
            f"- 到达 position error 最大={_fmt(motion['position_error']['maximum'])} m（阈值 {_fmt(motion['position_error']['threshold'])}）；orientation error 最大={_fmt(motion['orientation_error']['maximum'])} rad（阈值 {_fmt(motion['orientation_error']['threshold'])}）；joint speed 最大={_fmt(motion['joint_speed']['maximum'])}（阈值 {_fmt(motion['joint_speed']['threshold'])}）。",
            f"- grasp candidate / trial lift / confirmed / lost：{contact['grasp_candidate_count']} / {contact['trial_lift_completed_count']} / {contact['grasp_confirmed_count']} / {contact['grasp_lost_count']}。",
            f"- Safe-success aperture after close：{json.dumps(contact['aperture_after_close']['safe_success'], sort_keys=True)}。",
            f"- grasp_not_confirmed aperture after close：{json.dumps(contact['aperture_after_close']['grasp_not_confirmed'], sort_keys=True)}。",
            f"- grasp_not_confirmed bilateral contact duration：{json.dumps(contact['bilateral_contact_duration']['grasp_not_confirmed'], sort_keys=True)}。",
            f"- Transfer loss：{json.dumps(contact['transfer_loss'], sort_keys=True)}。",
            f"- 记录边界：{contact['recording_boundary']}",
            "",
            "## 10. 碰撞、FP/FN 与不可解释失败",
            "",
            f"- Collision seeds：{_fmt(core['collision_seeds'])}",
            f"- False-positive seeds：{_fmt(core['false_positive_seeds'])}",
            f"- False-negative seeds：{_fmt(core['false_negative_seeds'])}",
            f"- Unexplained-failure seeds：{_fmt(core['unexplained_failure_seeds'])}",
            "",
            "## 11. 参数问题、算法问题与下一步",
            "",
        ]
    )
    manual = analysis.get("manual_assessment")
    if isinstance(manual, Mapping):
        decision = manual.get("decision", {})
        lines.extend(
            [
                f"- 当前是否建议改参数：{_fmt(decision.get('parameter_change_recommended_now'))}",
                f"- 推荐的单一问题族：{decision.get('recommended_next_problem_family', '—')}",
                f"- 理由：{decision.get('reason', '—')}",
                f"- Viewer 复核：{_fmt(decision.get('viewer_review_required_before_any_parameter_change'))}；优先 seeds={_fmt(decision.get('viewer_priority_seeds', []))}",
                "",
            ]
        )
        classifications = manual.get("classifications", {})
        if isinstance(classifications, Mapping):
            labels = (
                ("明确参数问题", "explicit_parameter_issues"),
                (
                    "可能参数问题但证据不足",
                    "possible_parameter_issues_with_insufficient_evidence",
                ),
                ("外部视觉问题", "external_visual_state_issues"),
                ("算法或几何问题", "algorithm_or_geometry_issues"),
                ("仿真接触敏感性", "simulation_contact_sensitivity"),
                ("程序或记录问题", "program_or_recording_issues"),
                ("暂无法判断", "temporarily_undetermined"),
            )
            for title, key in labels:
                values = classifications.get(key, [])
                lines.append(f"### {title}")
                lines.append("")
                if not values:
                    lines.append("- 无")
                else:
                    for value in values:
                        lines.append(
                            "- "
                            + (
                                json.dumps(
                                    value,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                if isinstance(value, Mapping)
                                else str(value)
                            )
                        )
                lines.append("")
        do_not_adjust = manual.get("do_not_adjust_from_round_0", [])
        lines.extend(["### 本轮不建议修改", ""])
        if isinstance(do_not_adjust, list):
            for item in do_not_adjust:
                if isinstance(item, Mapping):
                    lines.append(
                        f"- {_fmt(item.get('parameters', []))}：{item.get('reason', '')}"
                    )
        adjustments = manual.get("round_1_parameter_adjustments", [])
        lines.extend(
            [
                "",
                "### 建议调整参数",
                "",
                (
                    "- 无；当前证据不足，不创建或运行 Round 1。"
                    if not adjustments
                    else "- "
                    + json.dumps(
                        adjustments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                "",
                f"- 停止说明：{manual.get('stop_statement', '等待人工审查。')}",
            ]
        )
    lines.extend(
        [
            "",
            "## 12. 能力边界",
            "",
            *[f"- {item}" for item in analysis["analysis_limits"]],
            "",
            "本轮仅完成 Round 0 Calibration 和诊断。",
            "未修改参数。",
            "未冻结 B1。",
            "等待用户审查后再决定是否进入 Round 1。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(output_dir: Path, analysis: Mapping[str, Any]) -> None:
    json_path = output_dir / "calibration_analysis.json"
    csv_path = output_dir / "calibration_analysis.csv"
    report_path = output_dir / "calibration_round_0_report.md"
    json_path.write_text(
        json.dumps(
            analysis,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_rows = _analysis_csv_rows(analysis)
    fieldnames = (
        "section",
        "group",
        "metric",
        "count",
        "rate",
        "minimum",
        "median",
        "mean",
        "maximum",
        "seeds",
        "details_json",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    report_path.write_text(_render_markdown(analysis), encoding="utf-8")
    _ensure_finite_text((json_path, csv_path, report_path))


def analyze_calibration(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir).expanduser().resolve()
    if not path.is_dir():
        raise CalibrationAnalysisError(
            f"Round 0 output directory does not exist: {path}"
        )
    archive = _validate_archive(path)
    analysis = _build_analysis(archive)
    _write_outputs(path, analysis)
    return analysis


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize an existing 30-seed B1 Round 0 archive. "
            "This tool is read-only with respect to run inputs and never runs "
            "controllers, searches parameters, or freezes a baseline."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis = analyze_calibration(args.output_dir)
    except Exception as exc:
        print(f"Calibration analysis error: {exc}", file=sys.stderr)
        return 1
    integrity = analysis["integrity"]
    core = analysis["b1_core_metrics"]
    print(
        "Calibration analysis finished: "
        f"pairs={integrity['completed_pair_count']}/"
        f"{integrity['requested_pair_count']}, "
        f"b1_safe_success={core['safe_task_success_count']}/"
        f"{core['valid_episode_count']}, "
        "parameter_changes=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
