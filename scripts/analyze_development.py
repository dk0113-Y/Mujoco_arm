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
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.manifest import repository_metadata, sha256_file
from benchmark.pairing import EpisodeFingerprint
from benchmark.schemas import write_csv, write_json
from benchmark.seed_io import load_seeds
from evaluation.protocol import load_protocol


METHOD_IDS = ("b0_oracle", "b1_vision")
PAIR_CATEGORIES = (
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
)
RAW_RUN_FILES = (
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
ANALYSIS_OUTPUT_FILES = (
    "development_run_validation.json",
    "development_analysis.json",
    "development_episode_analysis.csv",
    "development_pair_analysis.csv",
    "development_group_analysis.csv",
    "development_60_report.md",
    "b2_evidence_matrix.json",
    "diagnostic_seed_candidates.json",
)
PROTOCOL_PATH = PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml"
DEVELOPMENT_SEEDS_PATH = (
    PROJECT_ROOT / "configs/splits/evaluation_protocol_v1/development_v1.txt"
)
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_v1.toml"
FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT / "configs/baselines/b1_vision_v1_manifest.json"
)
CALIBRATION_REFERENCE_PATH = (
    PROJECT_ROOT / "outputs/calibration/b1_vision_v1/freeze_verification"
)
EXPECTED_PROTOCOL_SHA256 = (
    "7a47be9ddf3851b06c84068ec29030d5bf25ebf60f37057d55371823b07e10bd"
)
EXPECTED_FROZEN_CONFIG_SHA256 = (
    "6808c142ae8805695fc43d5e4743a9529cdbea15008810456184e40e1c4b7ea9"
)
EXPECTED_DEVELOPMENT_SPLIT_SHA256 = (
    "677ecd23f9e689b971fa7340f7d34d674f07dfca19bfa9cd4634598d497b98d6"
)
NONFINITE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:NaN|[-+]?Inf(?:inity)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
PERCEPTION_FAILURE_REASONS = frozenset(
    {
        "initial_perception_failed",
        "pregrasp_reacquisition_failed",
        "pregrasp_position_unstable",
        "final_object_not_found",
        "final_visual_place_xy_error",
        "final_visual_place_height_error",
    }
)
GRASP_FAILURE_REASONS = frozenset(
    {
        "empty_gripper_closure",
        "bilateral_contact_missing",
        "grasp_candidate_failed",
        "trial_lift_failed",
        "grasp_not_confirmed",
    }
)
STAGE_ORDER = (
    "scene_perception",
    "move_to_pregrasp",
    "pregrasp_reacquisition",
    "descend_to_grasp",
    "close_gripper",
    "grasp_candidate_check",
    "trial_lift",
    "grasp_confirmation",
    "transfer",
    "descend_to_place",
    "release",
    "withdraw",
    "final_visual_verification",
    "completed",
)


class DevelopmentAnalysisError(ValueError):
    """Raised when formal Development data are incomplete or inconsistent."""


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentAnalysisError(
                    f"{path.name} contains a non-finite JSON value: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentAnalysisError(f"Cannot parse {path}: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentAnalysisError(f"{label} must be a JSON object")
    return value


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DevelopmentAnalysisError(f"Cannot parse {path}: {exc}") from exc
    if not fields:
        raise DevelopmentAnalysisError(f"{path.name} has no CSV header")
    return fields, rows


def _bool(value: Any, label: str) -> bool:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise DevelopmentAnalysisError(f"{label} must be boolean, got {value!r}")


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise DevelopmentAnalysisError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DevelopmentAnalysisError(f"{label} must be an integer") from exc
    return result


def _float(value: Any, label: str, *, optional: bool = False) -> float | None:
    if value in (None, "") and optional:
        return None
    if isinstance(value, bool):
        raise DevelopmentAnalysisError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DevelopmentAnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise DevelopmentAnalysisError(f"{label} contains NaN or Inf")
    return result


def _json_cell(value: Any, label: str, *, optional: bool = False) -> Any:
    if value in (None, "") and optional:
        return None
    try:
        return json.loads(
            str(value),
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentAnalysisError(
                    f"{label} contains a non-finite value: {token}"
                )
            ),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise DevelopmentAnalysisError(f"{label} is not valid JSON") from exc


def _hashes(directory: Path, names: Iterable[str] = RAW_RUN_FILES) -> dict[str, str]:
    return {name: sha256_file(directory / name) for name in names}


def _ensure_finite_text(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DevelopmentAnalysisError(f"Cannot read {path}: {exc}") from exc
        if NONFINITE_PATTERN.search(text):
            raise DevelopmentAnalysisError(f"{path.name} contains a NaN/Inf token")


def _numeric_summary(values: Iterable[float | None]) -> dict[str, Any]:
    selected = [float(value) for value in values if value is not None]
    return {
        "count": len(selected),
        "median": statistics.median(selected) if selected else None,
        "mean": statistics.fmean(selected) if selected else None,
        "minimum": min(selected) if selected else None,
        "maximum": max(selected) if selected else None,
    }


def wilson_interval(count: int, denominator: int) -> dict[str, float | None]:
    """Return a two-sided Wilson 95% confidence interval."""

    if denominator < 0 or count < 0 or count > denominator:
        raise ValueError("Wilson inputs must satisfy 0 <= count <= denominator")
    if denominator == 0:
        return {"lower": None, "upper": None}
    z = 1.959963984540054
    proportion = count / denominator
    scale = 1.0 + z * z / denominator
    center = (proportion + z * z / (2.0 * denominator)) / scale
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / scale
    )
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def exact_paired_binomial(oracle_only: int, vision_only: int) -> dict[str, Any]:
    """Exact two-sided McNemar/binomial exploratory statistic, without SciPy."""

    if oracle_only < 0 or vision_only < 0:
        raise ValueError("Discordant counts cannot be negative")
    discordant = oracle_only + vision_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(oracle_only, vision_only)
        probability = sum(math.comb(discordant, index) for index in range(tail + 1))
        p_value = min(1.0, 2.0 * probability / (2**discordant))
    return {
        "label": "Development exploratory statistic",
        "oracle_only_success": oracle_only,
        "vision_only_success": vision_only,
        "discordant_pair_count": discordant,
        "exact_two_sided_p_value": p_value,
        "final_significance_claim": False,
    }


def _proportion(count: int, denominator: int) -> dict[str, Any]:
    return {
        "count": count,
        "denominator": denominator,
        "rate": None if denominator == 0 else count / denominator,
        "wilson_95": wilson_interval(count, denominator),
    }


def validate_seed_coverage(actual: Sequence[int], expected: Sequence[int]) -> None:
    if len(actual) != len(set(actual)):
        duplicates = sorted(seed for seed, count in Counter(actual).items() if count > 1)
        raise DevelopmentAnalysisError(f"Duplicate Development seeds: {duplicates}")
    if list(actual) != list(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise DevelopmentAnalysisError(
            f"Development seed coverage/order mismatch; missing={missing}, extra={extra}"
        )


def _fingerprint(row: Mapping[str, str]) -> str:
    return EpisodeFingerprint.from_episode_result(
        {
            "seed": _int(row.get("seed"), "episodes.seed"),
            "sampled_pick_position": _json_cell(
                row.get("sampled_pick_position"), "sampled_pick_position"
            ),
            "sampled_place_position": _json_cell(
                row.get("sampled_place_position"), "sampled_place_position"
            ),
            "pick_region": row.get("pick_region"),
            "place_region": row.get("place_region"),
            "sampled_mass": _float(row.get("sampled_mass"), "sampled_mass"),
            "sampled_friction": _json_cell(
                row.get("sampled_friction"), "sampled_friction"
            ),
        }
    ).digest


def validate_episode_rows(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    expected_seeds: Sequence[int],
    *,
    config_sha256: str,
    code_commit: str,
) -> None:
    if any(
        field.startswith("diagnostic.") or field.startswith("privileged_diagnostic.")
        for field in fields
    ):
        raise DevelopmentAnalysisError("Formal episode schema contains diagnostics")
    if len(rows) != 120:
        raise DevelopmentAnalysisError(
            f"episodes.csv contains {len(rows)} rows, expected 120"
        )
    actual_seed_order: list[int] = []
    for pair_index, expected_seed in enumerate(expected_seeds):
        selected = rows[2 * pair_index : 2 * pair_index + 2]
        if [row.get("method_id") for row in selected] != list(METHOD_IDS):
            raise DevelopmentAnalysisError(
                f"Method order changed for Development seed {expected_seed}"
            )
        seeds = [_int(row.get("seed"), "episodes.seed") for row in selected]
        if seeds != [expected_seed, expected_seed]:
            raise DevelopmentAnalysisError(
                f"Episode seed mismatch at Development pair {pair_index}"
            )
        actual_seed_order.append(expected_seed)
        if len({row.get("pair_id") for row in selected}) != 1:
            raise DevelopmentAnalysisError(f"Pair ID mismatch for seed {expected_seed}")
        if len({row.get("episode_fingerprint") for row in selected}) != 1:
            raise DevelopmentAnalysisError(
                f"Pair fingerprint mismatch for seed {expected_seed}"
            )
        for row in selected:
            label = f"episode seed={expected_seed} method={row.get('method_id')}"
            if not _bool(row.get("pair_valid"), f"{label}.pair_valid"):
                raise DevelopmentAnalysisError(f"{label} belongs to an invalid pair")
            if row.get("program_error") not in (None, ""):
                raise DevelopmentAnalysisError(f"{label} contains a program error")
            if not _bool(row.get("result_fields_complete"), f"{label}.complete"):
                raise DevelopmentAnalysisError(f"{label} has incomplete fields")
            if row.get("split_name") != "development":
                raise DevelopmentAnalysisError(f"{label} has the wrong split")
            if row.get("config_sha256") != config_sha256:
                raise DevelopmentAnalysisError(f"{label} config hash mismatch")
            if row.get("code_commit") != code_commit:
                raise DevelopmentAnalysisError(f"{label} code commit mismatch")
            if _fingerprint(row) != row.get("episode_fingerprint"):
                raise DevelopmentAnalysisError(f"{label} fingerprint mismatch")
            _float(row.get("simulation_time"), f"{label}.simulation_time")
            _float(row.get("sampled_mass"), f"{label}.sampled_mass")
            friction = _json_cell(row.get("sampled_friction"), f"{label}.friction")
            if not isinstance(friction, list) or len(friction) != 3:
                raise DevelopmentAnalysisError(f"{label} friction is invalid")
            for index, value in enumerate(friction):
                _float(value, f"{label}.friction[{index}]")
            _int(row.get("collision_count"), f"{label}.collision_count")
            for name in (
                "controller_reported_success",
                "privileged_ground_truth_success",
                "placement_success",
                "safe_task_success",
                "first_attempt_placement_success",
                "collision_episode",
                "unexplained_failure",
            ):
                _bool(row.get(name), f"{label}.{name}")
    validate_seed_coverage(actual_seed_order, expected_seeds)


def validate_pair_rows(
    rows: Sequence[Mapping[str, str]], expected_seeds: Sequence[int]
) -> None:
    if len(rows) != 60:
        raise DevelopmentAnalysisError(
            f"paired_results.csv contains {len(rows)} rows, expected 60"
        )
    actual: list[int] = []
    for index, row in enumerate(rows):
        seed = _int(row.get("seed"), "paired_results.seed")
        actual.append(seed)
        if not _bool(row.get("pair_valid"), "paired_results.pair_valid"):
            raise DevelopmentAnalysisError(f"Invalid pair for seed {seed}")
        if row.get("pair_error") not in (None, ""):
            raise DevelopmentAnalysisError(f"Pair error for seed {seed}")
        if row.get("outcome_category") not in PAIR_CATEGORIES:
            raise DevelopmentAnalysisError(
                f"Invalid outcome category for seed {seed}: {row.get('outcome_category')}"
            )
        if index >= len(expected_seeds) or seed != expected_seeds[index]:
            raise DevelopmentAnalysisError("Paired seed order mismatch")
    validate_seed_coverage(actual, expected_seeds)


def _failure_name(row: Mapping[str, str]) -> str:
    return (
        "success"
        if _bool(row.get("controller_reported_success"), "controller success")
        else str(row.get("failure_reason") or "unknown_failure")
    )


def _validate_failure_counts(
    episodes: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, str]]
) -> None:
    expected = Counter((row["method_id"], _failure_name(row)) for row in episodes)
    actual: Counter[tuple[str, str]] = Counter()
    for row in rows:
        actual[(row.get("method_id", ""), row.get("failure_reason", ""))] += _int(
            row.get("count"), "failure_counts.count"
        )
    if actual != expected:
        raise DevelopmentAnalysisError("failure_counts.csv is inconsistent")


def _strata_map(
    document: Mapping[str, Any], expected_seeds: Sequence[int], execution_commit: str
) -> dict[int, Mapping[str, Any]]:
    if document.get("split_name") != "development" or document.get("seed_count") != 60:
        raise DevelopmentAnalysisError("Development strata identity/count mismatch")
    inputs = _mapping(document.get("input_files"), "strata.input_files")
    if inputs.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentAnalysisError("Strata protocol hash mismatch")
    if inputs.get("seeds_file_sha256") != EXPECTED_DEVELOPMENT_SPLIT_SHA256:
        raise DevelopmentAnalysisError("Strata split hash mismatch")
    expected_flags = {
        "controller_outcomes_used": False,
        "calibration_outcomes_used": False,
        "development_outcomes_used": False,
        "held_out_data_read": False,
        "renderer_created": False,
        "ik_executed": False,
        "generation_git_dirty": False,
    }
    for name, expected in expected_flags.items():
        if document.get(name) is not expected:
            raise DevelopmentAnalysisError(f"Strata flag {name} is not {expected}")
    if document.get("generation_code_commit") != execution_commit:
        raise DevelopmentAnalysisError(
            "Strata generation commit differs from execution commit"
        )
    values = document.get("strata")
    if not isinstance(values, list):
        raise DevelopmentAnalysisError("strata.strata must be a list")
    seeds = [_int(item.get("seed"), "strata.seed") for item in values if isinstance(item, Mapping)]
    if len(seeds) != len(values):
        raise DevelopmentAnalysisError("Strata contains a non-object record")
    validate_seed_coverage(seeds, expected_seeds)
    return {int(item["seed"]): item for item in values}


def _crosscheck_strata(
    episodes: Sequence[Mapping[str, str]], strata: Mapping[int, Mapping[str, Any]]
) -> None:
    for row in episodes:
        seed = _int(row.get("seed"), "episode.seed")
        item = strata[seed]
        checks = {
            "pick_region": row.get("pick_region"),
            "place_region": row.get("place_region"),
        }
        for name, value in checks.items():
            if item.get(name) != value:
                raise DevelopmentAnalysisError(
                    f"Strata {name} mismatch for seed {seed}"
                )
        scalar_pairs = (
            ("mass", _float(row.get("sampled_mass"), "sampled_mass")),
            (
                "pick_place_distance",
                _float(row.get("pick_place_distance"), "pick_place_distance"),
            ),
        )
        for name, value in scalar_pairs:
            if not math.isclose(float(item[name]), float(value), rel_tol=0.0, abs_tol=1e-12):
                raise DevelopmentAnalysisError(
                    f"Strata {name} mismatch for seed {seed}"
                )
        friction = _json_cell(row.get("sampled_friction"), "sampled_friction")
        if any(
            not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(item["friction"], friction)
        ):
            raise DevelopmentAnalysisError(f"Strata friction mismatch for seed {seed}")


def _load_calibration_reference(path: Path) -> dict[str, Any]:
    if path.resolve() != CALIBRATION_REFERENCE_PATH.resolve():
        raise DevelopmentAnalysisError(
            "Calibration reference must be the B1 freeze-verification archive"
        )
    required = (
        "run_manifest.json",
        "episodes.csv",
        "paired_results.csv",
        "failure_counts.csv",
        "summary.json",
        "production_metrics.json",
        "freeze_comparison.json",
        "freeze_verification_report.md",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise DevelopmentAnalysisError(f"Calibration reference missing: {missing}")
    manifest = _mapping(_strict_json(path / "run_manifest.json"), "Calibration manifest")
    if (
        manifest.get("split_name") != "calibration"
        or manifest.get("completed_pairs") != 30
        or manifest.get("invalid_pairs") != 0
        or manifest.get("unhandled_errors") != 0
    ):
        raise DevelopmentAnalysisError("Calibration reference is not the complete freeze run")
    comparison = _mapping(
        _strict_json(path / "freeze_comparison.json"), "freeze_comparison"
    )
    if comparison.get("status") != "PASS":
        raise DevelopmentAnalysisError("Calibration freeze comparison did not pass")
    return {
        "manifest": dict(manifest),
        "summary": _strict_json(path / "summary.json"),
        "production_metrics": _strict_json(path / "production_metrics.json"),
        "failure_counts": _csv_rows(path / "failure_counts.csv")[1],
        "pairs": _csv_rows(path / "paired_results.csv")[1],
    }


def _validate_archive(
    *,
    run_dir: Path,
    strata_path: Path,
    freeze_manifest_path: Path,
    calibration_reference: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    missing = [name for name in RAW_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise DevelopmentAnalysisError(f"Development run is missing files: {missing}")
    _ensure_finite_text(run_dir / name for name in RAW_RUN_FILES)
    raw_hashes_before = _hashes(run_dir)

    if protocol_path.resolve() != PROTOCOL_PATH.resolve():
        raise DevelopmentAnalysisError("Analyzer accepts only Evaluation Protocol v1")
    if sha256_file(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentAnalysisError("Protocol SHA-256 mismatch")
    protocol = load_protocol(protocol_path, validate_splits=False)
    expected_seeds = load_seeds(DEVELOPMENT_SEEDS_PATH)
    if sha256_file(DEVELOPMENT_SEEDS_PATH) != EXPECTED_DEVELOPMENT_SPLIT_SHA256:
        raise DevelopmentAnalysisError("Registered Development split hash mismatch")

    manifest = _mapping(_strict_json(run_dir / "run_manifest.json"), "run_manifest")
    expected_manifest = {
        "total_requested_pairs": 60,
        "completed_pairs": 60,
        "invalid_pairs": 0,
        "unhandled_errors": 0,
        "pilot": False,
        "protocol_id": "evaluation_protocol",
        "protocol_version": "1.0.1",
        "metrics_schema_version": "1.0.0",
        "split_id": "evaluation_protocol_v1",
        "split_name": "development",
        "calibration_run": False,
        "development_run": True,
        "baseline_frozen": True,
        "automatic_parameter_search": False,
        "held_out_test_run": False,
        "diagnostics_enabled": False,
        "visualization_enabled": False,
        "git_dirty": False,
        "git_status_short": [],
        "effective_overrides": {},
        "methods": list(METHOD_IDS),
        "method_execution_order": list(METHOD_IDS),
        "frozen_baseline_id": "b1_vision_v1",
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise DevelopmentAnalysisError(
                f"run_manifest.{name}={manifest.get(name)!r}, expected {expected!r}"
            )
    if manifest.get("unhandled_error_details") not in (None, []):
        raise DevelopmentAnalysisError("Manifest contains unhandled error details")
    if manifest.get("protocol_config_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentAnalysisError("Manifest protocol hash mismatch")
    if manifest.get("seed_file_sha256") != EXPECTED_DEVELOPMENT_SPLIT_SHA256:
        raise DevelopmentAnalysisError("Manifest Development split hash mismatch")
    if manifest.get("config_sha256") != EXPECTED_FROZEN_CONFIG_SHA256:
        raise DevelopmentAnalysisError("Manifest frozen config hash mismatch")
    execution_commit = str(manifest.get("git_commit") or "")
    if re.fullmatch(r"[0-9a-f]{40}", execution_commit) is None:
        raise DevelopmentAnalysisError("Manifest execution commit is incomplete")

    if freeze_manifest_path.resolve() != FREEZE_MANIFEST_PATH.resolve():
        raise DevelopmentAnalysisError("Analyzer accepts only the registered freeze manifest")
    freeze = _mapping(_strict_json(freeze_manifest_path), "freeze manifest")
    freeze_hash = sha256_file(freeze_manifest_path)
    if manifest.get("freeze_manifest_sha256") != freeze_hash:
        raise DevelopmentAnalysisError("Run/freeze manifest SHA-256 mismatch")
    for name in (
        "verified_behavior_commit",
        "freeze_package_commit",
        "frozen_config_sha256",
    ):
        if manifest.get(name) != freeze.get(name):
            raise DevelopmentAnalysisError(f"Frozen provenance mismatch: {name}")
    if freeze.get("freeze_state") not in {"committed_pending_tag", "tagged", "frozen"}:
        raise DevelopmentAnalysisError("Freeze state is not formal")
    if freeze.get("behavior_frozen") is not True:
        raise DevelopmentAnalysisError("Freeze manifest does not freeze behavior")

    if sha256_file(run_dir / "config_snapshot.toml") != EXPECTED_FROZEN_CONFIG_SHA256:
        raise DevelopmentAnalysisError("Config snapshot differs from frozen config")
    if sha256_file(FROZEN_CONFIG_PATH) != EXPECTED_FROZEN_CONFIG_SHA256:
        raise DevelopmentAnalysisError("Frozen config changed")
    if sha256_file(run_dir / "protocol_snapshot.toml") != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentAnalysisError("Protocol snapshot changed")

    seeds_doc = _mapping(_strict_json(run_dir / "seeds.json"), "seeds.json")
    seeds_value = seeds_doc.get("seeds")
    if not isinstance(seeds_value, list):
        raise DevelopmentAnalysisError("seeds.json.seeds must be a list")
    run_seeds = [_int(seed, "seeds.json.seed") for seed in seeds_value]
    validate_seed_coverage(run_seeds, expected_seeds)
    if seeds_doc.get("seed_count") != 60 or seeds_doc.get("duplicates_present") is not False:
        raise DevelopmentAnalysisError("seeds.json metadata is invalid")

    episode_fields, episodes = _csv_rows(run_dir / "episodes.csv")
    _, pairs = _csv_rows(run_dir / "paired_results.csv")
    _, failure_counts = _csv_rows(run_dir / "failure_counts.csv")
    validate_episode_rows(
        episode_fields,
        episodes,
        expected_seeds,
        config_sha256=EXPECTED_FROZEN_CONFIG_SHA256,
        code_commit=execution_commit,
    )
    validate_pair_rows(pairs, expected_seeds)
    _validate_failure_counts(episodes, failure_counts)

    pair_by_seed = {_int(row["seed"], "pair.seed"): row for row in pairs}
    for index, seed in enumerate(expected_seeds):
        pair = pair_by_seed[seed]
        two = episodes[2 * index : 2 * index + 2]
        if pair.get("fingerprint") != two[0].get("episode_fingerprint"):
            raise DevelopmentAnalysisError(f"Pair archive fingerprint mismatch for seed {seed}")
        expected_category = (
            "both_success"
            if all(_bool(row["privileged_ground_truth_success"], "GT") for row in two)
            else "oracle_only_success"
            if _bool(two[0]["privileged_ground_truth_success"], "oracle GT")
            else "vision_only_success"
            if _bool(two[1]["privileged_ground_truth_success"], "vision GT")
            else "both_failed"
        )
        if pair.get("outcome_category") != expected_category:
            raise DevelopmentAnalysisError(f"Pair category mismatch for seed {seed}")

    summary = _mapping(_strict_json(run_dir / "summary.json"), "summary.json")
    production = _mapping(
        _strict_json(run_dir / "production_metrics.json"), "production_metrics"
    )
    for method_id in METHOD_IDS:
        summary_method = _mapping(summary.get("methods", {}).get(method_id), method_id)
        production_method = _mapping(
            production.get("methods", {}).get(method_id), method_id
        )
        if (
            summary_method.get("requested_episodes") != 60
            or summary_method.get("completed_episodes") != 60
            or summary_method.get("program_errors") != 0
            or production_method.get("requested_episode_count") != 60
            or production_method.get("valid_episode_count") != 60
            or production_method.get("invalid_numeric_episode_count") != 0
        ):
            raise DevelopmentAnalysisError(f"Incomplete metric coverage for {method_id}")
    summary_paired = _mapping(summary.get("paired"), "summary.paired")
    if (
        summary_paired.get("valid_pair_count") != 60
        or summary_paired.get("invalid_pair_count") != 0
        or summary_paired.get("program_error_pair_count") != 0
    ):
        raise DevelopmentAnalysisError("Summary reports invalid/incomplete pairs")

    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    if log_text.count("episode_start") != 120 or log_text.count("episode_end") != 120:
        raise DevelopmentAnalysisError("run.log does not contain 120 starts and ends")
    if any(marker in log_text for marker in (" ERROR ", "Traceback", "pair_rejected", "program_error")):
        raise DevelopmentAnalysisError("run.log contains an error marker")
    diagnostic_artifacts = [
        item.name
        for item in run_dir.iterdir()
        if item.is_file()
        and item.name not in ANALYSIS_OUTPUT_FILES
        and (
            item.name.lower().endswith((".png", ".mp4"))
            or "trace" in item.name.lower()
            or "diagnostic" in item.name.lower()
        )
    ]
    if diagnostic_artifacts:
        raise DevelopmentAnalysisError(
            f"Formal run contains diagnostic artifacts: {diagnostic_artifacts}"
        )

    strata_doc = _mapping(_strict_json(strata_path), "Development strata")
    strata = _strata_map(strata_doc, expected_seeds, execution_commit)
    _crosscheck_strata(episodes, strata)
    calibration = _load_calibration_reference(calibration_reference)
    return {
        "protocol": protocol,
        "manifest": dict(manifest),
        "freeze_manifest": dict(freeze),
        "strata_document": dict(strata_doc),
        "strata": strata,
        "seeds": expected_seeds,
        "episodes": episodes,
        "episode_fields": episode_fields,
        "pairs": pairs,
        "summary": dict(summary),
        "production_metrics": dict(production),
        "calibration": calibration,
        "raw_hashes_before": raw_hashes_before,
    }


def _method_core(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    denominator = len(rows)
    safe = sum(_bool(row["safe_task_success"], "safe_task_success") for row in rows)
    first = sum(
        _bool(row["first_attempt_placement_success"], "first_attempt")
        for row in rows
    )
    collision = sum(_bool(row["collision_episode"], "collision") for row in rows)
    unexplained = sum(
        _bool(row["unexplained_failure"], "unexplained_failure") for row in rows
    )
    safe_times = [
        _float(row["simulation_time"], "simulation_time")
        for row in rows
        if _bool(row["safe_task_success"], "safe_task_success")
    ]
    return {
        "valid_episode_count": denominator,
        "safe_task_success": _proportion(safe, denominator),
        "first_attempt_placement_success": _proportion(first, denominator),
        "collision_episode": _proportion(collision, denominator),
        "safe_successful_simulation_time": _numeric_summary(safe_times),
        "unexplained_failure": _proportion(unexplained, denominator),
        "simulation_time_semantics": (
            "MuJoCo simulation time is a task-flow proxy, not wall-clock latency "
            "or an industrial production cycle time."
        ),
        "confidence_interval_scope": (
            "Wilson 95% intervals describe only this Development set and are not "
            "a final Held-out generalization statement."
        ),
    }


def _pair_analysis(
    pairs: Sequence[Mapping[str, str]], b1_by_seed: Mapping[int, Mapping[str, str]]
) -> dict[str, Any]:
    counts = Counter(str(row["outcome_category"]) for row in pairs)
    categories = {
        category: {
            "count": counts[category],
            "seeds": [
                _int(row["seed"], "pair.seed")
                for row in pairs
                if row["outcome_category"] == category
            ],
        }
        for category in PAIR_CATEGORIES
    }
    categories["invalid_pair"] = {"count": 0, "seeds": []}
    categories["program_error"] = {"count": 0, "seeds": []}
    failure_by_category: dict[str, dict[str, int]] = {}
    for category in PAIR_CATEGORIES:
        failure_by_category[category] = dict(
            sorted(
                Counter(
                    _failure_name(b1_by_seed[_int(row["seed"], "pair.seed")])
                    for row in pairs
                    if row["outcome_category"] == category
                ).items()
            )
        )
    oracle_success = counts["both_success"] + counts["oracle_only_success"]
    vision_success = counts["both_success"] + counts["vision_only_success"]
    statistic = exact_paired_binomial(
        counts["oracle_only_success"], counts["vision_only_success"]
    )
    statistic["b1_minus_b0_paired_success_rate"] = (
        vision_success - oracle_success
    ) / len(pairs)
    return {
        "categories": categories,
        "b1_failure_reason_by_pair_category": failure_by_category,
        "b0_success_count": oracle_success,
        "b1_success_count": vision_success,
        "paired_success_rate_difference_b1_minus_b0": (
            vision_success - oracle_success
        )
        / len(pairs),
        "exploratory_statistic": statistic,
    }


def _failure_and_stage_analysis(
    rows: Sequence[Mapping[str, str]], pair_category: Mapping[int, str]
) -> dict[str, Any]:
    failures = [row for row in rows if _failure_name(row) != "success"]
    total_failures = len(failures)
    by_reason: dict[str, Any] = {}
    for reason in sorted({_failure_name(row) for row in failures}):
        selected = [row for row in failures if _failure_name(row) == reason]
        by_reason[reason] = {
            "count": len(selected),
            "rate_all_b1": len(selected) / len(rows),
            "share_of_all_b1_failures": (
                None if total_failures == 0 else len(selected) / total_failures
            ),
            "seeds": [_int(row["seed"], "seed") for row in selected],
            "final_stage_counts": dict(
                sorted(Counter(row["final_stage"] for row in selected).items())
            ),
            "pair_category_counts": dict(
                sorted(
                    Counter(
                        pair_category[_int(row["seed"], "seed")] for row in selected
                    ).items()
                )
            ),
            "pick_region_counts": dict(
                sorted(Counter(row["pick_region"] for row in selected).items())
            ),
        }
    final_stages: dict[str, Any] = {}
    for stage in sorted({row["final_stage"] for row in rows}):
        selected = [row for row in rows if row["final_stage"] == stage]
        final_stages[stage] = {
            "count": len(selected),
            "rate": len(selected) / len(rows),
            "failure_reason_counts": dict(
                sorted(Counter(_failure_name(row) for row in selected).items())
            ),
            "seeds": [_int(row["seed"], "seed") for row in selected],
        }
    cross = Counter(
        (_failure_name(row), row["final_stage"])
        for row in rows
    )
    stage_fields = sorted(
        field
        for field in {name for row in rows for name in row}
        if field.startswith("stage_duration.")
    )
    durations: dict[str, Any] = {}
    for field in stage_fields:
        available = [
            _float(row.get(field), field, optional=True)
            for row in rows
        ]
        if not any(value is not None for value in available):
            continue
        success_rows = [
            row for row in rows if _bool(row["safe_task_success"], "safe success")
        ]
        failed_rows = [row for row in rows if row not in success_rows]
        durations[field.split(".", 1)[1]] = {
            "all_reaching_stage": _numeric_summary(available),
            "safe_success": _numeric_summary(
                _float(row.get(field), field, optional=True) for row in success_rows
            ),
            "failed": _numeric_summary(
                _float(row.get(field), field, optional=True) for row in failed_rows
            ),
        }
    return {
        "total_b1_failures": total_failures,
        "failure_reasons": by_reason,
        "final_stages": final_stages,
        "failure_reason_by_final_stage": [
            {"failure_reason": reason, "final_stage": stage, "count": count}
            for (reason, stage), count in sorted(cross.items())
        ],
        "stage_durations": durations,
    }


def _passed_funnel_step(row: Mapping[str, str], step: str) -> bool:
    rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    final_rank = rank.get(str(row.get("final_stage")), -1)
    if step == "episode_start":
        return True
    if step == "scene_perception_valid":
        return _int(row.get("initial_valid_frame_count"), "initial valid") >= 3
    if step == "pregrasp_reacquisition_valid":
        return _passed_funnel_step(row, "scene_perception_valid") and _int(
            row.get("pregrasp_valid_frame_count"), "pregrasp valid"
        ) >= 2
    if step == "descend_complete":
        return final_rank > rank["descend_to_grasp"]
    if step == "grasp_candidate":
        return _bool(row.get("grasp_candidate"), "grasp_candidate")
    if step == "trial_lift_complete":
        return _bool(row.get("trial_lift_completed"), "trial_lift_completed")
    if step == "grasp_confirmed":
        return _bool(row.get("grasp_confirmed"), "grasp_confirmed")
    if step == "transfer_complete":
        return final_rank > rank["transfer"]
    if step == "release_complete":
        return _bool(row.get("object_released"), "object_released")
    if step == "final_placement_success":
        return _bool(row.get("placement_success"), "placement_success")
    raise KeyError(step)


def _funnel(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    steps = (
        "episode_start",
        "scene_perception_valid",
        "pregrasp_reacquisition_valid",
        "descend_complete",
        "grasp_candidate",
        "trial_lift_complete",
        "grasp_confirmed",
        "transfer_complete",
        "release_complete",
        "final_placement_success",
    )
    previous = list(rows)
    result: list[dict[str, Any]] = []
    for step in steps:
        passed = [row for row in previous if _passed_funnel_step(row, step)]
        dropped = [row for row in previous if row not in passed]
        reasons = Counter(_failure_name(row) for row in dropped)
        result.append(
            {
                "step": step,
                "entered_count": len(previous),
                "passed_count": len(passed),
                "attrition_count": len(dropped),
                "main_failure_reason": (
                    None
                    if not reasons
                    else sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[0][0]
                ),
                "failure_reason_counts": dict(sorted(reasons.items())),
                "observability": "reconstructed from formal D0 archive fields",
            }
        )
        previous = passed
    return result


def _interpretation_level(n: int) -> str:
    if n < 5:
        return "do_not_interpret"
    if n < 10:
        return "descriptive_signal_only"
    return "preliminary_trend"


def _group_summary(
    seeds: Sequence[int],
    *,
    episodes_by_seed: Mapping[int, Mapping[str, Mapping[str, str]]],
    pair_by_seed: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    b0 = [episodes_by_seed[seed]["b0_oracle"] for seed in seeds]
    b1 = [episodes_by_seed[seed]["b1_vision"] for seed in seeds]
    b0_safe = sum(_bool(row["safe_task_success"], "b0 safe") for row in b0)
    b1_safe = sum(_bool(row["safe_task_success"], "b1 safe") for row in b1)
    return {
        "n": len(seeds),
        "interpretation_level": _interpretation_level(len(seeds)),
        "seeds": list(seeds),
        "b0_safe_success_count": b0_safe,
        "b0_safe_success_rate": None if not seeds else b0_safe / len(seeds),
        "b1_safe_success_count": b1_safe,
        "b1_safe_success_rate": None if not seeds else b1_safe / len(seeds),
        "pair_category_counts": dict(
            sorted(Counter(pair_by_seed[seed]["outcome_category"] for seed in seeds).items())
        ),
        "b1_failure_reason_counts": dict(
            sorted(Counter(_failure_name(row) for row in b1).items())
        ),
        "b0_collision_episode_count": sum(
            _bool(row["collision_episode"], "b0 collision") for row in b0
        ),
        "b1_collision_episode_count": sum(
            _bool(row["collision_episode"], "b1 collision") for row in b1
        ),
        "b0_safe_success_cycle_median": _numeric_summary(
            _float(row["simulation_time"], "b0 time")
            for row in b0
            if _bool(row["safe_task_success"], "b0 safe")
        )["median"],
        "b1_safe_success_cycle_median": _numeric_summary(
            _float(row["simulation_time"], "b1 time")
            for row in b1
            if _bool(row["safe_task_success"], "b1 safe")
        )["median"],
    }


def _group_analyses(
    *,
    seeds: Sequence[int],
    strata: Mapping[int, Mapping[str, Any]],
    episodes_by_seed: Mapping[int, Mapping[str, Mapping[str, str]]],
    pair_by_seed: Mapping[int, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    dimensions: dict[str, Callable[[Mapping[str, Any]], str]] = {
        "pick_region": lambda item: str(item["pick_region"]),
        "place_region": lambda item: str(item["place_region"]),
        "region_pair": lambda item: str(item["region_pair"]),
        "same_cross": lambda item: str(item["same_cross"]),
        "mass": lambda item: str(item["mass_bin"]["name"]),
        "sliding_friction": lambda item: str(item["sliding_friction_bin"]["name"]),
        "torsional_friction": lambda item: str(item["torsional_friction_bin"]["name"]),
        "rolling_friction": lambda item: str(item["rolling_friction_bin"]["name"]),
        "pick_place_distance": lambda item: str(item["pick_place_distance_bin"]["name"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for dimension, key in dimensions.items():
        groups: dict[str, list[int]] = defaultdict(list)
        for seed in seeds:
            groups[key(strata[seed])].append(seed)
        result[dimension] = {
            group: _group_summary(
                selected,
                episodes_by_seed=episodes_by_seed,
                pair_by_seed=pair_by_seed,
            )
            for group, selected in sorted(groups.items())
        }
    return result


def _perception_group(row: Mapping[str, str]) -> str:
    reason = _failure_name(row)
    if reason in PERCEPTION_FAILURE_REASONS:
        return "perception_unavailable_or_rejected"
    if _int(row.get("initial_valid_frame_count"), "initial frames") < 3:
        return "perception_unavailable_or_rejected"
    stage_rank = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    reached_pregrasp_reacquisition = stage_rank.get(str(row.get("final_stage")), -1) >= stage_rank[
        "pregrasp_reacquisition"
    ]
    if reached_pregrasp_reacquisition and _int(
        row.get("pregrasp_valid_frame_count"), "pregrasp frames"
    ) < 2:
        return "perception_unavailable_or_rejected"
    if _bool(row["safe_task_success"], "safe success"):
        return "normal_perception_and_success"
    return "normal_perception_but_later_failure"


def _perception_analysis(
    rows: Sequence[Mapping[str, str]], pair_category: Mapping[int, str]
) -> dict[str, Any]:
    fields = (
        "initial_valid_frame_count",
        "pregrasp_valid_frame_count",
        "final_visual_valid_frame_count",
        "initial_object_confidence",
        "initial_target_confidence",
        "initial_position_spread",
        "initial_object_position_spread",
        "initial_target_position_spread",
        "pregrasp_correction_magnitude",
        "pregrasp_position_spread",
        "object_position_error",
        "target_position_error",
        "final_visual_xy_error",
        "final_visual_height_error",
    )
    grouped: dict[str, Any] = {}
    for group in (
        "normal_perception_and_success",
        "normal_perception_but_later_failure",
        "perception_unavailable_or_rejected",
    ):
        selected = [row for row in rows if _perception_group(row) == group]
        grouped[group] = {
            "count": len(selected),
            "seeds": [_int(row["seed"], "seed") for row in selected],
            "failure_reason_counts": dict(
                sorted(Counter(_failure_name(row) for row in selected).items())
            ),
            "pick_region_counts": dict(
                sorted(Counter(row["pick_region"] for row in selected).items())
            ),
            "pair_category_counts": dict(
                sorted(
                    Counter(
                        pair_category[_int(row["seed"], "seed")] for row in selected
                    ).items()
                )
            ),
            "field_summaries": {
                field: _numeric_summary(
                    _float(row.get(field), field, optional=True) for row in selected
                )
                for field in fields
            },
        }
    return {
        "mutually_exclusive_groups": grouped,
        "perception_failure_reason_counts": dict(
            sorted(
                Counter(
                    _failure_name(row)
                    for row in rows
                    if _failure_name(row) in PERCEPTION_FAILURE_REASONS
                ).items()
            )
        ),
        "oracle_success_vision_perception_failure_seeds": [
            _int(row["seed"], "seed")
            for row in rows
            if _failure_name(row) in PERCEPTION_FAILURE_REASONS
            and pair_category[_int(row["seed"], "seed")] == "oracle_only_success"
        ],
        "both_failed_seeds": [
            _int(row["seed"], "seed")
            for row in rows
            if pair_category[_int(row["seed"], "seed")] == "both_failed"
        ],
        "visual_normal_but_grasp_failed_seeds": [
            _int(row["seed"], "seed")
            for row in rows
            if _perception_group(row) == "normal_perception_but_later_failure"
            and _failure_name(row) in GRASP_FAILURE_REASONS
        ],
        "interpretation_boundary": (
            "Initial object/target position errors are independent first-provider-sample "
            "labels. Group averages are descriptive and are not causal estimates."
        ),
    }


def _contact_analysis(
    b0: Sequence[Mapping[str, str]],
    b1: Sequence[Mapping[str, str]],
    episodes_by_seed: Mapping[int, Mapping[str, Mapping[str, str]]],
) -> dict[str, Any]:
    def counts(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
        return {
            "grasp_candidate": sum(_bool(row["grasp_candidate"], "candidate") for row in rows),
            "trial_lift_completed": sum(_bool(row["trial_lift_completed"], "trial") for row in rows),
            "grasp_confirmed": sum(_bool(row["grasp_confirmed"], "confirmed") for row in rows),
            "grasp_lost": sum(_bool(row["grasp_lost"], "lost") for row in rows),
        }

    b1_grasp = [row for row in b1 if _failure_name(row) in GRASP_FAILURE_REASONS]
    shared = [
        row
        for row in b1_grasp
        if _failure_name(
            episodes_by_seed[_int(row["seed"], "seed")]["b0_oracle"]
        )
        in GRASP_FAILURE_REASONS
    ]
    transfer = [
        row for row in b1 if _failure_name(row) == "grasp_lost_during_transfer"
    ]
    grasp_not_confirmed = [
        row for row in b1 if _failure_name(row) == "grasp_not_confirmed"
    ]
    return {
        "formal_counts": {"b0_oracle": counts(b0), "b1_vision": counts(b1)},
        "grasp_not_confirmed": {
            "b0_count": sum(_failure_name(row) == "grasp_not_confirmed" for row in b0),
            "b1_count": len(grasp_not_confirmed),
            "b1_seeds": [_int(row["seed"], "seed") for row in grasp_not_confirmed],
            "b1_pick_region_counts": dict(
                sorted(Counter(row["pick_region"] for row in grasp_not_confirmed).items())
            ),
            "remains_main_failure": bool(
                grasp_not_confirmed
                and len(grasp_not_confirmed)
                == max(Counter(_failure_name(row) for row in b1 if _failure_name(row) != "success").values())
            ),
        },
        "shared_grasp_failure": {
            "count": len(shared),
            "seeds": [_int(row["seed"], "seed") for row in shared],
        },
        "transfer_drop": {
            "count": len(transfer),
            "seeds": [_int(row["seed"], "seed") for row in transfer],
            "became_new_major_problem": len(transfer) >= max(3, len(b1) // 10),
        },
        "aperture_after_close": {
            "safe_success": _numeric_summary(
                _float(row.get("gripper_aperture_after_close"), "aperture", optional=True)
                for row in b1
                if _bool(row["safe_task_success"], "safe")
            ),
            "grasp_not_confirmed": _numeric_summary(
                _float(row.get("gripper_aperture_after_close"), "aperture", optional=True)
                for row in grasp_not_confirmed
            ),
        },
        "bilateral_contact_sufficiency": (
            "Bilateral binary contact is not sufficient evidence of stable grasp when "
            "a candidate/trial-lift episode still fails confirmation or later drops."
        ),
        "formal_d0_mechanism_boundary": (
            "not observable in formal D0 archive: object orientation, gripper-object "
            "centering, relative slip, contact patch, solver force, and failure-time "
            "quality-gate predicate traces. D0 cannot distinguish centering, approach "
            "orientation, and grasp-quality-gate mechanisms."
        ),
    }


def _calibration_comparison(
    archive: Mapping[str, Any],
    core: Mapping[str, Any],
    pair_analysis: Mapping[str, Any],
    failure_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = archive["calibration"]
    calibration_methods = _mapping(
        calibration["production_metrics"].get("methods"),
        "Calibration production_metrics.methods",
    )
    calibration_b1 = _mapping(calibration_methods.get("b1_vision"), "Calibration B1")
    calibration_b0 = _mapping(calibration_methods.get("b0_oracle"), "Calibration B0")
    calibration_failure_counts = dict(calibration_b1.get("failure_reason_counts", {}))
    development_failure_counts = {
        "success": (
            core["b1_vision"]["valid_episode_count"]
            - failure_analysis["total_b1_failures"]
        ),
        **{
            reason: item["count"]
            for reason, item in failure_analysis["failure_reasons"].items()
        },
    }
    new_families = sorted(
        reason
        for reason, count in development_failure_counts.items()
        if reason != "success" and count and not calibration_failure_counts.get(reason)
    )
    calibration_pairs = calibration["summary"]["paired"]
    return {
        "calibration": {
            "statistical_identity": "debugged Calibration set; not pooled",
            "seed_count": 30,
            "b0_safe_task_success": {
                "count": calibration_b0["safe_task_success_count"],
                "denominator": 30,
                "rate": calibration_b0["safe_task_success_rate"],
            },
            "b1_safe_task_success": {
                "count": calibration_b1["safe_task_success_count"],
                "denominator": 30,
                "rate": calibration_b1["safe_task_success_rate"],
            },
            "b1_failure_reason_counts": calibration_failure_counts,
            "pair_category_counts": {
                category: calibration_pairs[category] for category in PAIR_CATEGORIES
            },
            "b1_safe_success_cycle_time": {
                "count": calibration_b1["safe_successful_simulation_time_count"],
                "median": calibration_b1["safe_successful_simulation_time_median"],
                "mean": calibration_b1["safe_successful_simulation_time_mean"],
                "minimum": calibration_b1["safe_successful_simulation_time_minimum"],
                "maximum": calibration_b1["safe_successful_simulation_time_maximum"],
            },
        },
        "development": {
            "statistical_identity": "independent Development set; not pooled",
            "seed_count": 60,
            "b0_core_metrics": core["b0_oracle"],
            "b1_core_metrics": core["b1_vision"],
            "b1_failure_reason_counts": development_failure_counts,
            "pair_category_counts": {
                category: pair_analysis["categories"][category]["count"]
                for category in PAIR_CATEGORIES
            },
        },
        "new_failure_families_in_development": new_families,
        "all_observed_engineering_summary": {
            "calibration_seed_count": 30,
            "development_seed_count": 60,
            "pooled_success_rate_reported": False,
            "boundary": (
                "Calibration participated in debugging; Development has a distinct "
                "statistical identity. Neither replaces Held-out Test."
            ),
        },
    }


def _family_patterns(
    seeds: Sequence[int],
    strata: Mapping[int, Mapping[str, Any]],
    pair_category: Mapping[int, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pair_counts = dict(sorted(Counter(pair_category[seed] for seed in seeds).items()))
    region = {
        "pick_region_counts": dict(
            sorted(Counter(str(strata[seed]["pick_region"]) for seed in seeds).items())
        ),
        "place_region_counts": dict(
            sorted(Counter(str(strata[seed]["place_region"]) for seed in seeds).items())
        ),
        "region_pair_counts": dict(
            sorted(Counter(str(strata[seed]["region_pair"]) for seed in seeds).items())
        ),
    }
    physical = {}
    for name in (
        "mass_bin",
        "sliding_friction_bin",
        "torsional_friction_bin",
        "rolling_friction_bin",
        "pick_place_distance_bin",
    ):
        physical[name] = dict(
            sorted(Counter(str(strata[seed][name]["name"]) for seed in seeds).items())
        )
    return pair_counts, region, physical


def _evidence_strength(
    *,
    count: int,
    region_count: int,
    pair_consistent: bool,
    online_observable: bool,
    mechanism_confounding: bool,
) -> str:
    if count == 0:
        return "insufficient"
    if (
        count >= 4
        and region_count >= 2
        and pair_consistent
        and online_observable
        and not mechanism_confounding
    ):
        return "strong"
    if count >= 3 and region_count >= 2 and (pair_consistent or online_observable):
        return "moderate"
    return "weak"


def _evidence_matrix(
    *,
    b1: Sequence[Mapping[str, str]],
    episodes_by_seed: Mapping[int, Mapping[str, Mapping[str, str]]],
    strata: Mapping[int, Mapping[str, Any]],
    pair_category: Mapping[int, str],
) -> dict[str, Any]:
    total_failures = sum(_failure_name(row) != "success" for row in b1)
    b1_by_seed = {_int(row["seed"], "seed"): row for row in b1}
    grasp_seeds = sorted(
        seed
        for seed, row in b1_by_seed.items()
        if _failure_name(row) in GRASP_FAILURE_REASONS
    )
    shared_grasp_seeds = sorted(
        seed
        for seed in grasp_seeds
        if _failure_name(episodes_by_seed[seed]["b0_oracle"])
        in GRASP_FAILURE_REASONS
    )
    perception_seeds = sorted(
        seed
        for seed, row in b1_by_seed.items()
        if _failure_name(row) in PERCEPTION_FAILURE_REASONS
    )
    transfer_seeds = sorted(
        seed
        for seed, row in b1_by_seed.items()
        if _failure_name(row) == "grasp_lost_during_transfer"
    )

    definitions = (
        {
            "problem_family": "geometry-aware grasp centering",
            "seeds": grasp_seeds,
            "candidate_mechanism": (
                "Use deployable RGB-D object geometry to choose a centered grasp point."
            ),
            "online": [
                "RGB-D object position",
                "pregrasp correction",
                "gripper aperture",
                "binary bilateral contact",
            ],
            "privileged_only": False,
            "core": "safe_task_success_rate",
            "secondary": [
                "grasp_not_confirmed_rate",
                "trial_lift_completion_rate",
                "transfer_drop_rate",
            ],
            "risk": "Changed grasp points may increase table collision or IK failures.",
            "regressions": ["perception sensitivity", "reachability", "cycle time"],
            "scope": "B2 design only; perception-to-grasp geometry and evaluation",
            "confounders": [
                "approach orientation",
                "contact solver sensitivity",
                "quality-gate thresholds",
            ],
            "pair_consistent": bool(shared_grasp_seeds),
            "mechanism_confounding": True,
            "diagnostic_needed": True,
            "priority": 1,
        },
        {
            "problem_family": "approach orientation robustness",
            "seeds": grasp_seeds,
            "candidate_mechanism": (
                "Select a deployable approach orientation using visible object geometry "
                "and workspace constraints."
            ),
            "online": ["RGB-D geometry", "TCP pose", "IK result"],
            "privileged_only": False,
            "core": "safe_task_success_rate",
            "secondary": ["grasp_candidate_rate", "collision_episode_rate"],
            "risk": "Orientation changes can reduce reachability or increase collision risk.",
            "regressions": ["IK convergence", "table clearance", "cycle time"],
            "scope": "B2 design only; approach-pose policy and safety ablation",
            "confounders": ["centering", "unobserved object yaw/tilt", "contact quality"],
            "pair_consistent": bool(shared_grasp_seeds),
            "mechanism_confounding": True,
            "diagnostic_needed": True,
            "priority": 3,
        },
        {
            "problem_family": "contact/pose-aware grasp-quality gate",
            "seeds": sorted(set(grasp_seeds) | set(transfer_seeds)),
            "candidate_mechanism": (
                "Fuse deployable aperture/contact history with estimated pose quality "
                "before accepting a grasp."
            ),
            "online": [
                "gripper aperture",
                "bilateral contact",
                "trial-lift completion",
                "contact-loss events",
                "RGB-D pose quality",
            ],
            "privileged_only": False,
            "core": "safe_task_success_rate",
            "secondary": [
                "grasp_confirmation_rate",
                "transfer_drop_rate",
                "false_accept/reject diagnostic rates",
            ],
            "risk": "A permissive gate can accept unstable grasps; a strict gate can reject valid grasps.",
            "regressions": ["false rejection", "cycle time", "drop safety"],
            "scope": "B2 design only; online quality estimator/gate and ablation",
            "confounders": ["centering", "orientation", "binary-contact limitations"],
            "pair_consistent": bool(shared_grasp_seeds),
            "mechanism_confounding": True,
            "diagnostic_needed": True,
            "priority": 2,
        },
        {
            "problem_family": "perception availability and reacquisition robustness",
            "seeds": perception_seeds,
            "candidate_mechanism": (
                "Improve deployable RGB-D availability/reacquisition without privileged fallback."
            ),
            "online": [
                "valid frame counts",
                "confidence",
                "position spread",
                "structured rejection reason",
            ],
            "privileged_only": False,
            "core": "safe_task_success_rate",
            "secondary": [
                "initial_perception_failure_rate",
                "pregrasp_reacquisition_failure_rate",
            ],
            "risk": "Looser acceptance can inject bad positions and increase collision/grasp failures.",
            "regressions": ["position error", "collision", "false acceptance"],
            "scope": "B2 design only; availability/reacquisition mechanism and safeguards",
            "confounders": ["occlusion", "color/depth segmentation", "camera-clear geometry"],
            "pair_consistent": any(pair_category[seed] == "oracle_only_success" for seed in perception_seeds),
            "mechanism_confounding": False,
            "diagnostic_needed": True,
            "priority": 1,
        },
        {
            "problem_family": "transfer retention robustness",
            "seeds": transfer_seeds,
            "candidate_mechanism": (
                "Detect degrading grasp retention early and use a safe deployable response."
            ),
            "online": ["contact-loss events", "aperture drop", "TCP motion stage"],
            "privileged_only": False,
            "core": "safe_task_success_rate",
            "secondary": ["transfer_drop_rate", "collision_episode_rate"],
            "risk": "Recovery motion may increase collision exposure or cycle time.",
            "regressions": ["cycle time", "unnecessary aborts", "collision"],
            "scope": "B2 design only if Development shows a repeated transfer family",
            "confounders": ["initial grasp quality", "contact debounce", "dynamics"],
            "pair_consistent": any(pair_category[seed] == "both_failed" for seed in transfer_seeds),
            "mechanism_confounding": True,
            "diagnostic_needed": True,
            "priority": 4,
        },
    )
    records: list[dict[str, Any]] = []
    for definition in definitions:
        affected = list(definition["seeds"])
        pair_evidence, regional, physical = _family_patterns(
            affected, strata, pair_category
        )
        region_count = len(regional["pick_region_counts"])
        strength = _evidence_strength(
            count=len(affected),
            region_count=region_count,
            pair_consistent=bool(definition["pair_consistent"]),
            online_observable=bool(definition["online"]),
            mechanism_confounding=bool(definition["mechanism_confounding"]),
        )
        records.append(
            {
                "problem_family": definition["problem_family"],
                "observed_failure_count": len(affected),
                "affected_seeds": affected,
                "share_of_all_b1_failures": (
                    None if total_failures == 0 else len(affected) / total_failures
                ),
                "pair_category_evidence": pair_evidence,
                "regional_pattern": regional,
                "mass_friction_distance_pattern": physical,
                "formal_online_observability": definition["online"],
                "privileged_evidence_only": definition["privileged_only"],
                "candidate_mechanism": definition["candidate_mechanism"],
                "expected_core_metric": definition["core"],
                "expected_secondary_metrics": definition["secondary"],
                "safety_risk": definition["risk"],
                "possible_regressions": definition["regressions"],
                "implementation_scope": definition["scope"],
                "confounders": definition["confounders"],
                "evidence_strength": strength,
                "diagnostic_needed": definition["diagnostic_needed"],
                "priority": definition["priority"],
            }
        )
    return {
        "evidence_matrix_schema_version": "1.0.0",
        "baseline_id": "b1_vision_v1",
        "split_name": "development",
        "b2_implemented": False,
        "total_b1_failures": total_failures,
        "problem_families": records,
    }


def _metadata_vector(item: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(item["mass"]),
        *[float(value) for value in item["friction"]],
        float(item["pick_place_distance"]),
    )


def _normalized_distance(
    left: Sequence[float], right: Sequence[float], ranges: Sequence[float]
) -> float:
    return math.sqrt(
        sum(((a - b) / scale) ** 2 for a, b, scale in zip(left, right, ranges))
    )


def select_diagnostic_candidates(
    *,
    seeds: Sequence[int],
    strata: Mapping[int, Mapping[str, Any]],
    episodes_by_seed: Mapping[int, Mapping[str, Mapping[str, str]]],
    pair_by_seed: Mapping[int, Mapping[str, str]],
) -> dict[str, Any]:
    """Select 6--12 unique D0.5 candidates by predeclared deterministic rules."""

    b1 = {seed: episodes_by_seed[seed]["b1_vision"] for seed in seeds}
    ranges = (0.15, 0.60, 0.015, 0.0015, 1.5)
    success_seeds = [
        seed for seed in seeds if _bool(b1[seed]["safe_task_success"], "safe")
    ]
    family_specs = (
        (
            "shared_grasp_failure",
            [
                seed
                for seed in seeds
                if _failure_name(b1[seed]) in GRASP_FAILURE_REASONS
                and _failure_name(episodes_by_seed[seed]["b0_oracle"])
                in GRASP_FAILURE_REASONS
            ],
            [
                "object pose/orientation",
                "gripper-object centering",
                "relative slip",
                "contact patch",
                "quality-gate predicate trace",
            ],
        ),
        (
            "oracle_only_perception_failure",
            [
                seed
                for seed in seeds
                if _failure_name(b1[seed]) in PERCEPTION_FAILURE_REASONS
                and pair_by_seed[seed]["outcome_category"] == "oracle_only_success"
            ]
            or [
                seed
                for seed in seeds
                if _failure_name(b1[seed]) in PERCEPTION_FAILURE_REASONS
            ],
            [
                "RGB/depth frame",
                "component mask/rejection reason",
                "occlusion fraction",
                "same-time privileged position label",
            ],
        ),
        (
            "transfer_drop",
            [
                seed
                for seed in seeds
                if _failure_name(b1[seed]) == "grasp_lost_during_transfer"
            ],
            [
                "object pose/orientation",
                "relative slip",
                "contact patch",
                "aperture/contact time series",
            ],
        ),
    )
    selected: dict[int, dict[str, Any]] = {}

    def add(
        seed: int,
        family: str,
        role: str,
        score: float,
        reason: str,
        required: Sequence[str],
    ) -> None:
        if seed in selected:
            if family not in selected[seed]["supported_problem_families"]:
                selected[seed]["supported_problem_families"].append(family)
            selected[seed]["selection_roles"].append(role)
            selected[seed]["selection_reasons"].append(reason)
            selected[seed]["required_diagnostic_fields"] = sorted(
                set(selected[seed]["required_diagnostic_fields"]) | set(required)
            )
            return
        item = strata[seed]
        b0_row = episodes_by_seed[seed]["b0_oracle"]
        b1_row = b1[seed]
        selected[seed] = {
            "seed": seed,
            "problem_family": family,
            "selection_role": role,
            "b0_result": _failure_name(b0_row),
            "b1_result": _failure_name(b1_row),
            "pair_category": pair_by_seed[seed]["outcome_category"],
            "b1_failure_reason": (
                None if _failure_name(b1_row) == "success" else _failure_name(b1_row)
            ),
            "pick_region": item["pick_region"],
            "place_region": item["place_region"],
            "mass": item["mass"],
            "friction": item["friction"],
            "distance": item["pick_place_distance"],
            "selection_score_or_distance": score,
            "selection_reason": reason,
            "required_diagnostic_fields": list(required),
            "supported_problem_families": [family],
            "selection_roles": [role],
            "selection_reasons": [reason],
        }

    for family, eligible, required in family_specs:
        if not eligible:
            continue
        vectors = [_metadata_vector(strata[seed]) for seed in eligible]
        medians = tuple(statistics.median(values) for values in zip(*vectors))
        typical_scores = {
            seed: _normalized_distance(_metadata_vector(strata[seed]), medians, ranges)
            for seed in eligible
        }
        typical = min(eligible, key=lambda seed: (typical_scores[seed], seed))
        add(
            typical,
            family,
            "typical_failure",
            typical_scores[typical],
            "Nearest normalized mass/friction/distance vector to the family medians.",
            required,
        )
        extreme_scores = {
            seed: sum(
                abs(value - midpoint) / scale
                for value, midpoint, scale in zip(
                    _metadata_vector(strata[seed]),
                    (0.125, 1.10, 0.0125, 0.00125, 0.80),
                    ranges,
                )
            )
            for seed in eligible
        }
        extreme = max(eligible, key=lambda seed: (extreme_scores[seed], -seed))
        add(
            extreme,
            family,
            "extreme_failure",
            extreme_scores[extreme],
            "Largest predeclared normalized metadata extremeness score.",
            required,
        )
        discordant_seeds = [
            seed
            for seed in eligible
            if pair_by_seed[seed]["outcome_category"]
            in {"oracle_only_success", "vision_only_success"}
        ]
        if discordant_seeds:
            discordant = min(
                discordant_seeds,
                key=lambda seed: (
                    0
                    if pair_by_seed[seed]["outcome_category"] == "oracle_only_success"
                    else 1,
                    seed,
                ),
            )
            add(
                discordant,
                family,
                "discordant_pair",
                0.0,
                "Deterministic Oracle-only preference, then Vision-only, then seed order.",
                required,
            )
        controls = [
            seed
            for seed in success_seeds
            if strata[seed]["pick_region"] == strata[typical]["pick_region"]
        ]
        if controls:
            control_scores = {
                seed: _normalized_distance(
                    _metadata_vector(strata[seed]),
                    _metadata_vector(strata[typical]),
                    ranges,
                )
                for seed in controls
            }
            control = min(controls, key=lambda seed: (control_scores[seed], seed))
            add(
                control,
                family,
                "matched_success_control",
                control_scores[control],
                "Nearest B1 safe-success seed in the same pick region.",
                required,
            )

    fill_order = sorted(
        seeds,
        key=lambda seed: (
            0 if pair_by_seed[seed]["outcome_category"] == "oracle_only_success" else 1,
            0 if _failure_name(b1[seed]) != "success" else 1,
            seed,
        ),
    )
    for seed in fill_order:
        if len(selected) >= 6:
            break
        if seed in selected:
            continue
        add(
            seed,
            "cross_family_coverage",
            "deterministic_coverage_fill",
            float(seed),
            "Filled the predeclared minimum of six candidates by pair value, failure status, then seed.",
            ["stage-aligned passive trace", "object pose/orientation", "RGB-D frame when applicable"],
        )
    values = list(selected.values())[:12]
    if not 6 <= len(values) <= 12 or len({item["seed"] for item in values}) != len(values):
        raise DevelopmentAnalysisError(
            "Diagnostic selection did not produce 6--12 unique seeds"
        )
    return {
        "diagnostic_candidate_schema_version": "1.0.0",
        "selection_algorithm": (
            "Per problem family: normalized-metadata median exemplar, predeclared "
            "metadata extreme, Oracle-only-first discordant pair when available, and nearest same-pick-"
            "region B1 safe-success control; merge duplicate seeds; deterministic fill "
            "to six by pair value/failure/seed."
        ),
        "candidate_count": len(values),
        "unique_seed_count": len(values),
        "d0_5_episodes_run": 0,
        "candidates": values,
    }


def _stage_conclusion(evidence: Mapping[str, Any]) -> dict[str, Any]:
    families = list(evidence["problem_families"])
    strong = [item for item in families if item["evidence_strength"] == "strong"]
    moderate = [item for item in families if item["evidence_strength"] == "moderate"]
    deployable_strong = [
        item
        for item in strong
        if item["formal_online_observability"]
        and not item["privileged_evidence_only"]
        and not item["diagnostic_needed"]
    ]
    if deployable_strong:
        decision = "D-A"
        rationale = (
            "A repeated cross-region problem family has a deployable online signal and "
            "does not require privileged evidence to choose the mechanism."
        )
    elif strong or moderate:
        decision = "D-B"
        rationale = (
            "A repeated problem family is clear, but formal D0 cannot separate the "
            "candidate mechanisms; passive D0.5 evidence is required."
        )
    else:
        decision = "D-C"
        rationale = (
            "Development failures are too sparse or dispersed to select an interpretable "
            "B2 direction."
        )
    ordered = sorted(
        families,
        key=lambda item: (
            {"strong": 0, "moderate": 1, "weak": 2, "insufficient": 3}[
                item["evidence_strength"]
            ],
            item["priority"],
            -item["observed_failure_count"],
            item["problem_family"],
        ),
    )
    return {
        "decision": decision,
        "rationale": rationale,
        "priority_problem_family": ordered[0]["problem_family"] if ordered else None,
        "d0_5_recommended": decision in {"D-B", "D-C"},
        "development_100_recommended": decision == "D-C",
        "b1_modified": False,
        "protocol_or_split_modified": False,
        "held_out_test_run": False,
        "d0_5_run": False,
        "b2_implemented": False,
    }


def _build_analysis(archive: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    episodes = list(archive["episodes"])
    pairs = list(archive["pairs"])
    seeds = list(archive["seeds"])
    strata = archive["strata"]
    by_method = {
        method: [row for row in episodes if row["method_id"] == method]
        for method in METHOD_IDS
    }
    episodes_by_seed: dict[int, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in episodes:
        episodes_by_seed[_int(row["seed"], "seed")][row["method_id"]] = row
    pair_by_seed = {_int(row["seed"], "pair.seed"): row for row in pairs}
    pair_category = {
        seed: str(pair_by_seed[seed]["outcome_category"]) for seed in seeds
    }
    core = {method: _method_core(by_method[method]) for method in METHOD_IDS}
    paired = _pair_analysis(pairs, {seed: episodes_by_seed[seed]["b1_vision"] for seed in seeds})
    failure = _failure_and_stage_analysis(by_method["b1_vision"], pair_category)
    funnels = {method: _funnel(by_method[method]) for method in METHOD_IDS}
    funnel_difference = [
        {
            "step": b0["step"],
            "b0_passed": b0["passed_count"],
            "b1_passed": b1["passed_count"],
            "b1_minus_b0": b1["passed_count"] - b0["passed_count"],
        }
        for b0, b1 in zip(funnels["b0_oracle"], funnels["b1_vision"])
    ]
    groups = _group_analyses(
        seeds=seeds,
        strata=strata,
        episodes_by_seed=episodes_by_seed,
        pair_by_seed=pair_by_seed,
    )
    perception = _perception_analysis(by_method["b1_vision"], pair_category)
    contact = _contact_analysis(
        by_method["b0_oracle"], by_method["b1_vision"], episodes_by_seed
    )
    calibration = _calibration_comparison(archive, core, paired, failure)
    evidence = _evidence_matrix(
        b1=by_method["b1_vision"],
        episodes_by_seed=episodes_by_seed,
        strata=strata,
        pair_category=pair_category,
    )
    candidates = select_diagnostic_candidates(
        seeds=seeds,
        strata=strata,
        episodes_by_seed=episodes_by_seed,
        pair_by_seed=pair_by_seed,
    )
    conclusion = _stage_conclusion(evidence)
    analysis = {
        "development_analysis_schema_version": "1.0.0",
        "analysis_scope": (
            "Pure read-only Development D0 statistics for B0-Oracle and frozen "
            "B1-Vision v1; no replay, tuning, Held-out access, or B2 implementation."
        ),
        "run": {
            "execution_commit": archive["manifest"]["git_commit"],
            "git_branch": archive["manifest"]["git_branch"],
            "protocol_id": archive["manifest"]["protocol_id"],
            "protocol_version": archive["manifest"]["protocol_version"],
            "metrics_schema_version": archive["manifest"]["metrics_schema_version"],
            "split_id": archive["manifest"]["split_id"],
            "split_name": "development",
            "frozen_baseline_id": "b1_vision_v1",
            "verified_behavior_commit": archive["manifest"]["verified_behavior_commit"],
            "freeze_package_commit": archive["manifest"]["freeze_package_commit"],
        },
        "integrity": {
            "status": "PASS",
            "requested_pairs": 60,
            "completed_pairs": 60,
            "b0_episode_count": 60,
            "b1_episode_count": 60,
            "total_episode_count": 120,
            "unique_seed_count": 60,
            "invalid_pair_count": 0,
            "program_error_count": 0,
            "invalid_numeric_episode_count": 0,
            "diagnostics_enabled": False,
            "visualization_enabled": False,
            "effective_overrides": {},
            "git_clean_at_execution_start": True,
        },
        "core_metrics": core,
        "paired_analysis": paired,
        "b1_failure_and_stage_analysis": failure,
        "state_machine_funnel": {
            **funnels,
            "b0_b1_difference": funnel_difference,
        },
        "group_analysis": groups,
        "perception_chain_analysis": perception,
        "grasp_contact_and_transfer_analysis": contact,
        "calibration_development_comparison": calibration,
        "stage_conclusion": conclusion,
        "evidence_matrix_file": "b2_evidence_matrix.json",
        "diagnostic_candidate_file": "diagnostic_seed_candidates.json",
        "analysis_limits": [
            "Development confidence intervals describe this set only.",
            "Simulation time is a task-flow proxy, not wall-clock latency or industrial cycle time.",
            "not observable in formal D0 archive: centering, object orientation, relative slip, contact patch, solver force, and privileged trace.",
            "Calibration and Development were not pooled into a formal 90-seed rate.",
            "No Held-out Test data were read.",
        ],
    }
    return analysis, evidence, candidates


def _episode_output_rows(
    archive: Mapping[str, Any], analysis: Mapping[str, Any]
) -> list[dict[str, Any]]:
    pair_category = {
        _int(row["seed"], "pair.seed"): row["outcome_category"]
        for row in archive["pairs"]
    }
    strata = archive["strata"]
    rows: list[dict[str, Any]] = []
    for row in archive["episodes"]:
        seed = _int(row["seed"], "seed")
        item = strata[seed]
        rows.append(
            {
                "seed": seed,
                "method_id": row["method_id"],
                "pair_category": pair_category[seed],
                "pick_region": item["pick_region"],
                "place_region": item["place_region"],
                "region_pair": item["region_pair"],
                "same_cross": item["same_cross"],
                "mass_bin": item["mass_bin"]["name"],
                "sliding_friction_bin": item["sliding_friction_bin"]["name"],
                "torsional_friction_bin": item["torsional_friction_bin"]["name"],
                "rolling_friction_bin": item["rolling_friction_bin"]["name"],
                "pick_place_distance_bin": item["pick_place_distance_bin"]["name"],
                "safe_task_success": _bool(row["safe_task_success"], "safe"),
                "first_attempt_placement_success": _bool(
                    row["first_attempt_placement_success"], "first"
                ),
                "collision_episode": _bool(row["collision_episode"], "collision"),
                "unexplained_failure": _bool(
                    row["unexplained_failure"], "unexplained"
                ),
                "failure_reason": None if _failure_name(row) == "success" else _failure_name(row),
                "final_stage": row["final_stage"],
                "simulation_time": _float(row["simulation_time"], "time"),
                "perception_group": (
                    _perception_group(row) if row["method_id"] == "b1_vision" else None
                ),
                "initial_valid_frame_count": _int(
                    row["initial_valid_frame_count"], "initial valid"
                ),
                "pregrasp_valid_frame_count": _int(
                    row["pregrasp_valid_frame_count"], "pregrasp valid"
                ),
                "final_visual_valid_frame_count": _int(
                    row["final_visual_valid_frame_count"], "final valid"
                ),
                "grasp_candidate": _bool(row["grasp_candidate"], "candidate"),
                "trial_lift_completed": _bool(row["trial_lift_completed"], "trial"),
                "grasp_confirmed": _bool(row["grasp_confirmed"], "confirmed"),
                "contact_loss_event_count": _int(
                    row["contact_loss_event_count"], "contact loss"
                ),
                "grasp_lost": _bool(row["grasp_lost"], "grasp lost"),
            }
        )
    return rows


def _pair_output_rows(archive: Mapping[str, Any]) -> list[dict[str, Any]]:
    episodes_by_seed: dict[int, dict[str, Mapping[str, str]]] = defaultdict(dict)
    for row in archive["episodes"]:
        episodes_by_seed[_int(row["seed"], "seed")][row["method_id"]] = row
    result: list[dict[str, Any]] = []
    for pair in archive["pairs"]:
        seed = _int(pair["seed"], "pair.seed")
        item = archive["strata"][seed]
        b0 = episodes_by_seed[seed]["b0_oracle"]
        b1 = episodes_by_seed[seed]["b1_vision"]
        result.append(
            {
                "seed": seed,
                "pair_category": pair["outcome_category"],
                "b0_safe_task_success": _bool(b0["safe_task_success"], "b0 safe"),
                "b1_safe_task_success": _bool(b1["safe_task_success"], "b1 safe"),
                "b0_failure_reason": None if _failure_name(b0) == "success" else _failure_name(b0),
                "b1_failure_reason": None if _failure_name(b1) == "success" else _failure_name(b1),
                "b0_final_stage": b0["final_stage"],
                "b1_final_stage": b1["final_stage"],
                "pick_region": item["pick_region"],
                "place_region": item["place_region"],
                "region_pair": item["region_pair"],
                "mass": item["mass"],
                "friction": item["friction"],
                "pick_place_distance": item["pick_place_distance"],
            }
        )
    return result


def _group_output_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dimension, groups in analysis["group_analysis"].items():
        for group, item in groups.items():
            result.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "n": item["n"],
                    "interpretation_level": item["interpretation_level"],
                    "b0_safe_success_count": item["b0_safe_success_count"],
                    "b0_safe_success_rate": item["b0_safe_success_rate"],
                    "b1_safe_success_count": item["b1_safe_success_count"],
                    "b1_safe_success_rate": item["b1_safe_success_rate"],
                    "pair_category_counts": item["pair_category_counts"],
                    "b1_failure_reason_counts": item["b1_failure_reason_counts"],
                    "b0_collision_episode_count": item["b0_collision_episode_count"],
                    "b1_collision_episode_count": item["b1_collision_episode_count"],
                    "b0_safe_success_cycle_median": item["b0_safe_success_cycle_median"],
                    "b1_safe_success_cycle_median": item["b1_safe_success_cycle_median"],
                    "seeds": item["seeds"],
                }
            )
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "—"
    return str(value)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.1f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    values = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    values.extend("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows)
    return values


def _render_report(
    analysis: Mapping[str, Any],
    evidence: Mapping[str, Any],
    candidates: Mapping[str, Any],
) -> str:
    integrity = analysis["integrity"]
    core = analysis["core_metrics"]
    paired = analysis["paired_analysis"]
    failure = analysis["b1_failure_and_stage_analysis"]
    contact = analysis["grasp_contact_and_transfer_analysis"]
    comparison = analysis["calibration_development_comparison"]
    conclusion = analysis["stage_conclusion"]
    lines = [
        "# Development 60 D0 报告",
        "",
        "> 本报告只分析正式 Development D0；不构成 Held-out 泛化结论，也不实现 B2。",
        "",
        "## 1. 运行身份与完整性",
        "",
        f"- execution commit：`{analysis['run']['execution_commit']}`",
        f"- requested/completed pair：{integrity['requested_pairs']}/{integrity['completed_pairs']}",
        f"- B0/B1/total episode：{integrity['b0_episode_count']}/{integrity['b1_episode_count']}/{integrity['total_episode_count']}",
        f"- invalid pair/program error/invalid numeric：{integrity['invalid_pair_count']}/{integrity['program_error_count']}/{integrity['invalid_numeric_episode_count']}",
        "- development_run=true；baseline_frozen=true；calibration_run=false；automatic_parameter_search=false。",
        "- diagnostics=false；visualization=false；effective overrides={}；Git clean at execution start=true。",
        "",
        "## 2. 五个核心指标",
        "",
    ]
    for method in METHOD_IDS:
        item = core[method]
        lines.extend(
            [
                f"### {method}",
                "",
                *_table(
                    ("指标", "count", "denominator", "rate", "Wilson 95%"),
                    (
                        (
                            name,
                            item[name]["count"],
                            item[name]["denominator"],
                            _pct(item[name]["rate"]),
                            f"[{_fmt(item[name]['wilson_95']['lower'])}, {_fmt(item[name]['wilson_95']['upper'])}]",
                        )
                        for name in (
                            "safe_task_success",
                            "first_attempt_placement_success",
                            "collision_episode",
                            "unexplained_failure",
                        )
                    ),
                ),
                "",
                "安全成功 simulation time："
                f"count={item['safe_successful_simulation_time']['count']}，"
                f"median={_fmt(item['safe_successful_simulation_time']['median'])} s，"
                f"mean={_fmt(item['safe_successful_simulation_time']['mean'])} s，"
                f"min={_fmt(item['safe_successful_simulation_time']['minimum'])} s，"
                f"max={_fmt(item['safe_successful_simulation_time']['maximum'])} s。",
                "",
            ]
        )
    lines.extend(
        [
            "simulation time 是任务流程代理，不是 wall-clock latency，也不是工业生产节拍。Development CI 只描述当前 Development 不确定性。",
            "",
            "## 3. B0/B1 成对分析",
            "",
            *_table(
                ("类别", "count", "seeds"),
                (
                    (
                        category,
                        paired["categories"][category]["count"],
                        paired["categories"][category]["seeds"],
                    )
                    for category in (*PAIR_CATEGORIES, "invalid_pair", "program_error")
                ),
            ),
            "",
            f"Discordant pair={paired['exploratory_statistic']['discordant_pair_count']}；B1-B0 成对成功率差={_pct(paired['paired_success_rate_difference_b1_minus_b0'])}；exact two-sided p={_fmt(paired['exploratory_statistic']['exact_two_sided_p_value'])}。这是 Development exploratory statistic，不是最终显著性结论。",
            "",
            "## 4. B1 failure reason、最终阶段与阶段耗时",
            "",
            *_table(
                ("failure", "count", "rate(all B1)", "share(failures)", "seeds"),
                (
                    (
                        reason,
                        item["count"],
                        _pct(item["rate_all_b1"]),
                        _pct(item["share_of_all_b1_failures"]),
                        item["seeds"],
                    )
                    for reason, item in failure["failure_reasons"].items()
                ),
            ),
            "",
            *_table(
                ("final stage", "count", "rate", "failure reasons"),
                (
                    (stage, item["count"], _pct(item["rate"]), item["failure_reason_counts"])
                    for stage, item in failure["final_stages"].items()
                ),
            ),
            "",
            "## 5. 状态机漏斗",
            "",
            *_table(
                ("step", "B0 passed", "B1 passed", "B1-B0"),
                (
                    (item["step"], item["b0_passed"], item["b1_passed"], item["b1_minus_b0"])
                    for item in analysis["state_machine_funnel"]["b0_b1_difference"]
                ),
            ),
            "",
            "漏斗仅用正式 D0 字段重建；机制级的 centering/orientation/slip/contact patch 不可观测。",
            "",
            "## 6. 区域、质量、摩擦与距离",
            "",
            "所有物理/距离分组严格来自 outcome 前固定的 `development_strata.json`。n<5 不解释；5≤n<10 仅作描述；n≥10 只作初步趋势。九种 region pair 通常每组 6–7，不能作普遍性断言。",
            "",
        ]
    )
    for dimension in (
        "pick_region",
        "place_region",
        "region_pair",
        "same_cross",
        "mass",
        "sliding_friction",
        "torsional_friction",
        "rolling_friction",
        "pick_place_distance",
    ):
        lines.extend(
            [
                f"### {dimension}",
                "",
                *_table(
                    ("group", "n", "B0 safe", "B1 safe", "level", "B1 failures"),
                    (
                        (
                            group,
                            item["n"],
                            f"{item['b0_safe_success_count']}/{item['n']}",
                            f"{item['b1_safe_success_count']}/{item['n']}",
                            item["interpretation_level"],
                            item["b1_failure_reason_counts"],
                        )
                        for group, item in analysis["group_analysis"][dimension].items()
                    ),
                ),
                "",
            ]
        )
    perception = analysis["perception_chain_analysis"]
    lines.extend(
        [
            "## 7. 感知链路",
            "",
            *_table(
                ("互斥组", "count", "failure reasons", "pair categories", "seeds"),
                (
                    (
                        group,
                        item["count"],
                        item["failure_reason_counts"],
                        item["pair_category_counts"],
                        item["seeds"],
                    )
                    for group, item in perception["mutually_exclusive_groups"].items()
                ),
            ),
            "",
            f"Oracle success + Vision perception failure seeds：{_fmt(perception['oracle_success_vision_perception_failure_seeds'])}。",
            f"视觉正常但抓取失败 seeds：{_fmt(perception['visual_normal_but_grasp_failed_seeds'])}。",
            perception["interpretation_boundary"],
            "",
            "## 8. 抓取、接触与搬运",
            "",
            f"grasp_not_confirmed：B0={contact['grasp_not_confirmed']['b0_count']}，B1={contact['grasp_not_confirmed']['b1_count']}；跨 pick region={contact['grasp_not_confirmed']['b1_pick_region_counts']}。",
            f"shared grasp failure：{contact['shared_grasp_failure']['count']}，seeds={contact['shared_grasp_failure']['seeds']}。",
            f"transfer drop：{contact['transfer_drop']['count']}，seeds={contact['transfer_drop']['seeds']}；new major={str(contact['transfer_drop']['became_new_major_problem']).lower()}。",
            contact["bilateral_contact_sufficiency"],
            contact["formal_d0_mechanism_boundary"],
            "",
            "## 9. Calibration 与 Development（保持独立）",
            "",
            f"Calibration B1：{comparison['calibration']['b1_safe_task_success']['count']}/30；Development B1：{core['b1_vision']['safe_task_success']['count']}/60。",
            f"Development 新 failure family：{_fmt(comparison['new_failure_families_in_development'])}。",
            comparison["all_observed_engineering_summary"]["boundary"],
            "",
            "## 10. B2 evidence matrix",
            "",
            *_table(
                ("problem family", "failures", "strength", "priority", "diagnostic"),
                (
                    (
                        item["problem_family"],
                        item["observed_failure_count"],
                        item["evidence_strength"],
                        item["priority"],
                        item["diagnostic_needed"],
                    )
                    for item in evidence["problem_families"]
                ),
            ),
            "",
            "本任务不根据矩阵实现 B2。",
            "",
            "## 11. D0.5 候选（未运行）",
            "",
            f"候选 {candidates['candidate_count']} 个唯一 seed：{[item['seed'] for item in candidates['candidates']]}。",
            candidates["selection_algorithm"],
            "",
            "## 12. D0 阶段结论",
            "",
            f"结论：**{conclusion['decision']}**。{conclusion['rationale']}",
            f"优先问题族：{conclusion['priority_problem_family']}。",
            f"建议执行 D0.5：{str(conclusion['d0_5_recommended']).lower()}；建议考虑 Development 100：{str(conclusion['development_100_recommended']).lower()}。",
            "",
            "未修改 B1。未修改 protocol/split。未运行 D0.5。未运行 Held-out Test。未实现 B2。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_analysis_outputs(
    run_dir: Path,
    archive: Mapping[str, Any],
    analysis: dict[str, Any],
    evidence: Mapping[str, Any],
    candidates: Mapping[str, Any],
) -> None:
    write_json(run_dir / "development_analysis.json", analysis)
    write_json(run_dir / "b2_evidence_matrix.json", evidence)
    write_json(run_dir / "diagnostic_seed_candidates.json", candidates)

    episode_rows = _episode_output_rows(archive, analysis)
    write_csv(
        run_dir / "development_episode_analysis.csv",
        episode_rows,
        tuple(episode_rows[0]),
    )
    pair_rows = _pair_output_rows(archive)
    write_csv(
        run_dir / "development_pair_analysis.csv",
        pair_rows,
        tuple(pair_rows[0]),
    )
    group_rows = _group_output_rows(analysis)
    write_csv(
        run_dir / "development_group_analysis.csv",
        group_rows,
        tuple(group_rows[0]),
    )
    (run_dir / "development_60_report.md").write_text(
        _render_report(analysis, evidence, candidates),
        encoding="utf-8",
        newline="\n",
    )


def analyze_development(
    *,
    run_dir: str | Path,
    strata_path: str | Path,
    freeze_manifest_path: str | Path,
    calibration_reference: str | Path,
    protocol_path: str | Path,
) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise DevelopmentAnalysisError(f"Development run directory does not exist: {run}")
    archive = _validate_archive(
        run_dir=run,
        strata_path=Path(strata_path).expanduser().resolve(),
        freeze_manifest_path=Path(freeze_manifest_path).expanduser().resolve(),
        calibration_reference=Path(calibration_reference).expanduser().resolve(),
        protocol_path=Path(protocol_path).expanduser().resolve(),
    )
    analysis, evidence, candidates = _build_analysis(archive)
    repository = repository_metadata(PROJECT_ROOT)
    analysis["run"]["analysis_commit"] = repository["git_commit"]
    analysis["run"]["analysis_git_dirty"] = repository["git_dirty"]
    _write_analysis_outputs(run, archive, analysis, evidence, candidates)

    raw_hashes_after = _hashes(run)
    if raw_hashes_after != archive["raw_hashes_before"]:
        raise DevelopmentAnalysisError(
            "Formal Development raw files changed during analysis"
        )
    validation = {
        "development_run_validation_schema_version": "1.0.0",
        "status": "PASS",
        "execution_commit": archive["manifest"]["git_commit"],
        "analysis_commit": repository["git_commit"],
        "analysis_only_commit_difference": (
            repository["git_commit"] != archive["manifest"]["git_commit"]
        ),
        "requested_pairs": 60,
        "completed_pairs": 60,
        "b0_episode_count": 60,
        "b1_episode_count": 60,
        "total_episode_count": 120,
        "unique_seed_count": 60,
        "invalid_pair_count": 0,
        "program_error_count": 0,
        "invalid_numeric_episode_count": 0,
        "pair_fingerprint_valid": True,
        "initial_robot_state_fairness_validated_by_runner": True,
        "development_run": True,
        "calibration_run": False,
        "baseline_frozen": True,
        "automatic_parameter_search": False,
        "held_out_test_run": False,
        "diagnostics_enabled": False,
        "visualization_enabled": False,
        "effective_overrides": {},
        "git_clean_at_execution_start": True,
        "formal_schema_has_diagnostic_fields": False,
        "formal_schema_has_privileged_diagnostic_fields": False,
        "registered_development_seed_order_exact": True,
        "calibration_seed_in_formal_run": False,
        "held_out_seed_in_formal_run": False,
        "frozen_config_sha256": EXPECTED_FROZEN_CONFIG_SHA256,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "development_split_sha256": EXPECTED_DEVELOPMENT_SPLIT_SHA256,
        "freeze_manifest_sha256": sha256_file(FREEZE_MANIFEST_PATH),
        "formal_raw_file_hashes_before_analysis": archive["raw_hashes_before"],
        "formal_raw_file_hashes_after_analysis": raw_hashes_after,
        "formal_raw_files_unchanged": True,
        "access_boundary": {
            "controller_executed": False,
            "renderer_created": False,
            "config_modified": False,
            "formal_run_inputs_modified": False,
            "freeze_manifest_modified": False,
            "held_out_data_read": False,
            "b2_implemented": False,
            "d0_5_episode_run": False,
        },
    }
    write_json(run / "development_run_validation.json", validation)
    _ensure_finite_text(run / name for name in ANALYSIS_OUTPUT_FILES)
    if _hashes(run) != archive["raw_hashes_before"]:
        raise DevelopmentAnalysisError(
            "Formal Development raw files changed while finalizing validation"
        )
    return {"validation": validation, "analysis": analysis}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyze a completed Development 60 archive without "
            "running controllers, rendering, tuning, replaying, or reading Held-out data."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--calibration-reference", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = analyze_development(
            run_dir=args.run_dir,
            strata_path=args.strata,
            freeze_manifest_path=args.freeze_manifest,
            calibration_reference=args.calibration_reference,
            protocol_path=args.protocol,
        )
    except Exception as exc:
        print(f"Development analysis error: {exc}", file=sys.stderr)
        return 1
    analysis = result["analysis"]
    print(
        "Development analysis finished: "
        f"pairs={analysis['integrity']['completed_pairs']}/"
        f"{analysis['integrity']['requested_pairs']}, "
        f"b1_safe_success={analysis['core_metrics']['b1_vision']['safe_task_success']['count']}/60, "
        f"decision={analysis['stage_conclusion']['decision']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
