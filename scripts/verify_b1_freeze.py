from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.pairing import EpisodeFingerprint
from benchmark.schemas import write_csv, write_json
from benchmark.seed_io import load_seeds
from environments import load_config
from evaluation.protocol import (
    ProtocolConfig,
    load_protocol,
    validate_baseline_compatibility,
)


COMPARISON_SCHEMA_VERSION = "1.0.0"
# Round 0.5 established this as one MuJoCo step plus floating-point guard.
SIMULATION_TIME_ABSOLUTE_TOLERANCE = 0.0020000001
METHOD_IDS = ("b0_oracle", "b1_vision")
PAIR_CATEGORIES = (
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
)
REQUIRED_ARCHIVE_FILES = (
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
EPISODE_EXACT_FIELDS = (
    "pair_id",
    "external_state_source",
    "episode_fingerprint",
    "sampled_pick_position",
    "sampled_place_position",
    "pick_region",
    "place_region",
    "sampled_mass",
    "sampled_friction",
    "final_stage",
    "failure_reason",
    "controller_reported_success",
    "privileged_ground_truth_success",
    "placement_success",
    "safe_task_success",
    "collision_count",
    "collision_episode",
    "false_positive",
    "false_negative",
    "unexplained_failure",
    "program_error",
)
PAIR_EXACT_FIELDS = (
    "pair_id",
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
    "oracle_collision_count",
    "vision_collision_count",
    "outcome_category",
)


class FreezeVerificationError(ValueError):
    """Raised when a freeze candidate differs from the verified Round 0 behavior."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                FreezeVerificationError(
                    f"{path.name} contains a non-finite JSON value: {value}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FreezeVerificationError(f"Cannot parse {path}: {exc}") from exc


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = list(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FreezeVerificationError(f"Cannot parse {path}: {exc}") from exc
    if not fields:
        raise FreezeVerificationError(f"{path} has no CSV header")
    return fields, rows


def _int_cell(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise FreezeVerificationError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FreezeVerificationError(f"{label} must be an integer") from exc
    return result


def _float_cell(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise FreezeVerificationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FreezeVerificationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise FreezeVerificationError(f"{label} contains NaN or Inf")
    return result


def _bool_cell(value: Any, label: str) -> bool:
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise FreezeVerificationError(f"{label} must be boolean, got {value!r}")


def _program_error(value: Any) -> bool:
    return value not in (None, False, "", 0)


def _episode_fingerprint(row: Mapping[str, str]) -> str:
    def composite(name: str) -> Any:
        try:
            return json.loads(row[name])
        except (KeyError, json.JSONDecodeError) as exc:
            raise FreezeVerificationError(
                f"Invalid composite episode field {name}"
            ) from exc

    return EpisodeFingerprint.from_episode_result(
        {
            "seed": _int_cell(row.get("seed"), "episode.seed"),
            "sampled_pick_position": composite("sampled_pick_position"),
            "sampled_place_position": composite("sampled_place_position"),
            "pick_region": row.get("pick_region"),
            "place_region": row.get("place_region"),
            "sampled_mass": _float_cell(row.get("sampled_mass"), "sampled_mass"),
            "sampled_friction": composite("sampled_friction"),
        }
    ).digest


def _index_episode_rows(
    rows: Sequence[Mapping[str, str]],
    expected_seeds: Sequence[int],
    label: str,
) -> dict[tuple[int, str], Mapping[str, str]]:
    indexed: dict[tuple[int, str], Mapping[str, str]] = {}
    for row in rows:
        seed = _int_cell(row.get("seed"), f"{label}.seed")
        method = str(row.get("method_id") or "")
        if method not in METHOD_IDS:
            raise FreezeVerificationError(f"{label} has unknown method {method!r}")
        key = (seed, method)
        if key in indexed:
            raise FreezeVerificationError(f"{label} contains duplicate episode {key}")
        if not _bool_cell(row.get("pair_valid"), f"{label}.pair_valid"):
            raise FreezeVerificationError(f"{label} contains invalid pair {key}")
        if _program_error(row.get("program_error")):
            raise FreezeVerificationError(f"{label} contains program error {key}")
        if row.get("result_fields_complete") not in ("True", "true", "1"):
            raise FreezeVerificationError(f"{label} has incomplete result fields {key}")
        digest = _episode_fingerprint(row)
        if row.get("episode_fingerprint") != digest:
            raise FreezeVerificationError(f"{label} fingerprint mismatch for {key}")
        indexed[key] = row
    expected = {(seed, method) for seed in expected_seeds for method in METHOD_IDS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise FreezeVerificationError(
            f"{label} seed/method coverage mismatch; missing={missing}, extra={extra}"
        )
    return indexed


def compare_episode_rows(
    reference_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    reference = _index_episode_rows(reference_rows, expected_seeds, "Round 0")
    candidate = _index_episode_rows(
        candidate_rows, expected_seeds, "Freeze Verification"
    )
    comparisons: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for method in METHOD_IDS:
            key = (seed, method)
            left = reference[key]
            right = candidate[key]
            mismatches = [
                field
                for field in EPISODE_EXACT_FIELDS
                if left.get(field) != right.get(field)
            ]
            if mismatches:
                details = ", ".join(
                    f"{field}: {left.get(field)!r} != {right.get(field)!r}"
                    for field in mismatches
                )
                raise FreezeVerificationError(
                    f"Episode behavior mismatch for seed={seed}, method={method}: {details}"
                )
            reference_time = _float_cell(
                left.get("simulation_time"), "Round 0 simulation_time"
            )
            candidate_time = _float_cell(
                right.get("simulation_time"), "Freeze Verification simulation_time"
            )
            difference = abs(reference_time - candidate_time)
            if difference > SIMULATION_TIME_ABSOLUTE_TOLERANCE:
                raise FreezeVerificationError(
                    f"simulation_time mismatch for seed={seed}, method={method}: "
                    f"difference={difference}, allowed={SIMULATION_TIME_ABSOLUTE_TOLERANCE}"
                )
            comparisons.append(
                {
                    "seed": seed,
                    "method_id": method,
                    "pair_id": right["pair_id"],
                    "episode_fingerprint": right["episode_fingerprint"],
                    "final_stage": right["final_stage"],
                    "failure_reason": right["failure_reason"] or None,
                    "controller_reported_success": _bool_cell(
                        right["controller_reported_success"],
                        "controller_reported_success",
                    ),
                    "privileged_ground_truth_success": _bool_cell(
                        right["privileged_ground_truth_success"],
                        "privileged_ground_truth_success",
                    ),
                    "safe_task_success": _bool_cell(
                        right["safe_task_success"], "safe_task_success"
                    ),
                    "collision_count": _int_cell(
                        right["collision_count"], "collision_count"
                    ),
                    "simulation_time_reference": reference_time,
                    "simulation_time_candidate": candidate_time,
                    "simulation_time_absolute_difference": difference,
                    "all_exact_fields_match": True,
                    "simulation_time_within_tolerance": True,
                }
            )
    return comparisons


def _index_pair_rows(
    rows: Sequence[Mapping[str, str]], expected_seeds: Sequence[int], label: str
) -> dict[int, Mapping[str, str]]:
    indexed: dict[int, Mapping[str, str]] = {}
    for row in rows:
        seed = _int_cell(row.get("seed"), f"{label}.seed")
        if seed in indexed:
            raise FreezeVerificationError(f"{label} contains duplicate seed {seed}")
        if not _bool_cell(row.get("pair_valid"), f"{label}.pair_valid"):
            raise FreezeVerificationError(f"{label} contains invalid pair for seed {seed}")
        if row.get("pair_error") not in (None, ""):
            raise FreezeVerificationError(f"{label} contains pair_error for seed {seed}")
        indexed[seed] = row
    if set(indexed) != set(expected_seeds):
        raise FreezeVerificationError(f"{label} pair seed coverage mismatch")
    return indexed


def compare_pair_rows(
    reference_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    expected_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    reference = _index_pair_rows(reference_rows, expected_seeds, "Round 0 pairs")
    candidate = _index_pair_rows(
        candidate_rows, expected_seeds, "Freeze Verification pairs"
    )
    comparisons: list[dict[str, Any]] = []
    for seed in expected_seeds:
        left = reference[seed]
        right = candidate[seed]
        mismatches = [
            field for field in PAIR_EXACT_FIELDS if left.get(field) != right.get(field)
        ]
        if mismatches:
            raise FreezeVerificationError(
                f"Pair behavior mismatch for seed={seed}: {', '.join(mismatches)}"
            )
        for prefix in ("oracle", "vision"):
            field = f"{prefix}_simulation_time"
            difference = abs(
                _float_cell(left.get(field), f"Round 0 {field}")
                - _float_cell(right.get(field), f"Freeze Verification {field}")
            )
            if difference > SIMULATION_TIME_ABSOLUTE_TOLERANCE:
                raise FreezeVerificationError(
                    f"{field} mismatch for seed={seed}: difference={difference}"
                )
        comparisons.append(
            {
                "seed": seed,
                "pair_id": right["pair_id"],
                "fingerprint": right["fingerprint"],
                "outcome_category": right["outcome_category"],
                "all_exact_fields_match": True,
                "simulation_times_within_tolerance": True,
            }
        )
    return comparisons


def compare_nested_metrics(reference: Any, candidate: Any, path: str = "") -> None:
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        if set(reference) != set(candidate):
            raise FreezeVerificationError(f"Metric keys differ at {path or '/'}")
        for key in sorted(reference):
            compare_nested_metrics(
                reference[key], candidate[key], f"{path}/{key}"
            )
        return
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            raise FreezeVerificationError(f"Metric list length differs at {path}")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            compare_nested_metrics(left, right, f"{path}/{index}")
        return
    is_time = "simulation_time" in path
    numeric = (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
    )
    if is_time and numeric:
        if not (
            math.isfinite(float(reference))
            and math.isfinite(float(candidate))
            and abs(float(reference) - float(candidate))
            <= SIMULATION_TIME_ABSOLUTE_TOLERANCE
        ):
            raise FreezeVerificationError(f"Cycle-time metric mismatch at {path}")
    elif reference != candidate:
        raise FreezeVerificationError(
            f"Production metric mismatch at {path}: {reference!r} != {candidate!r}"
        )


def _expected_results(
    summary: Mapping[str, Any], production: Mapping[str, Any]
) -> dict[str, Any]:
    methods = production.get("methods", {})
    b0 = methods.get("b0_oracle", {})
    b1 = methods.get("b1_vision", {})
    summary_methods = summary.get("methods", {})
    paired = summary.get("paired", {})
    expected = {
        "b0_safe_task_success": (b0.get("safe_task_success_count"), 20),
        "b0_collision_episode": (b0.get("collision_episode_count"), 0),
        "b0_unexplained_failure": (b0.get("unexplained_failure_count"), 0),
        "b1_safe_task_success": (b1.get("safe_task_success_count"), 17),
        "b1_first_attempt_placement_success": (
            b1.get("first_attempt_placement_success_count"),
            17,
        ),
        "b1_placement_success": (b1.get("placement_success_count"), 17),
        "b1_collision_episode": (b1.get("collision_episode_count"), 0),
        "b1_unexplained_failure": (b1.get("unexplained_failure_count"), 0),
        "b1_controller_reported_success": (
            summary_methods.get("b1_vision", {}).get(
                "controller_reported_success_count"
            ),
            17,
        ),
        "b1_privileged_ground_truth_success": (
            summary_methods.get("b1_vision", {}).get("ground_truth_success_count"),
            17,
        ),
        "b1_false_positive": (
            summary_methods.get("b1_vision", {}).get("false_positive_count"),
            0,
        ),
        "b1_false_negative": (
            summary_methods.get("b1_vision", {}).get("false_negative_count"),
            0,
        ),
        "both_success": (paired.get("both_success"), 16),
        "oracle_only_success": (paired.get("oracle_only_success"), 4),
        "vision_only_success": (paired.get("vision_only_success"), 1),
        "both_failed": (paired.get("both_failed"), 9),
        "invalid_pair": (paired.get("invalid_pair_count"), 0),
        "program_error": (paired.get("program_error_pair_count"), 0),
    }
    failures = b1.get("failure_reason_counts", {})
    for reason, count in {
        "success": 17,
        "grasp_not_confirmed": 7,
        "initial_perception_failed": 3,
        "pregrasp_reacquisition_failed": 2,
        "grasp_lost_during_transfer": 1,
    }.items():
        expected[f"b1_failure_{reason}"] = (failures.get(reason), count)
    mismatches = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise FreezeVerificationError(
            "Expected B1 freeze metrics were not reproduced: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "b0": {
            "safe_task_success": 20,
            "collision_episode": 0,
            "unexplained_failure": 0,
        },
        "b1": {
            "safe_task_success": 17,
            "first_attempt_placement_success": 17,
            "placement_success": 17,
            "collision_episode": 0,
            "unexplained_failure": 0,
            "controller_reported_success": 17,
            "privileged_ground_truth_success": 17,
            "false_positive": 0,
            "false_negative": 0,
        },
        "pair_categories": {name: int(paired[name]) for name in PAIR_CATEGORIES},
        "b1_failure_reason_counts": dict(failures),
    }


def _load_protocol_and_calibration(
    protocol_path: Path,
) -> tuple[ProtocolConfig, tuple[int, ...]]:
    # Deliberately avoid split validation here: this verifier must not read
    # Development or Held-out Test. Formal split validation is a separate preflight.
    protocol = load_protocol(protocol_path, validate_splits=False)
    seeds = tuple(load_seeds(protocol.splits["calibration"].path))
    if len(seeds) != 30 or len(set(seeds)) != 30:
        raise FreezeVerificationError(
            "Calibration must contain exactly 30 unique seeds"
        )
    return protocol, seeds


def _validate_archive(
    path: Path,
    *,
    expected_seeds: Sequence[int],
    protocol: ProtocolConfig,
    label: str,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARCHIVE_FILES if not (path / name).is_file()]
    if missing:
        raise FreezeVerificationError(f"{label} is missing files: {missing}")
    manifest = _strict_json(path / "run_manifest.json")
    seeds_doc = _strict_json(path / "seeds.json")
    summary = _strict_json(path / "summary.json")
    production = _strict_json(path / "production_metrics.json")
    episode_fields, episodes = _csv_rows(path / "episodes.csv")
    _, pairs = _csv_rows(path / "paired_results.csv")
    _, failure_counts = _csv_rows(path / "failure_counts.csv")

    expected_manifest = {
        "total_requested_pairs": 30,
        "completed_pairs": 30,
        "invalid_pairs": 0,
        "unhandled_errors": 0,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "metrics_schema_version": protocol.metrics_schema_version,
        "split_id": protocol.split_id,
        "split_name": "calibration",
        "calibration_run": True,
        "baseline_frozen": False,
        "automatic_parameter_search": False,
        "pilot": False,
        "effective_overrides": {},
        "git_dirty": False,
        "git_status_short": [],
        "methods": list(METHOD_IDS),
        "method_execution_order": list(METHOD_IDS),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise FreezeVerificationError(
                f"{label} manifest {field}={manifest.get(field)!r}, expected {expected!r}"
            )
    if manifest.get("unhandled_error_details") not in (None, []):
        raise FreezeVerificationError(f"{label} contains unhandled error details")
    if manifest.get("diagnostics_enabled") not in (None, False):
        raise FreezeVerificationError(f"{label} enabled diagnostics")
    if manifest.get("visualization_enabled") not in (None, False):
        raise FreezeVerificationError(f"{label} enabled diagnostic visualization")
    if any(
        field.startswith("diagnostic.") or field.startswith("privileged_diagnostic.")
        for field in episode_fields
    ):
        raise FreezeVerificationError(f"{label} formal schema contains diagnostics")
    diagnostic_artifacts = [
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and (
            "diagnostic" in item.name.lower()
            or "trace" in item.name.lower()
            or item.suffix.lower() in {".png", ".mp4"}
        )
    ]
    if diagnostic_artifacts:
        raise FreezeVerificationError(
            f"{label} contains diagnostic artifacts: {diagnostic_artifacts}"
        )

    seeds = tuple(_int_cell(value, f"{label}.seeds") for value in seeds_doc.get("seeds", []))
    if seeds != tuple(expected_seeds):
        raise FreezeVerificationError(f"{label} Calibration seed order differs")
    if seeds_doc.get("seed_count") != 30 or seeds_doc.get("duplicates_present") is not False:
        raise FreezeVerificationError(f"{label} seed metadata is invalid")
    if len(episodes) != 60 or len(pairs) != 30:
        raise FreezeVerificationError(f"{label} episode/pair counts are incomplete")
    _index_episode_rows(episodes, expected_seeds, label)
    _index_pair_rows(pairs, expected_seeds, label)

    config_hash = _sha256(path / "config_snapshot.toml")
    protocol_hash = _sha256(path / "protocol_snapshot.toml")
    split_hash = _sha256(protocol.splits["calibration"].path)
    for field, actual, expected in (
        ("config_sha256", manifest.get("config_sha256"), config_hash),
        ("protocol_config_sha256", manifest.get("protocol_config_sha256"), protocol_hash),
        ("seed_file_sha256", manifest.get("seed_file_sha256"), split_hash),
    ):
        if actual != expected:
            raise FreezeVerificationError(f"{label} {field} mismatch")
    if protocol_hash != protocol.sha256:
        raise FreezeVerificationError(f"{label} protocol snapshot differs from protocol")
    if production.get("methods", {}).get("b0_oracle", {}).get(
        "invalid_numeric_episode_count"
    ) != 0 or production.get("methods", {}).get("b1_vision", {}).get(
        "invalid_numeric_episode_count"
    ) != 0:
        raise FreezeVerificationError(f"{label} contains invalid numeric episodes")
    expected_counts = Counter(
        (
            row["method_id"],
            "success"
            if _bool_cell(row["controller_reported_success"], "success")
            else (row["failure_reason"] or "unknown_failure"),
        )
        for row in episodes
    )
    actual_counts = Counter(
        {
            (row["method_id"], row["failure_reason"]): _int_cell(
                row["count"], "failure count"
            )
            for row in failure_counts
        }
    )
    if actual_counts != expected_counts:
        raise FreezeVerificationError(f"{label} failure_counts.csv is inconsistent")

    log_text = (path / "run.log").read_text(encoding="utf-8")
    if log_text.count("episode_start") != 60 or log_text.count("episode_end") != 60:
        raise FreezeVerificationError(f"{label} run.log is incomplete")
    if any(marker in log_text for marker in (" ERROR ", "Traceback", "pair_rejected", "program_error")):
        raise FreezeVerificationError(f"{label} run.log contains an error marker")
    return {
        "path": path,
        "manifest": manifest,
        "summary": summary,
        "production_metrics": production,
        "episodes": episodes,
        "pairs": pairs,
        "failure_counts": failure_counts,
        "hashes": {name: _sha256(path / name) for name in REQUIRED_ARCHIVE_FILES},
        "config_sha256": config_hash,
        "protocol_sha256": protocol_hash,
        "split_sha256": split_hash,
        "diagnostics_disabled": True,
    }


def _failure_count_map(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    return {
        f"{row['method_id']}:{row['failure_reason']}": _int_cell(
            row["count"], "failure_counts.count"
        )
        for row in rows
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    metrics = result["verified_metrics"]
    cycle = result["cycle_time"]
    lines = [
        "# B1-Vision v1 Freeze Verification",
        "",
        f"Status: **{result['status']}**",
        "",
        "## Integrity",
        "",
        f"- Requested/completed pairs: {result['completion']['requested_pairs']}/{result['completion']['completed_pairs']}",
        f"- Episodes: B0={result['completion']['b0_episode_count']}, B1={result['completion']['b1_episode_count']}",
        "- Invalid pairs / program errors / invalid numeric: 0 / 0 / 0",
        "- Diagnostics and diagnostic visualization: disabled",
        "- Automatic parameter search: false",
        "",
        "## Round 0 comparison",
        "",
        f"- Episode comparisons: {result['comparison']['episode_comparison_count']}/60 passed",
        f"- Pair comparisons: {result['comparison']['pair_comparison_count']}/30 passed",
        f"- Exact behavior mismatches: {result['comparison']['exact_mismatch_count']}",
        f"- Maximum simulation-time difference: {result['comparison']['maximum_simulation_time_absolute_difference']} s",
        f"- Allowed simulation-time absolute tolerance: {result['simulation_time_rule']['absolute_tolerance_seconds']} s",
        "- Production metrics, failure counts, final-stage counts, pair categories, and cycle-time summaries match.",
        "",
        "## Reproduced metrics",
        "",
        f"- B0 safe success: {metrics['b0']['safe_task_success']}/30",
        f"- B1 safe/first-attempt/placement success: {metrics['b1']['safe_task_success']}/{metrics['b1']['first_attempt_placement_success']}/{metrics['b1']['placement_success']} of 30",
        f"- B1 collision/unexplained/FP/FN: {metrics['b1']['collision_episode']}/{metrics['b1']['unexplained_failure']}/{metrics['b1']['false_positive']}/{metrics['b1']['false_negative']}",
        f"- Pair categories: {json.dumps(metrics['pair_categories'], sort_keys=True)}",
        f"- B1 failure reasons: {json.dumps(metrics['b1_failure_reason_counts'], sort_keys=True)}",
        f"- B1 safe-success cycle time median/mean/min/max: {cycle['median']} / {cycle['mean']} / {cycle['minimum']} / {cycle['maximum']} s",
        "",
        "## Frozen config candidate",
        "",
        f"- Path: `{result['frozen_config']['path']}`",
        f"- SHA-256: `{result['frozen_config']['sha256']}`",
        f"- Byte-identical to source template: {str(result['frozen_config']['byte_identical_to_template']).lower()}",
        f"- Loaded EnvConfig, ControllerConfig, and B1Config equivalent: {str(result['frozen_config']['behavior_equivalent']).lower()}",
        "",
        "No controller was run by this verifier. Development and Held-out Test were not read.",
        "",
    ]
    return "\n".join(lines)


def verify_b1_freeze(
    *,
    round_zero_dir: str | Path,
    verification_dir: str | Path,
    frozen_config: str | Path,
    protocol_path: str | Path,
    write_outputs: bool = True,
) -> dict[str, Any]:
    round_zero = Path(round_zero_dir).expanduser().resolve()
    verification = Path(verification_dir).expanduser().resolve()
    frozen_path = Path(frozen_config).expanduser().resolve()
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol, seeds = _load_protocol_and_calibration(protocol_file)
    reference = _validate_archive(
        round_zero,
        expected_seeds=seeds,
        protocol=protocol,
        label="Round 0",
    )
    candidate = _validate_archive(
        verification,
        expected_seeds=seeds,
        protocol=protocol,
        label="Freeze Verification",
    )
    if (
        reference["config_sha256"] != candidate["config_sha256"]
        or reference["protocol_sha256"] != candidate["protocol_sha256"]
        or reference["split_sha256"] != candidate["split_sha256"]
    ):
        raise FreezeVerificationError("Formal config, protocol, or split hash changed")

    episode_comparisons = compare_episode_rows(
        reference["episodes"], candidate["episodes"], seeds
    )
    pair_comparisons = compare_pair_rows(
        reference["pairs"], candidate["pairs"], seeds
    )
    if _failure_count_map(reference["failure_counts"]) != _failure_count_map(
        candidate["failure_counts"]
    ):
        raise FreezeVerificationError("Failure reason counts changed")
    compare_nested_metrics(reference["summary"], candidate["summary"])
    compare_nested_metrics(
        reference["production_metrics"], candidate["production_metrics"]
    )
    verified_metrics = _expected_results(
        candidate["summary"], candidate["production_metrics"]
    )

    template_path = (
        PROJECT_ROOT / str(protocol.raw["calibration"]["baseline_template_path"])
    ).resolve()
    if not frozen_path.is_file():
        raise FreezeVerificationError(f"Frozen config candidate is missing: {frozen_path}")
    template_config = load_config(template_path)
    frozen = load_config(frozen_path)
    validate_baseline_compatibility(protocol, frozen)
    behavior_equivalent = frozen == template_config
    if not behavior_equivalent:
        raise FreezeVerificationError("Frozen config is not behavior-equivalent to template")
    byte_identical = frozen_path.read_bytes() == template_path.read_bytes()
    if not byte_identical:
        raise FreezeVerificationError("Frozen config candidate is not an exact template copy")
    if not (
        frozen.controller == template_config.controller
        and frozen.b1 == template_config.b1
        and frozen.b1.final_place_xy_tolerance
        == protocol.environment.b1.final_place_xy_tolerance
        and frozen.b1.final_place_height_tolerance
        == protocol.environment.b1.final_place_height_tolerance
    ):
        raise FreezeVerificationError("Frozen config changed protected success/controller fields")

    maximum_time_difference = max(
        row["simulation_time_absolute_difference"] for row in episode_comparisons
    )
    b1_cycle = candidate["production_metrics"]["methods"]["b1_vision"]
    result: dict[str, Any] = {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "PASS",
        "baseline_id": "b1_vision_v1",
        "protocol": {
            "protocol_id": protocol.protocol_id,
            "protocol_version": protocol.protocol_version,
            "metrics_schema_version": protocol.metrics_schema_version,
            "sha256": protocol.sha256,
        },
        "split": {
            "split_id": protocol.split_id,
            "name": "calibration",
            "seed_count": len(seeds),
            "sha256": candidate["split_sha256"],
        },
        "completion": {
            "requested_pairs": 30,
            "completed_pairs": 30,
            "b0_episode_count": 30,
            "b1_episode_count": 30,
            "invalid_pairs": 0,
            "program_errors": 0,
            "invalid_numeric_episodes": 0,
        },
        "diagnostics": {
            "enabled": False,
            "visualization_enabled": False,
            "diagnostic_artifacts_present": False,
            "privileged_diagnostic_schema_fields_present": False,
        },
        "automatic_parameter_search": False,
        "comparison": {
            "reference": str(round_zero),
            "candidate": str(verification),
            "episode_comparison_count": len(episode_comparisons),
            "pair_comparison_count": len(pair_comparisons),
            "exact_mismatch_count": 0,
            "maximum_simulation_time_absolute_difference": maximum_time_difference,
            "all_episode_fields_match": True,
            "all_pair_fields_match": True,
            "failure_reason_counts_match": True,
            "final_stage_counts_match": True,
            "production_metrics_match": True,
            "cycle_time_summary_match": True,
        },
        "simulation_time_rule": {
            "relative_tolerance": 0.0,
            "absolute_tolerance_seconds": SIMULATION_TIME_ABSOLUTE_TOLERANCE,
            "source": "Round 0.5 replay comparison",
            "not_relaxed": True,
        },
        "verified_metrics": verified_metrics,
        "cycle_time": {
            "count": b1_cycle["safe_successful_simulation_time_count"],
            "median": b1_cycle["safe_successful_simulation_time_median"],
            "mean": b1_cycle["safe_successful_simulation_time_mean"],
            "minimum": b1_cycle["safe_successful_simulation_time_minimum"],
            "maximum": b1_cycle["safe_successful_simulation_time_maximum"],
            "unit": "simulation_seconds",
        },
        "frozen_config": {
            "path": frozen_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(frozen_path),
            "source_template_path": template_path.relative_to(PROJECT_ROOT).as_posix(),
            "source_template_sha256": _sha256(template_path),
            "byte_identical_to_template": byte_identical,
            "behavior_equivalent": behavior_equivalent,
            "env_config_equivalent": frozen == template_config,
            "controller_config_equivalent": frozen.controller == template_config.controller,
            "b1_config_equivalent": frozen.b1 == template_config.b1,
        },
        "input_hashes": {
            "round_0": reference["hashes"],
            "freeze_verification_formal_run": candidate["hashes"],
        },
        "input_access_boundary": {
            "calibration_read": True,
            "development_read": False,
            "held_out_test_read": False,
            "controller_executed": False,
            "config_modified": False,
        },
        "episode_comparisons": episode_comparisons,
        "pair_comparisons": pair_comparisons,
    }
    result["comparison_payload_sha256"] = _canonical_sha256(result)
    if write_outputs:
        csv_rows = [
            {
                "seed": row["seed"],
                "method_id": row["method_id"],
                "pair_id": row["pair_id"],
                "episode_fingerprint": row["episode_fingerprint"],
                "final_stage": row["final_stage"],
                "failure_reason": row["failure_reason"],
                "controller_reported_success": row["controller_reported_success"],
                "privileged_ground_truth_success": row[
                    "privileged_ground_truth_success"
                ],
                "safe_task_success": row["safe_task_success"],
                "collision_count": row["collision_count"],
                "simulation_time_reference": row["simulation_time_reference"],
                "simulation_time_candidate": row["simulation_time_candidate"],
                "simulation_time_absolute_difference": row[
                    "simulation_time_absolute_difference"
                ],
                "all_exact_fields_match": True,
                "simulation_time_within_tolerance": True,
            }
            for row in episode_comparisons
        ]
        write_csv(
            verification / "freeze_comparison.csv",
            csv_rows,
            tuple(csv_rows[0]),
        )
        write_json(verification / "freeze_comparison.json", result)
        (verification / "freeze_verification_report.md").write_text(
            _render_markdown(result), encoding="utf-8", newline="\n"
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly compare a completed formal B1 freeze-verification archive "
            "with Round 0. This tool never runs a controller or modifies a config."
        )
    )
    parser.add_argument("--round-zero-dir", type=Path, required=True)
    parser.add_argument("--verification-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_b1_freeze(
            round_zero_dir=args.round_zero_dir,
            verification_dir=args.verification_dir,
            frozen_config=args.frozen_config,
            protocol_path=args.protocol,
        )
    except Exception as exc:
        print(f"FREEZE VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "B1-Vision v1 freeze comparison passed: "
        f"episodes={result['comparison']['episode_comparison_count']}/60, "
        f"pairs={result['comparison']['pair_comparison_count']}/30, "
        "exact_mismatches=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
