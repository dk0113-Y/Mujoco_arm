from __future__ import annotations

from collections import Counter
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .protocol import ProtocolConfig


PAIR_CATEGORIES = (
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
)

_REQUIRED_RESULT_FIELDS = (
    "seed",
    "method_id",
    "pair_valid",
    "program_error",
    "episode_fingerprint",
    "pick_region",
    "place_region",
    "sampled_pick_position",
    "sampled_place_position",
    "sampled_mass",
    "sampled_friction",
    "final_stage",
    "simulation_time",
    "collision_count",
    "controller_reported_success",
    "privileged_ground_truth_success",
    "failure_reason",
)


def _program_error(row: Mapping[str, Any]) -> bool:
    value = row.get("program_error")
    return value not in (None, False, "", 0)


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _finite_vector(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _required_fields_complete(row: Mapping[str, Any], protocol: ProtocolConfig) -> bool:
    if any(name not in row for name in _REQUIRED_RESULT_FIELDS):
        return False
    seed = row.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        return False
    if not isinstance(row.get("method_id"), str) or not row.get("method_id"):
        return False
    if not isinstance(row.get("pair_valid"), bool):
        return False
    if not isinstance(row.get("episode_fingerprint"), str) or not row.get(
        "episode_fingerprint"
    ):
        return False
    if row.get("pick_region") not in {"front", "left", "right"}:
        return False
    if row.get("place_region") not in {"front", "left", "right"}:
        return False
    if not _finite_vector(row.get("sampled_pick_position"), 3):
        return False
    if not _finite_vector(row.get("sampled_place_position"), 3):
        return False
    if not _finite_number(row.get("sampled_mass"), minimum=0.0):
        return False
    friction = row.get("sampled_friction")
    if not _finite_vector(friction, 3) or any(float(item) <= 0.0 for item in friction):
        return False
    if row.get("final_stage") not in set(protocol.declared_stages):
        return False
    if not _finite_number(row.get("simulation_time"), minimum=0.0):
        return False
    collision_count = row.get("collision_count")
    if (
        isinstance(collision_count, bool)
        or not isinstance(collision_count, int)
        or collision_count < 0
    ):
        return False
    if not isinstance(row.get("controller_reported_success"), bool):
        return False
    if not isinstance(row.get("privileged_ground_truth_success"), bool):
        return False
    failure_reason = row.get("failure_reason")
    if failure_reason is not None and not isinstance(failure_reason, str):
        return False
    return True


def _invalid_numeric(row: Mapping[str, Any]) -> bool:
    scalar_names = ("simulation_time", "sampled_mass")
    for name in scalar_names:
        if name in row and row[name] is not None and not _finite_number(row[name]):
            return True
    for name in ("sampled_pick_position", "sampled_place_position", "sampled_friction"):
        value = row.get(name)
        if value is not None and not _finite_vector(value, 3):
            return True
    return False


def derive_episode_protocol_fields(
    row: Mapping[str, Any],
    protocol: ProtocolConfig,
) -> dict[str, Any]:
    """Derive protocol outcomes without feeding any value back to control."""

    final_stage = row.get("final_stage")
    released = final_stage in set(protocol.released_stages)
    simulation_time = row.get("simulation_time")
    within_timeout = bool(
        _finite_number(simulation_time, minimum=0.0)
        and float(simulation_time) <= protocol.episode_timeout + 1e-9
    )
    structured_end = final_stage in set(protocol.declared_stages)
    ground_truth = row.get("privileged_ground_truth_success") is True
    placement_success = bool(
        released and ground_truth and structured_end and within_timeout
    )
    collision_count = row.get("collision_count")
    collision_episode = bool(
        isinstance(collision_count, int)
        and not isinstance(collision_count, bool)
        and collision_count > 0
    )
    fields_complete = _required_fields_complete(row, protocol)
    unexpected_exception = row.get("failure_reason") == "unexpected_exception"
    safe_task_success = bool(
        placement_success
        and not collision_episode
        and not _program_error(row)
        and not unexpected_exception
        and fields_complete
        and within_timeout
    )
    regrasp_count = row.get("full_regrasp_count", 0)
    first_attempt = bool(
        placement_success
        and isinstance(regrasp_count, int)
        and not isinstance(regrasp_count, bool)
        and regrasp_count == 0
    )

    invalid_pair = row.get("pair_valid") is not True
    missing_reason = bool(
        not placement_success
        and row.get("controller_reported_success") is False
        and not row.get("failure_reason")
    )
    unknown_failure = row.get("failure_reason") == "unknown_failure"
    unexplained = bool(
        _program_error(row)
        or unexpected_exception
        or invalid_pair
        or unknown_failure
        or missing_reason
        or not fields_complete
        or not structured_end
        or _invalid_numeric(row)
    )

    pick = row.get("sampled_pick_position")
    place = row.get("sampled_place_position")
    distance: float | None = None
    if _finite_vector(pick, 3) and _finite_vector(place, 3):
        distance = math.sqrt(
            sum((float(left) - float(right)) ** 2 for left, right in zip(pick, place))
        )
    pick_region = row.get("pick_region")
    place_region = row.get("place_region")
    pair = (
        f"{pick_region}->{place_region}"
        if isinstance(pick_region, str) and isinstance(place_region, str)
        else None
    )
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "split_id": protocol.split_id,
        "placement_success": placement_success,
        "safe_task_success": safe_task_success,
        "first_attempt_placement_success": first_attempt,
        "collision_episode": collision_episode,
        "unexplained_failure": unexplained,
        "result_fields_complete": fields_complete,
        "object_released": released,
        "completed_within_timeout": within_timeout,
        "pick_place_region_pair": pair,
        "same_region": bool(pair is not None and pick_region == place_region),
        "pick_place_distance": distance,
    }


def _rate(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else count / denominator


def build_production_metrics(
    episode_rows: Iterable[Mapping[str, Any]],
    paired_rows: Iterable[Mapping[str, Any]] = (),
    *,
    protocol: ProtocolConfig,
) -> dict[str, Any]:
    """Calculate Evaluation Protocol v1 metrics as strict pure Python."""

    rows = [dict(row) for row in episode_rows]
    derived = [derive_episode_protocol_fields(row, protocol) for row in rows]
    enriched = [{**row, **fields} for row, fields in zip(rows, derived)]
    valid = [
        row
        for row in enriched
        if row.get("pair_valid") is True and not _program_error(row)
    ]
    placement_count = sum(row["placement_success"] is True for row in valid)
    safe_count = sum(row["safe_task_success"] is True for row in valid)
    first_attempt_count = sum(
        row["first_attempt_placement_success"] is True for row in valid
    )
    collision_count = sum(row["collision_episode"] is True for row in valid)
    safe_times = [
        float(row["simulation_time"])
        for row in valid
        if row["safe_task_success"] is True
        and _finite_number(row.get("simulation_time"), minimum=0.0)
    ]
    unexplained_count = sum(row["unexplained_failure"] is True for row in enriched)
    failures: Counter[str] = Counter()
    final_stages: Counter[str] = Counter()
    for row in valid:
        reason = (
            "success"
            if row.get("controller_reported_success") is True
            else str(row.get("failure_reason") or "unknown_failure")
        )
        failures[reason] += 1
        final_stages[str(row.get("final_stage") or "missing")] += 1
    pair_counts = Counter(
        str(row.get("outcome_category"))
        for row in paired_rows
        if row.get("outcome_category") in PAIR_CATEGORIES
    )
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "metrics_schema_version": protocol.metrics_schema_version,
        "requested_episode_count": len(enriched),
        "valid_episode_count": len(valid),
        "placement_success_count": placement_count,
        "placement_success_rate": _rate(placement_count, len(valid)),
        "safe_task_success_count": safe_count,
        "safe_task_success_rate": _rate(safe_count, len(valid)),
        "first_attempt_placement_success_count": first_attempt_count,
        "first_attempt_placement_success_rate": _rate(first_attempt_count, len(valid)),
        "collision_episode_count": collision_count,
        "collision_episode_rate": _rate(collision_count, len(valid)),
        "safe_successful_simulation_time_count": len(safe_times),
        "safe_successful_simulation_time_median": (
            statistics.median(safe_times) if safe_times else None
        ),
        "safe_successful_simulation_time_mean": (
            statistics.fmean(safe_times) if safe_times else None
        ),
        "safe_successful_simulation_time_minimum": min(safe_times) if safe_times else None,
        "safe_successful_simulation_time_maximum": max(safe_times) if safe_times else None,
        "unexplained_failure_count": unexplained_count,
        "unexplained_failure_rate": _rate(unexplained_count, len(enriched)),
        "invalid_numeric_episode_count": sum(_invalid_numeric(row) for row in rows),
        "failure_reason_counts": dict(sorted(failures.items())),
        "final_stage_counts": dict(sorted(final_stages.items())),
        "oracle_vision_pair_counts": {
            category: pair_counts[category] for category in PAIR_CATEGORIES
        },
    }


__all__ = [
    "PAIR_CATEGORIES",
    "build_production_metrics",
    "derive_episode_protocol_fields",
]
