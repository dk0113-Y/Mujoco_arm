from __future__ import annotations

from collections import Counter
import statistics
from typing import Any, Iterable, Mapping, Sequence


PAIR_CATEGORIES = (
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
    "invalid_pair",
    "program_error",
)


def _is_program_error(row: Mapping[str, Any]) -> bool:
    return bool(row.get("program_error"))


def _valid_completed(row: Mapping[str, Any]) -> bool:
    return bool(row.get("pair_valid")) and not _is_program_error(row)


def _rate(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else count / denominator


def failure_counts_rows(
    episode_rows: Iterable[Mapping[str, Any]],
    method_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = list(episode_rows)
    result: list[dict[str, Any]] = []
    for method_id in method_ids:
        counter: Counter[str] = Counter()
        for row in rows:
            if row.get("method_id") != method_id or not _valid_completed(row):
                continue
            reason = (
                "success"
                if bool(row.get("controller_reported_success"))
                else str(row.get("failure_reason") or "unknown_failure")
            )
            counter[reason] += 1
        for reason in sorted(counter):
            result.append(
                {
                    "method_id": method_id,
                    "failure_reason": reason,
                    "count": counter[reason],
                }
            )
    return result


def build_summary(
    episode_rows: Iterable[Mapping[str, Any]],
    paired_rows: Iterable[Mapping[str, Any]],
    method_ids: Sequence[str],
    *,
    requested_episode_count: int | None = None,
) -> dict[str, Any]:
    episodes = list(episode_rows)
    pairs = list(paired_rows)
    method_summary: dict[str, dict[str, Any]] = {}
    for method_id in method_ids:
        requested = (
            requested_episode_count
            if requested_episode_count is not None
            else sum(row.get("method_id") == method_id for row in episodes)
        )
        method_rows = [row for row in episodes if row.get("method_id") == method_id]
        completed_rows = [row for row in method_rows if _valid_completed(row)]
        completed = len(completed_rows)
        program_errors = sum(_is_program_error(row) for row in method_rows)
        ground_truth_success = sum(
            bool(row.get("privileged_ground_truth_success"))
            for row in completed_rows
        )
        controller_success = sum(
            bool(row.get("controller_reported_success"))
            for row in completed_rows
        )
        successful_times = [
            float(row["simulation_time"])
            for row in completed_rows
            if bool(row.get("privileged_ground_truth_success"))
            and row.get("simulation_time") is not None
        ]
        failures = Counter(
            (
                "success"
                if bool(row.get("controller_reported_success"))
                else str(row.get("failure_reason") or "unknown_failure")
            )
            for row in completed_rows
        )
        method_summary[method_id] = {
            "requested_episodes": requested,
            "completed_episodes": completed,
            "program_errors": program_errors,
            "ground_truth_success_count": ground_truth_success,
            "ground_truth_success_rate": _rate(ground_truth_success, completed),
            "controller_reported_success_count": controller_success,
            "controller_reported_success_rate": _rate(controller_success, completed),
            "false_positive_count": sum(
                bool(row.get("false_positive")) for row in completed_rows
            ),
            "false_negative_count": sum(
                bool(row.get("false_negative")) for row in completed_rows
            ),
            "collision_episode_count": sum(
                int(row.get("collision_count") or 0) > 0 for row in completed_rows
            ),
            "successful_episode_simulation_time_mean": (
                statistics.fmean(successful_times) if successful_times else None
            ),
            "successful_episode_simulation_time_median": (
                statistics.median(successful_times) if successful_times else None
            ),
            "failure_reason_counts": dict(sorted(failures.items())),
        }

    category_counts = Counter(
        str(row.get("outcome_category") or "program_error") for row in pairs
    )
    paired_summary = {
        "valid_pair_count": sum(bool(row.get("pair_valid")) for row in pairs),
        "both_success": category_counts["both_success"],
        "oracle_only_success": category_counts["oracle_only_success"],
        "vision_only_success": category_counts["vision_only_success"],
        "both_failed": category_counts["both_failed"],
        "invalid_pair_count": category_counts["invalid_pair"],
        "program_error_pair_count": category_counts["program_error"],
    }
    return {"methods": method_summary, "paired": paired_summary}

