from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _stack(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=float)


def _settling_times(
    times: np.ndarray,
    errors: np.ndarray,
    *,
    tolerance: float,
    applicable: bool,
) -> list[float | None]:
    if not applicable:
        return [None] * errors.shape[1]
    result: list[float | None] = []
    for joint in range(errors.shape[1]):
        settled: float | None = None
        within = np.abs(errors[:, joint]) <= tolerance
        for index in range(len(times)):
            if bool(np.all(within[index:])):
                settled = float(times[index] - times[0])
                break
        result.append(settled)
    return result


def compute_episode_metrics(
    *,
    episode_id: str,
    experiment: str,
    case_name: str,
    rows: Sequence[dict[str, Any]],
    termination_reason: str,
    wall_clock_duration: float,
    settling_applicable: bool,
    settling_tolerance: float = 0.02,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute metrics for an episode with no rows")
    q = _stack(rows, "q")
    dq = _stack(rows, "dq")
    q_target = _stack(rows, "q_target")
    dq_target = _stack(rows, "dq_target")
    position_error = q_target - q
    velocity_error = dq_target - dq
    final_torque = _stack(rows, "final_torque")
    saturation = np.asarray(
        [row["saturation_mask"] for row in rows], dtype=bool
    )
    rate_limit = np.asarray(
        [row["rate_limit_mask"] for row in rows], dtype=bool
    )
    times = np.asarray([row["sim_time"] for row in rows], dtype=float)

    arrays = (
        q,
        dq,
        q_target,
        dq_target,
        position_error,
        velocity_error,
        final_torque,
        times,
    )
    finite = all(np.all(np.isfinite(array)) for array in arrays)
    if not finite:
        raise FloatingPointError("Episode metrics input contains NaN or Inf")
    if not math.isfinite(wall_clock_duration) or wall_clock_duration < 0.0:
        raise ValueError("wall_clock_duration must be finite and non-negative")

    tail_start = max(0, int(math.floor(len(rows) * 0.8)))
    steady_error = np.mean(position_error[tail_start:], axis=0)
    initial_position = q[0]
    final_target = q_target[-1]
    displacement = final_target - initial_position
    overshoot = np.empty(7, dtype=float)
    for joint in range(7):
        if abs(displacement[joint]) <= 1e-9:
            overshoot[joint] = float(
                np.max(np.abs(q[:, joint] - final_target[joint]))
            )
        else:
            direction = math.copysign(1.0, displacement[joint])
            overshoot[joint] = max(
                0.0,
                float(
                    np.max(direction * (q[:, joint] - final_target[joint]))
                ),
            )
    saturation_count = int(np.count_nonzero(saturation))
    rate_limit_count = int(np.count_nonzero(rate_limit))
    denominator = int(saturation.size)
    terminated = termination_reason != "completed"
    simulated_duration = float(times[-1])

    return {
        "episode_id": episode_id,
        "experiment": experiment,
        "case": case_name,
        "position_rmse": np.sqrt(np.mean(position_error**2, axis=0)).tolist(),
        "maximum_absolute_position_error": np.max(
            np.abs(position_error), axis=0
        ).tolist(),
        "velocity_rmse": np.sqrt(np.mean(velocity_error**2, axis=0)).tolist(),
        "maximum_absolute_torque": np.max(
            np.abs(final_torque), axis=0
        ).tolist(),
        "rms_torque": np.sqrt(np.mean(final_torque**2, axis=0)).tolist(),
        "torque_saturation_count": saturation_count,
        "torque_saturation_ratio": saturation_count / denominator,
        "torque_rate_limit_count": rate_limit_count,
        "torque_rate_limit_ratio": rate_limit_count / denominator,
        "maximum_joint_velocity": np.max(np.abs(dq), axis=0).tolist(),
        "final_position_error": position_error[-1].tolist(),
        "steady_state_error": steady_error.tolist(),
        "overshoot": overshoot.tolist(),
        "settling_time": _settling_times(
            times,
            position_error,
            tolerance=settling_tolerance,
            applicable=settling_applicable,
        ),
        "finite_value_status": True,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "simulated_duration": simulated_duration,
        "wall_clock_duration": float(wall_clock_duration),
    }


def summarize_metrics(metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("At least one episode metric is required")
    reasons: dict[str, int] = {}
    for metric in metrics:
        reason = str(metric["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    total_saturation = sum(
        int(metric["torque_saturation_count"]) for metric in metrics
    )
    total_rate_limit = sum(
        int(metric["torque_rate_limit_count"]) for metric in metrics
    )
    return {
        "episode_count": len(metrics),
        "completed_episode_count": sum(
            metric["termination_reason"] == "completed" for metric in metrics
        ),
        "terminated_episode_count": sum(bool(metric["terminated"]) for metric in metrics),
        "termination_reason_counts": dict(sorted(reasons.items())),
        "finite_value_status": all(
            bool(metric["finite_value_status"]) for metric in metrics
        ),
        "total_torque_saturation_count": total_saturation,
        "total_torque_rate_limit_count": total_rate_limit,
        "maximum_position_rmse": max(
            max(float(value) for value in metric["position_rmse"])
            for metric in metrics
        ),
        "maximum_absolute_torque": max(
            max(float(value) for value in metric["maximum_absolute_torque"])
            for metric in metrics
        ),
        "simulated_duration": sum(
            float(metric["simulated_duration"]) for metric in metrics
        ),
        "wall_clock_duration": sum(
            float(metric["wall_clock_duration"]) for metric in metrics
        ),
    }
