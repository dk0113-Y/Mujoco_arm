"""Episode metrics for the fixed-gain Cartesian impedance benchmark."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _stack(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([row[field] for row in rows], dtype=float)


def _norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def _rms(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    result = np.sqrt(np.mean(values**2, axis=axis))
    return float(result) if np.ndim(result) == 0 else result


def compute_cartesian_episode_metrics(
    *,
    episode_id: str,
    experiment: str,
    case_name: str,
    rows: Sequence[dict[str, Any]],
    termination_reason: str,
    wall_clock_duration: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute Cartesian metrics without rows")
    position_error = _stack(rows, "position_error")
    orientation_error = _stack(rows, "orientation_error")
    linear_velocity_error = _stack(rows, "linear_velocity_error")
    angular_velocity_error = _stack(rows, "angular_velocity_error")
    task_wrench = _stack(rows, "task_wrench")
    final_torque = _stack(rows, "final_torque")
    dq = _stack(rows, "dq")
    singular_values = _stack(rows, "jacobian_singular_values")
    ranks = np.asarray([row["jacobian_rank"] for row in rows], dtype=int)
    conditions = np.asarray(
        [row["jacobian_condition_number"] for row in rows], dtype=float
    )
    saturation = np.asarray(
        [row["torque_saturation_mask"] for row in rows], dtype=bool
    )
    rate_limit = np.asarray(
        [row["torque_rate_limit_mask"] for row in rows], dtype=bool
    )
    times = np.asarray([row["sim_time"] for row in rows], dtype=float)
    finite_status = np.asarray(
        [row["finite_value_status"] for row in rows], dtype=bool
    )
    arrays = (
        position_error,
        orientation_error,
        linear_velocity_error,
        angular_velocity_error,
        task_wrench,
        final_torque,
        dq,
        singular_values,
        conditions,
        times,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("Cartesian metrics input contains NaN or Inf")
    if not math.isfinite(wall_clock_duration) or wall_clock_duration < 0.0:
        raise ValueError("wall_clock_duration must be finite and non-negative")

    position_norm = _norm(position_error)
    orientation_angle = _norm(orientation_error)
    linear_velocity_norm = _norm(linear_velocity_error)
    angular_velocity_norm = _norm(angular_velocity_error)
    task_force = _norm(task_wrench[:, :3])
    task_moment = _norm(task_wrench[:, 3:])
    tail_start = max(0, int(math.floor(len(rows) * 0.8)))
    saturation_count = int(np.count_nonzero(saturation))
    rate_limit_count = int(np.count_nonzero(rate_limit))
    denominator = int(saturation.size)
    unexpected_contact = any(bool(row["unexpected_contact"]) for row in rows)

    return {
        "episode_id": episode_id,
        "experiment": experiment,
        "case": case_name,
        "xyz_position_rmse": _rms(position_error, axis=0).tolist(),
        "position_error_norm_rmse": _rms(position_norm),
        "maximum_position_error_norm": float(np.max(position_norm)),
        "final_position_error": position_error[-1].tolist(),
        "steady_state_position_error": np.mean(
            position_error[tail_start:], axis=0
        ).tolist(),
        "steady_state_position_error_norm": float(
            np.mean(position_norm[tail_start:])
        ),
        "orientation_geodesic_angle_rmse": _rms(orientation_angle),
        "maximum_orientation_error": float(np.max(orientation_angle)),
        "final_orientation_error": float(orientation_angle[-1]),
        "final_orientation_error_vector": orientation_error[-1].tolist(),
        "xyz_linear_velocity_rmse": _rms(
            linear_velocity_error, axis=0
        ).tolist(),
        "linear_velocity_error_norm_rmse": _rms(linear_velocity_norm),
        "xyz_angular_velocity_rmse": _rms(
            angular_velocity_error, axis=0
        ).tolist(),
        "angular_velocity_error_norm_rmse": _rms(angular_velocity_norm),
        "maximum_task_force": float(np.max(task_force)),
        "rms_task_force": _rms(task_force),
        "maximum_task_moment": float(np.max(task_moment)),
        "rms_task_moment": _rms(task_moment),
        "maximum_absolute_joint_torque": np.max(
            np.abs(final_torque), axis=0
        ).tolist(),
        "rms_joint_torque": _rms(final_torque, axis=0).tolist(),
        "torque_saturation_count": saturation_count,
        "torque_saturation_ratio": saturation_count / denominator,
        "torque_rate_limit_count": rate_limit_count,
        "torque_rate_limit_ratio": rate_limit_count / denominator,
        "minimum_jacobian_singular_value": float(
            np.min(singular_values[:, -1])
        ),
        "maximum_jacobian_condition_number": float(np.max(conditions)),
        "minimum_observed_jacobian_rank": int(np.min(ranks)),
        "maximum_joint_velocity": np.max(np.abs(dq), axis=0).tolist(),
        "finite_value_status": bool(np.all(finite_status)),
        "unexpected_contact": unexpected_contact,
        "terminated": termination_reason != "completed",
        "termination_reason": termination_reason,
        "simulated_duration": float(times[-1]),
        "wall_clock_duration": float(wall_clock_duration),
    }


def summarize_cartesian_metrics(
    metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("At least one Cartesian episode metric is required")
    reasons: dict[str, int] = {}
    experiment_metrics: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        reason = str(metric["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
        experiment_metrics.setdefault(str(metric["experiment"]), []).append(metric)
    by_experiment = {
        name: {
            "episode_count": len(values),
            "completed_episode_count": sum(
                value["termination_reason"] == "completed" for value in values
            ),
            "maximum_position_error_norm": max(
                float(value["maximum_position_error_norm"]) for value in values
            ),
            "maximum_orientation_error": max(
                float(value["maximum_orientation_error"]) for value in values
            ),
            "maximum_jacobian_condition_number": max(
                float(value["maximum_jacobian_condition_number"])
                for value in values
            ),
        }
        for name, values in sorted(experiment_metrics.items())
    }
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
        "unexpected_contact_episode_count": sum(
            bool(metric["unexpected_contact"]) for metric in metrics
        ),
        "total_torque_saturation_count": sum(
            int(metric["torque_saturation_count"]) for metric in metrics
        ),
        "total_torque_rate_limit_count": sum(
            int(metric["torque_rate_limit_count"]) for metric in metrics
        ),
        "maximum_position_error_norm": max(
            float(metric["maximum_position_error_norm"]) for metric in metrics
        ),
        "maximum_orientation_error": max(
            float(metric["maximum_orientation_error"]) for metric in metrics
        ),
        "minimum_jacobian_singular_value": min(
            float(metric["minimum_jacobian_singular_value"])
            for metric in metrics
        ),
        "maximum_jacobian_condition_number": max(
            float(metric["maximum_jacobian_condition_number"])
            for metric in metrics
        ),
        "minimum_observed_jacobian_rank": min(
            int(metric["minimum_observed_jacobian_rank"]) for metric in metrics
        ),
        "simulated_duration": sum(
            float(metric["simulated_duration"]) for metric in metrics
        ),
        "wall_clock_duration": sum(
            float(metric["wall_clock_duration"]) for metric in metrics
        ),
        "by_experiment": by_experiment,
    }
