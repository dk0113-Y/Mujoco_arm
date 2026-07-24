"""Seven-axis joint impedance control for the isolated Panda torque model.

The feedback equation follows the structure of the Apache-2.0 libfranka
joint-impedance example (tag 0.21.2).  This module is a MuJoCo adaptation, not
Franka official code: the caller must supply the verified physical motor
compensation required by the simulation plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


JOINT_COUNT = 7


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must have shape ({JOINT_COUNT},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


@dataclass(frozen=True)
class JointImpedanceDiagnostics:
    position_error: np.ndarray
    velocity_error: np.ndarray
    feedback_torque: np.ndarray
    dynamics_compensation: np.ndarray
    raw_torque: np.ndarray
    rate_limited_torque: np.ndarray
    final_torque: np.ndarray
    saturation_mask: np.ndarray
    rate_limit_mask: np.ndarray
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "position_error": self.position_error.copy(),
            "velocity_error": self.velocity_error.copy(),
            "feedback_torque": self.feedback_torque.copy(),
            "dynamics_compensation": self.dynamics_compensation.copy(),
            "raw_torque": self.raw_torque.copy(),
            "rate_limited_torque": self.rate_limited_torque.copy(),
            "final_torque": self.final_torque.copy(),
            "saturation_mask": self.saturation_mask.copy(),
            "rate_limit_mask": self.rate_limit_mask.copy(),
            "finite": self.finite,
        }


class JointImpedanceController:
    """Independent seven-joint spring-damper controller with torque guards."""

    def __init__(
        self,
        *,
        stiffness: np.ndarray,
        damping: np.ndarray,
        torque_limits: np.ndarray,
        torque_rate_limits: np.ndarray,
    ) -> None:
        self.stiffness = _vector(stiffness, "stiffness").copy()
        self.damping = _vector(damping, "damping").copy()
        self.torque_limits = _vector(torque_limits, "torque_limits").copy()
        self.torque_rate_limits = _vector(
            torque_rate_limits, "torque_rate_limits"
        ).copy()
        if np.any(self.stiffness <= 0.0):
            raise ValueError("stiffness values must be positive")
        if np.any(self.damping < 0.0):
            raise ValueError("damping values must be non-negative")
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("torque_limits values must be positive")
        if np.any(self.torque_rate_limits <= 0.0):
            raise ValueError("torque_rate_limits values must be positive")
        self._previous_torque = np.zeros(JOINT_COUNT, dtype=float)

    @property
    def previous_torque(self) -> np.ndarray:
        return self._previous_torque.copy()

    def reset(self, previous_torque: np.ndarray | None = None) -> None:
        self._previous_torque = (
            np.zeros(JOINT_COUNT, dtype=float)
            if previous_torque is None
            else _vector(previous_torque, "previous_torque").copy()
        )

    def compute(
        self,
        *,
        q: np.ndarray,
        dq: np.ndarray,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        dynamics_compensation: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, JointImpedanceDiagnostics]:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        q_value = _vector(q, "q")
        dq_value = _vector(dq, "dq")
        target_value = _vector(q_target, "q_target")
        target_velocity = _vector(dq_target, "dq_target")
        compensation = _vector(
            dynamics_compensation, "dynamics_compensation"
        )

        position_error = target_value - q_value
        velocity_error = target_velocity - dq_value
        feedback = (
            self.stiffness * position_error + self.damping * velocity_error
        )
        raw = feedback + compensation
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("joint impedance computation produced NaN or Inf")

        maximum_delta = self.torque_rate_limits * float(dt)
        rate_limited = np.clip(
            raw,
            self._previous_torque - maximum_delta,
            self._previous_torque + maximum_delta,
        )
        rate_mask = np.abs(rate_limited - raw) > 1e-12
        final = np.clip(rate_limited, -self.torque_limits, self.torque_limits)
        saturation_mask = np.abs(final - rate_limited) > 1e-12
        if not np.all(np.isfinite(final)):
            raise FloatingPointError("limited joint torque contains NaN or Inf")
        self._previous_torque = final.copy()

        diagnostics = JointImpedanceDiagnostics(
            position_error=position_error.copy(),
            velocity_error=velocity_error.copy(),
            feedback_torque=feedback.copy(),
            dynamics_compensation=compensation.copy(),
            raw_torque=raw.copy(),
            rate_limited_torque=rate_limited.copy(),
            final_torque=final.copy(),
            saturation_mask=saturation_mask.copy(),
            rate_limit_mask=rate_mask.copy(),
            finite=True,
        )
        return final.copy(), diagnostics
