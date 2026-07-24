"""Fixed-gain world-frame Cartesian impedance for the isolated Panda."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from control_benchmarks.kinematics import (
    orientation_error_world,
    validate_rotation_matrix,
)


JOINT_COUNT = 7


def _vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _jacobian(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (6, JOINT_COUNT):
        raise ValueError(f"jacobian must have shape (6, 7), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("jacobian contains NaN or Inf")
    return result


@dataclass(frozen=True)
class CartesianImpedanceDiagnostics:
    position_error: np.ndarray
    orientation_error: np.ndarray
    linear_velocity_error: np.ndarray
    angular_velocity_error: np.ndarray
    pose_error: np.ndarray
    twist_error: np.ndarray
    task_wrench: np.ndarray
    task_torque: np.ndarray
    dynamics_compensation: np.ndarray
    raw_torque: np.ndarray
    rate_limited_torque: np.ndarray
    final_torque: np.ndarray
    saturation_mask: np.ndarray
    rate_limit_mask: np.ndarray
    finite: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            field: value.copy() if isinstance(value, np.ndarray) else value
            for field, value in self.__dict__.items()
        }


class CartesianImpedanceController:
    """Six-dimensional spring-damper mapped by the world-frame Jacobian."""

    def __init__(
        self,
        *,
        translational_stiffness: np.ndarray,
        rotational_stiffness: np.ndarray,
        translational_damping: np.ndarray,
        rotational_damping: np.ndarray,
        torque_limits: np.ndarray,
        torque_rate_limits: np.ndarray,
    ) -> None:
        self.translational_stiffness = _vector(
            translational_stiffness, 3, "translational_stiffness"
        ).copy()
        self.rotational_stiffness = _vector(
            rotational_stiffness, 3, "rotational_stiffness"
        ).copy()
        self.translational_damping = _vector(
            translational_damping, 3, "translational_damping"
        ).copy()
        self.rotational_damping = _vector(
            rotational_damping, 3, "rotational_damping"
        ).copy()
        self.torque_limits = _vector(
            torque_limits, JOINT_COUNT, "torque_limits"
        ).copy()
        self.torque_rate_limits = _vector(
            torque_rate_limits, JOINT_COUNT, "torque_rate_limits"
        ).copy()
        if np.any(
            np.concatenate(
                (
                    self.translational_stiffness,
                    self.rotational_stiffness,
                )
            )
            <= 0.0
        ):
            raise ValueError("Cartesian stiffness values must be positive")
        if np.any(
            np.concatenate(
                (
                    self.translational_damping,
                    self.rotational_damping,
                )
            )
            < 0.0
        ):
            raise ValueError("Cartesian damping values must be non-negative")
        if np.any(self.torque_limits <= 0.0):
            raise ValueError("torque_limits values must be positive")
        if np.any(self.torque_rate_limits <= 0.0):
            raise ValueError("torque_rate_limits values must be positive")
        self._stiffness = np.concatenate(
            (self.translational_stiffness, self.rotational_stiffness)
        )
        self._damping = np.concatenate(
            (self.translational_damping, self.rotational_damping)
        )
        self._previous_torque = np.zeros(JOINT_COUNT, dtype=float)

    @property
    def previous_torque(self) -> np.ndarray:
        return self._previous_torque.copy()

    def reset(self, previous_torque: np.ndarray | None = None) -> None:
        self._previous_torque = (
            np.zeros(JOINT_COUNT, dtype=float)
            if previous_torque is None
            else _vector(
                previous_torque, JOINT_COUNT, "previous_torque"
            ).copy()
        )

    def compute(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        target_linear_velocity: np.ndarray,
        target_angular_velocity: np.ndarray,
        jacobian: np.ndarray,
        dynamics_compensation: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, CartesianImpedanceDiagnostics]:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        position_value = _vector(position, 3, "position")
        current_rotation = validate_rotation_matrix(rotation, "rotation")
        linear_velocity_value = _vector(
            linear_velocity, 3, "linear_velocity"
        )
        angular_velocity_value = _vector(
            angular_velocity, 3, "angular_velocity"
        )
        target_position_value = _vector(
            target_position, 3, "target_position"
        )
        target_rotation_value = validate_rotation_matrix(
            target_rotation, "target_rotation"
        )
        target_linear_velocity_value = _vector(
            target_linear_velocity, 3, "target_linear_velocity"
        )
        target_angular_velocity_value = _vector(
            target_angular_velocity, 3, "target_angular_velocity"
        )
        jacobian_value = _jacobian(jacobian)
        compensation = _vector(
            dynamics_compensation, JOINT_COUNT, "dynamics_compensation"
        )

        position_error = target_position_value - position_value
        orientation_error = orientation_error_world(
            current_rotation, target_rotation_value
        )
        linear_velocity_error = (
            target_linear_velocity_value - linear_velocity_value
        )
        angular_velocity_error = (
            target_angular_velocity_value - angular_velocity_value
        )
        pose_error = np.concatenate((position_error, orientation_error))
        twist_error = np.concatenate(
            (linear_velocity_error, angular_velocity_error)
        )
        task_wrench = self._stiffness * pose_error + self._damping * twist_error
        task_torque = jacobian_value.T @ task_wrench
        raw = task_torque + compensation
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError(
                "Cartesian impedance computation produced NaN or Inf"
            )

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
            raise FloatingPointError("limited Cartesian torque contains NaN or Inf")
        self._previous_torque = final.copy()

        diagnostics = CartesianImpedanceDiagnostics(
            position_error=position_error.copy(),
            orientation_error=orientation_error.copy(),
            linear_velocity_error=linear_velocity_error.copy(),
            angular_velocity_error=angular_velocity_error.copy(),
            pose_error=pose_error.copy(),
            twist_error=twist_error.copy(),
            task_wrench=task_wrench.copy(),
            task_torque=task_torque.copy(),
            dynamics_compensation=compensation.copy(),
            raw_torque=raw.copy(),
            rate_limited_torque=rate_limited.copy(),
            final_torque=final.copy(),
            saturation_mask=saturation_mask.copy(),
            rate_limit_mask=rate_mask.copy(),
            finite=True,
        )
        return final.copy(), diagnostics
