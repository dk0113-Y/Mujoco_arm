"""Deterministic task-space trajectories for CI-Baseline v1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import (
    rotation_vector_to_matrix,
    validate_rotation_matrix,
)
from .trajectories import _minimum_jerk_scaling, _smooth_window


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _positive_duration(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("duration must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class CartesianTrajectorySample:
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


class CartesianHoldTrajectory:
    def __init__(
        self, position: np.ndarray, rotation: np.ndarray, duration: float
    ) -> None:
        self.position = _vector3(position, "position").copy()
        self.rotation = validate_rotation_matrix(
            rotation, "rotation"
        ).copy()
        self.duration = _positive_duration(duration)

    def sample(self, time: float) -> CartesianTrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        return CartesianTrajectorySample(
            position=self.position.copy(),
            rotation=self.rotation.copy(),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        )


class AxisTranslationTrajectory:
    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        axis: int,
        amplitude: float,
        frequency_hz: float,
        duration: float,
        ramp_duration: float,
    ) -> None:
        self.position = _vector3(position, "position").copy()
        self.rotation = validate_rotation_matrix(
            rotation, "rotation"
        ).copy()
        if isinstance(axis, bool) or int(axis) not in (0, 1, 2):
            raise ValueError("axis must be one of 0, 1, 2")
        values = (amplitude, frequency_hz, duration, ramp_duration)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("translation trajectory parameters must be finite")
        if amplitude <= 0.0 or frequency_hz <= 0.0:
            raise ValueError("amplitude and frequency_hz must be positive")
        if duration <= 0.0 or ramp_duration <= 0.0:
            raise ValueError("duration and ramp_duration must be positive")
        if 2.0 * ramp_duration > duration:
            raise ValueError("ramp_duration must be at most duration/2")
        self.axis = int(axis)
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)

    def sample(self, time: float) -> CartesianTrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        window, dwindow, _ = _smooth_window(
            float(time), self.duration, self.ramp_duration
        )
        omega = 2.0 * np.pi * self.frequency_hz
        sine = np.sin(omega * float(time))
        cosine = np.cos(omega * float(time))
        offset = self.amplitude * window * sine
        velocity = self.amplitude * (
            dwindow * sine + window * omega * cosine
        )
        position = self.position.copy()
        linear_velocity = np.zeros(3, dtype=float)
        position[self.axis] += offset
        linear_velocity[self.axis] = velocity
        return CartesianTrajectorySample(
            position=position,
            rotation=self.rotation.copy(),
            linear_velocity=linear_velocity,
            angular_velocity=np.zeros(3, dtype=float),
        )


class OrientationAxisTrajectory:
    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        axis: int,
        amplitude: float,
        frequency_hz: float,
        duration: float,
        ramp_duration: float,
    ) -> None:
        self.position = _vector3(position, "position").copy()
        self.rotation = validate_rotation_matrix(
            rotation, "rotation"
        ).copy()
        if isinstance(axis, bool) or int(axis) not in (0, 1, 2):
            raise ValueError("axis must be one of 0, 1, 2")
        values = (amplitude, frequency_hz, duration, ramp_duration)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("orientation trajectory parameters must be finite")
        if amplitude <= 0.0 or frequency_hz <= 0.0:
            raise ValueError("amplitude and frequency_hz must be positive")
        if duration <= 0.0 or ramp_duration <= 0.0:
            raise ValueError("duration and ramp_duration must be positive")
        if 2.0 * ramp_duration > duration:
            raise ValueError("ramp_duration must be at most duration/2")
        self.axis = int(axis)
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)
        self.axis_vector = np.eye(3, dtype=float)[self.axis]

    def sample(self, time: float) -> CartesianTrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        window, dwindow, _ = _smooth_window(
            float(time), self.duration, self.ramp_duration
        )
        omega = 2.0 * np.pi * self.frequency_hz
        sine = np.sin(omega * float(time))
        cosine = np.cos(omega * float(time))
        angle = self.amplitude * window * sine
        angular_speed = self.amplitude * (
            dwindow * sine + window * omega * cosine
        )
        target_rotation = (
            rotation_vector_to_matrix(self.axis_vector * angle) @ self.rotation
        )
        return CartesianTrajectorySample(
            position=self.position.copy(),
            rotation=target_rotation,
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=self.axis_vector * angular_speed,
        )


class StraightLineTrajectory:
    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        displacement: np.ndarray,
        duration: float,
    ) -> None:
        self.position = _vector3(position, "position").copy()
        self.rotation = validate_rotation_matrix(
            rotation, "rotation"
        ).copy()
        self.displacement = _vector3(displacement, "displacement").copy()
        self.duration = _positive_duration(duration)

    def sample(self, time: float) -> CartesianTrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        scale, velocity, _ = _minimum_jerk_scaling(
            float(time), self.duration
        )
        return CartesianTrajectorySample(
            position=self.position + scale * self.displacement,
            rotation=self.rotation.copy(),
            linear_velocity=velocity * self.displacement,
            angular_velocity=np.zeros(3, dtype=float),
        )


class CircleTrajectory:
    """Closed fixed-orientation circle with zero endpoint velocity."""

    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        *,
        radius: float,
        plane_axes: tuple[int, int],
        duration: float,
    ) -> None:
        self.position = _vector3(position, "position").copy()
        self.rotation = validate_rotation_matrix(
            rotation, "rotation"
        ).copy()
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if (
            len(set(plane_axes)) != 2
            or any(axis not in (0, 1, 2) for axis in plane_axes)
        ):
            raise ValueError("plane_axes must contain two distinct xyz indices")
        self.radius = float(radius)
        self.duration = _positive_duration(duration)
        self.first_axis = np.eye(3, dtype=float)[plane_axes[0]]
        self.second_axis = np.eye(3, dtype=float)[plane_axes[1]]

    def sample(self, time: float) -> CartesianTrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        scale, scale_rate, _ = _minimum_jerk_scaling(
            float(time), self.duration
        )
        phase = 2.0 * np.pi * scale
        phase_rate = 2.0 * np.pi * scale_rate
        offset = self.radius * (
            (np.cos(phase) - 1.0) * self.first_axis
            + np.sin(phase) * self.second_axis
        )
        velocity = self.radius * phase_rate * (
            -np.sin(phase) * self.first_axis
            + np.cos(phase) * self.second_axis
        )
        return CartesianTrajectorySample(
            position=self.position + offset,
            rotation=self.rotation.copy(),
            linear_velocity=velocity,
            angular_velocity=np.zeros(3, dtype=float),
        )


def validate_workspace(
    trajectory: object,
    *,
    workspace_min: np.ndarray,
    workspace_max: np.ndarray,
    sample_count: int = 201,
) -> None:
    lower = _vector3(workspace_min, "workspace_min")
    upper = _vector3(workspace_max, "workspace_max")
    if np.any(lower >= upper):
        raise ValueError("workspace bounds must be strictly ordered")
    duration = float(getattr(trajectory, "duration"))
    for time in np.linspace(0.0, duration, sample_count):
        sample = trajectory.sample(float(time))
        if np.any(sample.position < lower) or np.any(sample.position > upper):
            raise ValueError(
                "Cartesian trajectory leaves the configured safe workspace"
            )
        validate_rotation_matrix(sample.rotation, "trajectory rotation")
