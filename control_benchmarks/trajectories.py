from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_COUNT = 7


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (JOINT_COUNT,):
        raise ValueError(f"{name} must have shape ({JOINT_COUNT},), got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _minimum_jerk_scaling(
    elapsed: float, duration: float
) -> tuple[float, float, float]:
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if elapsed <= 0.0:
        return 0.0, 0.0, 0.0
    if elapsed >= duration:
        return 1.0, 0.0, 0.0
    u = float(elapsed / duration)
    scale = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    velocity = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
    acceleration = (
        60.0 * u - 180.0 * u**2 + 120.0 * u**3
    ) / duration**2
    return scale, velocity, acceleration


@dataclass(frozen=True)
class TrajectorySample:
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray


class HoldTrajectory:
    def __init__(self, pose: np.ndarray, duration: float) -> None:
        self.pose = _vector(pose, "pose").copy()
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        self.duration = float(duration)

    def sample(self, time: float) -> TrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        return TrajectorySample(
            q=self.pose.copy(),
            dq=np.zeros(JOINT_COUNT, dtype=float),
            ddq=np.zeros(JOINT_COUNT, dtype=float),
        )


class MinimumJerkTrajectory:
    def __init__(
        self, start: np.ndarray, goal: np.ndarray, duration: float
    ) -> None:
        self.start = _vector(start, "start").copy()
        self.goal = _vector(goal, "goal").copy()
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        self.duration = float(duration)

    def sample(self, time: float) -> TrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        scale, velocity, acceleration = _minimum_jerk_scaling(
            float(time), self.duration
        )
        displacement = self.goal - self.start
        return TrajectorySample(
            q=self.start + scale * displacement,
            dq=velocity * displacement,
            ddq=acceleration * displacement,
        )


def _smooth_window(
    time: float, duration: float, ramp_duration: float
) -> tuple[float, float, float]:
    if time <= 0.0 or time >= duration:
        return 0.0, 0.0, 0.0
    if time < ramp_duration:
        return _minimum_jerk_scaling(time, ramp_duration)
    if time > duration - ramp_duration:
        scale, velocity, acceleration = _minimum_jerk_scaling(
            duration - time, ramp_duration
        )
        return scale, -velocity, acceleration
    return 1.0, 0.0, 0.0


class SingleJointSineTrajectory:
    def __init__(
        self,
        base_pose: np.ndarray,
        *,
        joint_index: int,
        amplitude: float,
        frequency_hz: float,
        duration: float,
        ramp_duration: float,
    ) -> None:
        self.base_pose = _vector(base_pose, "base_pose").copy()
        if isinstance(joint_index, bool) or not 0 <= int(joint_index) < JOINT_COUNT:
            raise ValueError("joint_index must be an integer in [0, 6]")
        values = (amplitude, frequency_hz, duration, ramp_duration)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("sine parameters must be finite")
        if amplitude < 0.0 or frequency_hz <= 0.0 or duration <= 0.0:
            raise ValueError("sine amplitude must be non-negative; frequency/duration positive")
        if ramp_duration <= 0.0 or 2.0 * ramp_duration > duration:
            raise ValueError("ramp_duration must be positive and at most duration/2")
        self.joint_index = int(joint_index)
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)

    def sample(self, time: float) -> TrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        window, dwindow, ddwindow = _smooth_window(
            float(time), self.duration, self.ramp_duration
        )
        omega = 2.0 * np.pi * self.frequency_hz
        sine = np.sin(omega * float(time))
        cosine = np.cos(omega * float(time))
        offset = self.amplitude * window * sine
        velocity = self.amplitude * (dwindow * sine + window * omega * cosine)
        acceleration = self.amplitude * (
            ddwindow * sine
            + 2.0 * dwindow * omega * cosine
            - window * omega**2 * sine
        )
        q = self.base_pose.copy()
        dq = np.zeros(JOINT_COUNT, dtype=float)
        ddq = np.zeros(JOINT_COUNT, dtype=float)
        q[self.joint_index] += offset
        dq[self.joint_index] = velocity
        ddq[self.joint_index] = acceleration
        return TrajectorySample(q=q, dq=dq, ddq=ddq)


class MultiJointSmoothTrajectory:
    def __init__(
        self,
        base_pose: np.ndarray,
        *,
        amplitudes: np.ndarray,
        frequencies_hz: np.ndarray,
        phases: np.ndarray,
        duration: float,
        ramp_duration: float,
    ) -> None:
        self.base_pose = _vector(base_pose, "base_pose").copy()
        self.amplitudes = _vector(amplitudes, "amplitudes").copy()
        self.frequencies_hz = _vector(
            frequencies_hz, "frequencies_hz"
        ).copy()
        self.phases = _vector(phases, "phases").copy()
        if np.any(self.amplitudes < 0.0):
            raise ValueError("amplitudes must be non-negative")
        if np.any(self.frequencies_hz <= 0.0):
            raise ValueError("frequencies_hz must be positive")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and positive")
        if (
            not np.isfinite(ramp_duration)
            or ramp_duration <= 0.0
            or 2.0 * ramp_duration > duration
        ):
            raise ValueError("ramp_duration must be positive and at most duration/2")
        self.duration = float(duration)
        self.ramp_duration = float(ramp_duration)

    def sample(self, time: float) -> TrajectorySample:
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        window, dwindow, ddwindow = _smooth_window(
            float(time), self.duration, self.ramp_duration
        )
        omega = 2.0 * np.pi * self.frequencies_hz
        angle = omega * float(time) + self.phases
        sine = np.sin(angle)
        cosine = np.cos(angle)
        offset = self.amplitudes * window * sine
        velocity = self.amplitudes * (
            dwindow * sine + window * omega * cosine
        )
        acceleration = self.amplitudes * (
            ddwindow * sine
            + 2.0 * dwindow * omega * cosine
            - window * omega**2 * sine
        )
        return TrajectorySample(
            q=self.base_pose + offset,
            dq=velocity,
            ddq=acceleration,
        )
