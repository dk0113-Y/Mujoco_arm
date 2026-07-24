from __future__ import annotations

import unittest

import numpy as np

from control_benchmarks.trajectories import (
    HoldTrajectory,
    MinimumJerkTrajectory,
    MultiJointSmoothTrajectory,
    SingleJointSineTrajectory,
)


class ControlTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, -0.7])

    def test_hold_is_constant_with_zero_derivatives(self) -> None:
        trajectory = HoldTrajectory(self.base, 1.0)
        for time in (0.0, 0.4, 1.0):
            sample = trajectory.sample(time)
            np.testing.assert_array_equal(sample.q, self.base)
            np.testing.assert_array_equal(sample.dq, 0.0)
            np.testing.assert_array_equal(sample.ddq, 0.0)

    def test_minimum_jerk_start_boundary(self) -> None:
        goal = self.base + 0.1
        sample = MinimumJerkTrajectory(self.base, goal, 2.0).sample(0.0)
        np.testing.assert_array_equal(sample.q, self.base)
        np.testing.assert_array_equal(sample.dq, 0.0)
        np.testing.assert_array_equal(sample.ddq, 0.0)

    def test_minimum_jerk_end_boundary(self) -> None:
        goal = self.base + 0.1
        sample = MinimumJerkTrajectory(self.base, goal, 2.0).sample(2.0)
        np.testing.assert_array_equal(sample.q, goal)
        np.testing.assert_array_equal(sample.dq, 0.0)
        np.testing.assert_array_equal(sample.ddq, 0.0)

    def test_minimum_jerk_midpoint_is_half_displacement(self) -> None:
        goal = self.base + 0.2
        sample = MinimumJerkTrajectory(self.base, goal, 2.0).sample(1.0)
        np.testing.assert_allclose(sample.q, self.base + 0.1)

    def test_single_joint_sine_moves_only_selected_joint(self) -> None:
        trajectory = SingleJointSineTrajectory(
            self.base,
            joint_index=3,
            amplitude=0.1,
            frequency_hz=0.25,
            duration=3.0,
            ramp_duration=0.5,
        )
        sample = trajectory.sample(1.0)
        stationary = [0, 1, 2, 4, 5, 6]
        np.testing.assert_array_equal(sample.q[stationary], self.base[stationary])
        np.testing.assert_array_equal(sample.dq[stationary], 0.0)
        self.assertNotEqual(float(sample.q[3]), float(self.base[3]))

    def test_sine_window_has_smooth_zero_boundaries(self) -> None:
        trajectory = SingleJointSineTrajectory(
            self.base,
            joint_index=1,
            amplitude=0.1,
            frequency_hz=0.25,
            duration=3.0,
            ramp_duration=0.5,
        )
        for time in (0.0, 3.0):
            sample = trajectory.sample(time)
            np.testing.assert_array_equal(sample.q, self.base)
            np.testing.assert_array_equal(sample.dq, 0.0)
            np.testing.assert_array_equal(sample.ddq, 0.0)

    def test_multi_joint_trajectory_is_bounded_and_repeatable(self) -> None:
        amplitudes = np.linspace(0.02, 0.08, 7)
        trajectory = MultiJointSmoothTrajectory(
            self.base,
            amplitudes=amplitudes,
            frequencies_hz=np.linspace(0.1, 0.2, 7),
            phases=np.linspace(0.0, 1.0, 7),
            duration=3.0,
            ramp_duration=0.5,
        )
        for time in np.linspace(0.0, 3.0, 31):
            first = trajectory.sample(float(time))
            second = trajectory.sample(float(time))
            np.testing.assert_array_equal(first.q, second.q)
            self.assertTrue(np.all(np.abs(first.q - self.base) <= amplitudes + 1e-12))


if __name__ == "__main__":
    unittest.main()
