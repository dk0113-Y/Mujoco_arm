from __future__ import annotations

import unittest

import numpy as np

from control_benchmarks.cartesian_trajectories import (
    AxisTranslationTrajectory,
    CartesianHoldTrajectory,
    CircleTrajectory,
    OrientationAxisTrajectory,
    StraightLineTrajectory,
    validate_workspace,
)
from control_benchmarks.kinematics import orientation_error_world


class CartesianTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.position = np.asarray([0.5, 0.0, 0.55])
        self.rotation = np.eye(3)

    def test_hold_is_constant_with_zero_twist(self) -> None:
        trajectory = CartesianHoldTrajectory(
            self.position, self.rotation, 1.0
        )
        for time in (0.0, 0.4, 1.0):
            sample = trajectory.sample(time)
            np.testing.assert_array_equal(sample.position, self.position)
            np.testing.assert_array_equal(sample.rotation, self.rotation)
            np.testing.assert_array_equal(sample.linear_velocity, 0.0)
            np.testing.assert_array_equal(sample.angular_velocity, 0.0)

    def test_translation_axis_isolation_and_smooth_boundaries(self) -> None:
        for axis in range(3):
            trajectory = AxisTranslationTrajectory(
                self.position,
                self.rotation,
                axis=axis,
                amplitude=0.02,
                frequency_hz=0.25,
                duration=4.0,
                ramp_duration=0.5,
            )
            stationary = [value for value in range(3) if value != axis]
            midpoint = trajectory.sample(1.0)
            np.testing.assert_array_equal(
                midpoint.position[stationary],
                self.position[stationary],
            )
            for time in (0.0, 4.0):
                sample = trajectory.sample(time)
                np.testing.assert_allclose(sample.position, self.position)
                np.testing.assert_allclose(sample.linear_velocity, 0.0)

    def test_orientation_axis_uses_so3_and_isolates_world_axis(self) -> None:
        trajectory = OrientationAxisTrajectory(
            self.position,
            self.rotation,
            axis=1,
            amplitude=0.1,
            frequency_hz=0.25,
            duration=4.0,
            ramp_duration=0.5,
        )
        sample = trajectory.sample(1.0)
        error = orientation_error_world(self.rotation, sample.rotation)
        self.assertAlmostEqual(error[0], 0.0, places=14)
        self.assertGreater(error[1], 0.0)
        self.assertAlmostEqual(error[2], 0.0, places=14)
        for time in (0.0, 4.0):
            boundary = trajectory.sample(time)
            np.testing.assert_allclose(boundary.rotation, self.rotation)
            np.testing.assert_allclose(boundary.angular_velocity, 0.0)

    def test_straight_line_minimum_jerk_boundaries(self) -> None:
        displacement = np.asarray([0.04, 0.02, 0.03])
        trajectory = StraightLineTrajectory(
            self.position,
            self.rotation,
            displacement=displacement,
            duration=3.0,
        )
        start = trajectory.sample(0.0)
        end = trajectory.sample(3.0)
        midpoint = trajectory.sample(1.5)
        np.testing.assert_allclose(start.position, self.position)
        np.testing.assert_allclose(end.position, self.position + displacement)
        np.testing.assert_allclose(
            midpoint.position, self.position + 0.5 * displacement
        )
        np.testing.assert_allclose(start.linear_velocity, 0.0)
        np.testing.assert_allclose(end.linear_velocity, 0.0)

    def test_circle_closes_with_zero_endpoint_velocity(self) -> None:
        trajectory = CircleTrajectory(
            self.position,
            self.rotation,
            radius=0.025,
            plane_axes=(0, 1),
            duration=4.0,
        )
        start = trajectory.sample(0.0)
        end = trajectory.sample(4.0)
        np.testing.assert_allclose(start.position, self.position, atol=1e-15)
        np.testing.assert_allclose(end.position, self.position, atol=1e-14)
        np.testing.assert_allclose(start.linear_velocity, 0.0)
        np.testing.assert_allclose(end.linear_velocity, 0.0, atol=1e-14)
        center = self.position - np.asarray([0.025, 0.0, 0.0])
        for time in np.linspace(0.0, 4.0, 31):
            sample = trajectory.sample(float(time))
            radius = np.linalg.norm((sample.position - center)[:2])
            self.assertAlmostEqual(float(radius), 0.025, places=14)

    def test_trajectories_are_repeatable(self) -> None:
        trajectory = CircleTrajectory(
            self.position,
            self.rotation,
            radius=0.025,
            plane_axes=(0, 2),
            duration=4.0,
        )
        for time in np.linspace(0.0, 4.0, 21):
            first = trajectory.sample(float(time))
            second = trajectory.sample(float(time))
            np.testing.assert_array_equal(first.position, second.position)
            np.testing.assert_array_equal(
                first.linear_velocity, second.linear_velocity
            )

    def test_workspace_precheck_accepts_safe_and_rejects_unsafe(self) -> None:
        safe = StraightLineTrajectory(
            self.position,
            self.rotation,
            displacement=np.asarray([0.04, 0.02, 0.03]),
            duration=3.0,
        )
        validate_workspace(
            safe,
            workspace_min=np.asarray([0.2, -0.5, 0.25]),
            workspace_max=np.asarray([0.75, 0.5, 0.9]),
        )
        unsafe = StraightLineTrajectory(
            self.position,
            self.rotation,
            displacement=np.asarray([1.0, 0.0, 0.0]),
            duration=3.0,
        )
        with self.assertRaisesRegex(ValueError, "safe workspace"):
            validate_workspace(
                unsafe,
                workspace_min=np.asarray([0.2, -0.5, 0.25]),
                workspace_max=np.asarray([0.75, 0.5, 0.9]),
            )


if __name__ == "__main__":
    unittest.main()
