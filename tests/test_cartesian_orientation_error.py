from __future__ import annotations

import unittest

import numpy as np

from control_benchmarks.kinematics import (
    normalize_quaternion_wxyz,
    orientation_error_world,
    quaternion_wxyz_to_rotation,
    rotation_to_quaternion_wxyz,
    rotation_vector_to_matrix,
)


class CartesianOrientationErrorTests(unittest.TestCase):
    def test_zero_orientation_error(self) -> None:
        rotation = rotation_vector_to_matrix(np.asarray([0.2, -0.1, 0.3]))
        np.testing.assert_allclose(
            orientation_error_world(rotation, rotation), 0.0, atol=1e-15
        )

    def test_world_axis_signs_for_small_positive_and_negative_angles(self) -> None:
        identity = np.eye(3)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                vector = np.zeros(3)
                vector[axis] = sign * 0.02
                error = orientation_error_world(
                    identity, rotation_vector_to_matrix(vector)
                )
                np.testing.assert_allclose(error, vector, atol=1e-14)

    def test_quaternion_hemisphere_is_canonical(self) -> None:
        quaternion = np.asarray([-0.4, 0.2, -0.3, 0.5])
        first = normalize_quaternion_wxyz(quaternion)
        second = normalize_quaternion_wxyz(-quaternion)
        np.testing.assert_allclose(first, second, atol=1e-15)
        self.assertGreaterEqual(first[0], 0.0)

    def test_quaternion_sign_flip_gives_identical_rotation_and_error(self) -> None:
        target = rotation_vector_to_matrix(np.asarray([0.1, -0.2, 0.05]))
        quaternion = rotation_to_quaternion_wxyz(target)
        positive_rotation = quaternion_wxyz_to_rotation(quaternion)
        negative_rotation = quaternion_wxyz_to_rotation(-quaternion)
        np.testing.assert_allclose(positive_rotation, negative_rotation)
        np.testing.assert_allclose(
            orientation_error_world(np.eye(3), positive_rotation),
            orientation_error_world(np.eye(3), negative_rotation),
        )

    def test_near_pi_error_is_finite_and_has_geodesic_magnitude(self) -> None:
        target = rotation_vector_to_matrix(
            np.asarray([np.pi - 1e-9, 0.0, 0.0])
        )
        error = orientation_error_world(np.eye(3), target)
        self.assertTrue(np.all(np.isfinite(error)))
        self.assertAlmostEqual(
            float(np.linalg.norm(error)), np.pi - 1e-9, places=8
        )

    def test_error_is_expressed_in_world_frame(self) -> None:
        current = rotation_vector_to_matrix(np.asarray([0.0, 0.0, 0.7]))
        world_delta = rotation_vector_to_matrix(np.asarray([0.03, 0.0, 0.0]))
        target = world_delta @ current
        error = orientation_error_world(current, target)
        np.testing.assert_allclose(error, [0.03, 0.0, 0.0], atol=1e-14)

    def test_invalid_zero_quaternion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "norm"):
            normalize_quaternion_wxyz(np.zeros(4))


if __name__ == "__main__":
    unittest.main()
