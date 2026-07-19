from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from environments import PandaUTableEnv
from perception.camera_geometry import (
    ProjectionError,
    camera_to_pixel,
    extrinsics_from_mujoco,
    intrinsics_from_fovy,
    pixel_depth_to_world,
    world_to_camera,
    world_to_pixel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class CameraGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = PandaUTableEnv(CONFIG_PATH)
        cls.env.reset(seed=42)
        cls.extrinsics = extrinsics_from_mujoco(
            cls.env.model, cls.env.data, cls.env.overhead_camera_id
        )
        cls.intrinsics = intrinsics_from_fovy(
            width=cls.env.config.camera.width,
            height=cls.env.config.camera.height,
            fovy_degrees=cls.env.config.camera.fovy,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def test_multiple_world_projection_round_trips(self) -> None:
        points = (
            np.array([0.50, 0.12, 0.27]),
            np.array([0.50, -0.20, 0.224]),
            np.array([-0.20, 0.57, 0.22]),
            np.array([0.70, -0.40, 0.22]),
        )
        for point in points:
            with self.subTest(point=point.tolist()):
                u, v, depth = world_to_pixel(
                    point, self.intrinsics, self.extrinsics
                )
                reconstructed = pixel_depth_to_world(
                    (u, v), depth, self.intrinsics, self.extrinsics
                )
                self.assertLess(float(np.linalg.norm(point - reconstructed)), 1e-9)

    def test_camera_axis_and_image_v_signs(self) -> None:
        center_u, center_v, _ = camera_to_pixel(
            np.array([0.0, 0.0, -1.0]), self.intrinsics
        )
        right_u, _, _ = camera_to_pixel(
            np.array([0.1, 0.0, -1.0]), self.intrinsics
        )
        _, up_v, _ = camera_to_pixel(
            np.array([0.0, 0.1, -1.0]), self.intrinsics
        )
        self.assertGreater(right_u, center_u)
        self.assertLess(up_v, center_v)
        with self.assertRaisesRegex(ProjectionError, "behind"):
            camera_to_pixel(np.array([0.0, 0.0, 1.0]), self.intrinsics)

    def test_invalid_depth_and_image_edge_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectionError, "positive"):
            pixel_depth_to_world(
                (10.0, 10.0), 0.0, self.intrinsics, self.extrinsics
            )
        with self.assertRaisesRegex(ProjectionError, "outside"):
            pixel_depth_to_world(
                (-1.0, 10.0), 1.0, self.intrinsics, self.extrinsics
            )

    def test_extrinsic_matrices_are_inverses(self) -> None:
        np.testing.assert_allclose(
            self.extrinsics.camera_to_world @ self.extrinsics.world_to_camera,
            np.eye(4),
            atol=1e-10,
        )
        camera_point = world_to_camera(
            np.array([0.5, 0.0, 0.22]), self.extrinsics
        )
        self.assertLess(camera_point[2], 0.0)


if __name__ == "__main__":
    unittest.main()
