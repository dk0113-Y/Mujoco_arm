from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from environments import PandaUTableEnv
from perception.camera_geometry import (
    extrinsics_from_mujoco,
    intrinsics_from_fovy,
    world_to_pixel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"
SCENE_PATH = PROJECT_ROOT / "scenes" / "panda_u_table_scene.xml"


class CameraModelTests(unittest.TestCase):
    def test_camera_is_fixed_worldbody_camera_and_covers_workspace(self) -> None:
        scene_worldbody = ET.parse(SCENE_PATH).getroot().find("worldbody")
        self.assertIsNotNone(scene_worldbody)
        direct_camera = scene_worldbody.find("camera[@name='overhead_rgbd']")
        self.assertIsNotNone(direct_camera)

        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        camera_id = env.overhead_camera_id
        self.assertEqual(int(env.model.cam_bodyid[camera_id]), 0)
        self.assertEqual(env.config.camera.width, 512)
        self.assertEqual(env.config.camera.height, 512)
        self.assertGreater(float(env.model.cam_fovy[camera_id]), 0.0)
        extrinsics = extrinsics_from_mujoco(env.model, env.data, camera_id)
        intrinsics = intrinsics_from_fovy(
            width=env.config.camera.width,
            height=env.config.camera.height,
            fovy_degrees=float(env.model.cam_fovy[camera_id]),
        )
        look_direction = -extrinsics.rotation_world_from_camera[:, 2]
        self.assertLess(look_direction[2], -0.9)

        minimum_margin = float("inf")
        for region in env.workspace.regions.values():
            min_x, max_x, min_y, max_y = region.bounds(
                env.config.pick.edge_margin
            )
            for x in (min_x, max_x):
                for y in (min_y, max_y):
                    u, v, _ = world_to_pixel(
                        np.array([x, y, region.top_z]), intrinsics, extrinsics
                    )
                    minimum_margin = min(
                        minimum_margin,
                        u,
                        v,
                        intrinsics.width - 1 - u,
                        intrinsics.height - 1 - v,
                    )
        self.assertGreater(minimum_margin, 80.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
