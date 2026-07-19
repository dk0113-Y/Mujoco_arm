from __future__ import annotations

from pathlib import Path
import unittest

import mujoco

from environments import PandaUTableEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class ModelLoadingTests(unittest.TestCase):
    def test_required_model_objects_exist(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        required = {
            mujoco.mjtObj.mjOBJ_BODY: (
                "link0",
                "hand",
                "u_table_front",
                "u_table_left",
                "u_table_right",
                "pick_object",
            ),
            mujoco.mjtObj.mjOBJ_GEOM: (
                "u_table_front_geom",
                "u_table_left_geom",
                "u_table_right_geom",
                "pick_object_geom",
            ),
            mujoco.mjtObj.mjOBJ_SITE: ("place_target", "gripper_tcp"),
            mujoco.mjtObj.mjOBJ_CAMERA: ("overhead_rgbd",),
            mujoco.mjtObj.mjOBJ_JOINT: (
                *(f"joint{index}" for index in range(1, 8)),
                "finger_joint1",
                "finger_joint2",
                "pick_object_free_joint",
            ),
            mujoco.mjtObj.mjOBJ_ACTUATOR: tuple(
                f"actuator{index}" for index in range(1, 9)
            ),
        }
        for object_type, names in required.items():
            for name in names:
                with self.subTest(name=name):
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(env.model, object_type, name), 0
                    )
        self.assertEqual(len(env.table_geom_ids), 3)
        env.close()


if __name__ == "__main__":
    unittest.main()
