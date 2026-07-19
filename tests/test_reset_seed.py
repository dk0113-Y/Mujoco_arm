from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from environments import PandaUTableEnv, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class ResetSeedTests(unittest.TestCase):
    def test_reset_seed_reproduces_task_and_state(self) -> None:
        config = load_config(CONFIG_PATH).with_modes(
            pick_mode="random", place_mode="random", physics_mode="random"
        )
        env = PandaUTableEnv(config)
        observation1, info1 = env.reset(seed=42)
        qpos1 = env.data.qpos.copy()
        qvel1 = env.data.qvel.copy()
        mass1 = float(env.model.body_mass[env.object_body_id])
        friction1 = env.model.geom_friction[env.object_geom_id].copy()

        observation2, info2 = env.reset(seed=42)
        np.testing.assert_array_equal(qpos1, env.data.qpos)
        np.testing.assert_array_equal(qvel1, env.data.qvel)
        np.testing.assert_array_equal(friction1, env.model.geom_friction[env.object_geom_id])
        self.assertEqual(mass1, float(env.model.body_mass[env.object_body_id]))
        self.assertEqual(info1, info2)
        for key in observation1:
            np.testing.assert_array_equal(
                np.asarray(observation1[key]), np.asarray(observation2[key])
            )
        self.assertTrue(np.all(np.isfinite(env.data.qpos)))
        self.assertTrue(np.all(np.isfinite(env.data.qvel)))
        self.assertTrue(
            np.all(np.isfinite(env.data.xpos[env.object_body_id]))
        )
        self.assertTrue(
            np.all(np.isfinite(env.data.site_xpos[env.place_target_site_id]))
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
