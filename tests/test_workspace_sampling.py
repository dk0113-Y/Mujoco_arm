from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from environments import PandaUTableEnv, load_config
from environments.randomization import sample_episode_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class WorkspaceSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_config(CONFIG_PATH)
        cls.config = base.with_modes(
            pick_mode="random", place_mode="random", physics_mode="random"
        )
        cls.env = PandaUTableEnv(cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def sample(self, seed: int):
        return sample_episode_parameters(
            np.random.default_rng(seed),
            self.config,
            self.env.workspace,
            seed=seed,
        )

    def test_fixed_seed_is_deterministic(self) -> None:
        self.assertEqual(self.sample(42), self.sample(42))

    def test_different_seeds_usually_differ(self) -> None:
        first = self.sample(42)
        second = self.sample(43)
        self.assertNotEqual(first.pick_position, second.pick_position)
        self.assertNotEqual(first.place_position, second.place_position)

    def test_1000_samples_stay_in_valid_geometry(self) -> None:
        rng = np.random.default_rng(20260719)
        for sample_index in range(1000):
            episode = sample_episode_parameters(
                rng,
                self.config,
                self.env.workspace,
                seed=sample_index,
            )
            pick_xy = np.asarray(episode.pick_position[:2])
            place_xy = np.asarray(episode.place_position[:2])
            self.assertTrue(
                self.env.workspace.region(episode.pick_region).contains_xy(
                    pick_xy, self.config.pick.edge_margin
                )
            )
            self.assertTrue(
                self.env.workspace.region(episode.place_region).contains_xy(
                    place_xy, self.config.place.edge_margin
                )
            )
            self.assertTrue(self.env.workspace.is_clear_of_base(pick_xy))
            self.assertTrue(self.env.workspace.is_clear_of_base(place_xy))
            self.assertGreaterEqual(
                float(np.linalg.norm(pick_xy - place_xy)),
                self.config.place.minimum_xy_distance,
            )


if __name__ == "__main__":
    unittest.main()
