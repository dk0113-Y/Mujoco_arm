from __future__ import annotations

from pathlib import Path
import unittest

from controllers import FixedDLSPickPlaceController
from environments import PandaUTableEnv
from evaluation import EpisodeResult, FailureReason


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class HeadlessSmokeTests(unittest.TestCase):
    def test_fixed_episode_finishes_with_structured_result(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        controller = FixedDLSPickPlaceController(env.config.controller)
        result = controller.run_episode(env, seed=42)
        self.assertIsInstance(result, EpisodeResult)
        self.assertIn("success", result.to_dict())
        self.assertLessEqual(
            result.simulation_time,
            env.config.simulation.episode_timeout + env.model.opt.timestep,
        )
        if not result.success:
            self.assertIn(result.failure_reason, {reason.value for reason in FailureReason})
        env.close()


if __name__ == "__main__":
    unittest.main()
