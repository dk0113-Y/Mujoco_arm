from __future__ import annotations

from pathlib import Path
import unittest

from controllers import FixedDLSPickPlaceController
from environments import PandaUTableEnv, load_config
from evaluation import EpisodeResult, FailureReason
from perception import ColorDepthDetector, OverheadRGBDCamera, RGBDPerceptionProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class PerceptionEpisodeTests(unittest.TestCase):
    def test_seed_42_perception_episode_ends_structurally(self) -> None:
        config = load_config(CONFIG_PATH).with_modes(observation_source="perception")
        env = PandaUTableEnv(config)
        observation, _ = env.reset(seed=42)
        self.assertNotIn("privileged_object_position", observation)
        self.assertNotIn("privileged_place_target_position", observation)
        provider = RGBDPerceptionProvider(
            OverheadRGBDCamera(env.model, config.camera),
            env.data,
            ColorDepthDetector(config.perception),
        )
        controller = FixedDLSPickPlaceController(config.controller)
        result = controller.run_episode(env, seed=42, state_provider=provider)
        self.assertIsInstance(result, EpisodeResult)
        self.assertEqual(result.observation_source, "perception")
        self.assertIsNotNone(result.estimated_object_position)
        self.assertIsNotNone(result.estimated_target_position)
        self.assertIsNotNone(result.object_position_error)
        self.assertIsNotNone(result.target_position_error)
        self.assertNotEqual(
            result.failure_reason, FailureReason.UNEXPECTED_EXCEPTION.value
        )
        if not result.success:
            self.assertIn(result.failure_reason, {reason.value for reason in FailureReason})
        provider.close()
        env.close()


if __name__ == "__main__":
    unittest.main()
