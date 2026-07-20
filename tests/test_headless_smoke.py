from __future__ import annotations

from pathlib import Path
import unittest

from controllers import FixedDLSPickPlaceController, SensorEventPickPlaceController
from environments import PandaUTableEnv, load_config
from evaluation import EpisodeResult, FailureReason
from perception import ColorDepthDetector, OverheadRGBDCamera, RGBDPerceptionProvider


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

    def test_b1_perception_episode_finishes_with_sensor_metrics(self) -> None:
        config = load_config(CONFIG_PATH).with_modes(
            controller_type="sensor_event_b1",
            observation_source="perception",
        )
        env = PandaUTableEnv(config)
        provider = RGBDPerceptionProvider(
            OverheadRGBDCamera(env.model, config.camera),
            env.data,
            ColorDepthDetector(config.perception),
        )
        controller = SensorEventPickPlaceController(config.controller, config.b1)
        result = controller.run_episode(env, seed=42, state_provider=provider)
        self.assertIsInstance(result, EpisodeResult)
        self.assertEqual(result.controller_type, "sensor_event_b1")
        self.assertEqual(result.observation_source, "perception")
        self.assertNotEqual(
            result.failure_reason, FailureReason.UNEXPECTED_EXCEPTION.value
        )
        self.assertIsNotNone(result.initial_perception_frame_count)
        self.assertIsNotNone(result.initial_valid_frame_count)
        self.assertIsNotNone(result.locked_target_position)
        self.assertIsNotNone(result.grasp_candidate)
        self.assertIsNotNone(result.grasp_confirmed)
        self.assertIsNotNone(result.controller_reported_success)
        self.assertIsNotNone(result.privileged_ground_truth_success)
        self.assertIsInstance(result.stage_durations, dict)
        if not result.success:
            b1_reasons = {
                reason.value
                for reason in FailureReason
                if reason.value != FailureReason.UNEXPECTED_EXCEPTION.value
            }
            self.assertIn(result.failure_reason, b1_reasons)
        provider.close()
        env.close()


if __name__ == "__main__":
    unittest.main()
