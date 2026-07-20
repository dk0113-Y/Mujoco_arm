from __future__ import annotations

from pathlib import Path
import unittest

import mujoco

from environments import PandaUTableEnv, load_config
from evaluation.episode_result import FailureReason
from evaluation.perception_evaluator import EpisodeOutcome, build_episode_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class B1EpisodeResultTests(unittest.TestCase):
    def make_env(self) -> PandaUTableEnv:
        config = load_config(CONFIG_PATH).with_modes(observation_source="perception")
        env = PandaUTableEnv(config)
        env.reset(seed=42)
        return env

    def test_controller_success_vs_ground_truth_sets_false_positive(self) -> None:
        env = self.make_env()
        metrics = {
            "initial_perception_frame_count": 5,
            "initial_valid_frame_count": 5,
            "initial_object_position": (0.50, 0.12, 0.245),
            "locked_target_position": (0.50, -0.20, 0.222),
            "grasp_candidate": True,
            "trial_lift_completed": True,
            "grasp_confirmed": True,
            "final_visual_object_position": (0.50, -0.20, 0.245),
            "final_visual_xy_error": 0.0,
            "final_visual_height_error": 0.0,
            "stage_durations": {"scene_perception": 0.01},
        }
        result = build_episode_result(
            env,
            EpisodeOutcome(
                success=True,
                failure_reason=None,
                stage="completed",
                lift_height=0.04,
                exception_message=None,
                key_errors={},
                controller_type="sensor_event_b1",
                b1_metrics=metrics,
            ),
            None,
            None,
        )
        self.assertTrue(result.controller_reported_success)
        self.assertFalse(result.privileged_ground_truth_success)
        self.assertTrue(result.false_positive)
        self.assertFalse(result.false_negative)
        self.assertTrue(result.perception_success)
        self.assertEqual(result.estimated_target_position, metrics["locked_target_position"])
        self.assertIn("final_visual_xy_error", result.to_dict())
        self.assertIn('"controller_type": "sensor_event_b1"', result.to_json())
        csv_row = result.to_flat_dict()
        self.assertIn("stage_duration.scene_perception", csv_row)
        self.assertFalse(
            any(isinstance(value, (dict, list, tuple)) for value in csv_row.values())
        )
        env.close()

    def test_controller_failure_vs_ground_truth_sets_false_negative(self) -> None:
        env = self.make_env()
        address = env.object_qpos_address
        env.data.qpos[address : address + 7] = (
            0.50,
            -0.20,
            0.245,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        mujoco.mj_forward(env.model, env.data)
        result = build_episode_result(
            env,
            EpisodeOutcome(
                success=False,
                failure_reason=None,
                stage="final_visual_verification",
                lift_height=0.04,
                exception_message=None,
                key_errors={},
                controller_type="sensor_event_b1",
                b1_metrics={},
            ),
            None,
            None,
        )
        self.assertFalse(result.controller_reported_success)
        self.assertTrue(result.privileged_ground_truth_success)
        self.assertFalse(result.false_positive)
        self.assertTrue(result.false_negative)
        env.close()

    def test_b1_perception_stage_failures_are_reported_consistently(self) -> None:
        env = self.make_env()
        metrics = {
            "initial_object_position": (0.50, 0.12, 0.245),
            "locked_target_position": (0.50, -0.20, 0.222),
        }
        reasons = (
            FailureReason.INITIAL_PERCEPTION_FAILED,
            FailureReason.PREGRASP_REACQUISITION_FAILED,
            FailureReason.PREGRASP_POSITION_UNSTABLE,
            FailureReason.FINAL_OBJECT_NOT_FOUND,
            FailureReason.FINAL_VISUAL_PLACE_XY_ERROR,
            FailureReason.FINAL_VISUAL_PLACE_HEIGHT_ERROR,
        )
        try:
            for reason in reasons:
                with self.subTest(reason=reason.value):
                    result = build_episode_result(
                        env,
                        EpisodeOutcome(
                            success=False,
                            failure_reason=reason,
                            stage="final_visual_verification",
                            lift_height=None,
                            exception_message="expected test failure",
                            key_errors={},
                            controller_type="sensor_event_b1",
                            b1_metrics=metrics,
                        ),
                        None,
                        None,
                    )
                    self.assertFalse(result.perception_success)
                    self.assertEqual(result.perception_failure_reason, reason.value)
        finally:
            env.close()

    def test_non_perception_b1_failure_preserves_successful_perception(self) -> None:
        env = self.make_env()
        try:
            result = build_episode_result(
                env,
                EpisodeOutcome(
                    success=False,
                    failure_reason=FailureReason.GRASP_CANDIDATE_FAILED,
                    stage="grasp_candidate_check",
                    lift_height=None,
                    exception_message="expected test failure",
                    key_errors={},
                    controller_type="sensor_event_b1",
                    b1_metrics={
                        "initial_object_position": (0.50, 0.12, 0.245),
                        "locked_target_position": (0.50, -0.20, 0.222),
                    },
                ),
                None,
                None,
            )
            self.assertTrue(result.perception_success)
            self.assertIsNone(result.perception_failure_reason)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
