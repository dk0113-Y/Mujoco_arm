from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from controllers.sensor_event_controller import (
    B1ControllerFailure,
    B1Stage,
    SensorEventPickPlaceController,
)
from environments import PandaUTableEnv, load_config
from evaluation import FailureReason
from perception.types import DetectionResult, TaskPerceptionFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


def detection(
    kind: str,
    position: tuple[float, float, float] | None,
    *,
    confidence: float = 1.0,
) -> DetectionResult:
    success = position is not None
    return DetectionResult(
        detection_id="pick_object_0" if kind == "object" else "place_target_0",
        success=success,
        mask=np.ones((2, 2), dtype=bool) if success else np.zeros((2, 2), dtype=bool),
        pixel_count=4 if success else 0,
        center_pixel=(0.5, 0.5) if success else None,
        position=position,
        confidence=confidence if success else 0.0,
        failure_reason=(
            None
            if success
            else (
                "perception_object_not_found"
                if kind == "object"
                else "perception_target_not_found"
            )
        ),
    )


def frame(
    object_position: tuple[float, float, float] | None,
    target_position: tuple[float, float, float] | None,
    *,
    timestamp: float = 1.0,
) -> TaskPerceptionFrame:
    return TaskPerceptionFrame(
        object_detection=detection("object", object_position),
        target_detection=detection("target", target_position),
        timestamp=timestamp,
        latency_ms=0.25,
        camera_name="fake_rgbd",
        image_resolution=(32, 32),
    )


class FakeFrameProvider:
    source = "perception"

    def __init__(self, frames: list[TaskPerceptionFrame]) -> None:
        self.frames = list(frames)

    def observe(self) -> TaskPerceptionFrame:
        if not self.frames:
            raise RuntimeError("Fake provider has no remaining frames")
        return self.frames.pop(0)


class B1VisualMemoryTests(unittest.TestCase):
    def make_env_controller(self, **b1_changes):
        base = load_config(CONFIG_PATH).with_modes(observation_source="perception")
        config = replace(
            base,
            b1=replace(
                base.b1,
                initial_perception_frames=3,
                minimum_valid_perception_frames=2,
                pregrasp_perception_frames=3,
                minimum_valid_pregrasp_frames=2,
                final_verification_frames=3,
                final_minimum_valid_frames=2,
                **b1_changes,
            ),
        )
        env = PandaUTableEnv(config)
        env.reset(seed=42)
        controller = SensorEventPickPlaceController(config.controller, config.b1)
        return env, controller, controller._make_runtime(env)

    def test_target_is_locked_and_pregrasp_updates_only_object(self) -> None:
        env, controller, runtime = self.make_env_controller(
            maximum_pregrasp_correction=0.005
        )
        initial_object = (0.50, 0.12, 0.246)
        locked_target = (0.50, -0.20, 0.222)
        scene_provider = FakeFrameProvider(
            [frame(initial_object, locked_target) for _ in range(3)]
        )
        controller._collect_scene_perception(env, runtime, scene_provider, None)
        np.testing.assert_allclose(runtime.locked_target_position, locked_target)

        corrected = (0.52, 0.12, 0.246)
        object_only = FakeFrameProvider(
            [frame(corrected, None) for _ in range(3)]
        )
        controller._collect_pregrasp_object(env, runtime, object_only, None)
        np.testing.assert_allclose(runtime.corrected_object_position, corrected)
        np.testing.assert_allclose(runtime.locked_target_position, locked_target)
        self.assertTrue(
            runtime.metrics["pregrasp_correction_exceeded_threshold"]
        )
        env.close()

    def test_unstable_pregrasp_estimates_fail_explicitly(self) -> None:
        env, controller, runtime = self.make_env_controller(
            maximum_position_spread=0.01
        )
        runtime.initial_object_position = np.array([0.50, 0.12, 0.246])
        estimates = (
            (0.48, 0.12, 0.246),
            (0.50, 0.12, 0.246),
            (0.52, 0.12, 0.246),
        )
        provider = FakeFrameProvider([frame(position, None) for position in estimates])
        with self.assertRaises(B1ControllerFailure) as caught:
            controller._collect_pregrasp_object(env, runtime, provider, None)
        self.assertEqual(
            caught.exception.reason, FailureReason.PREGRASP_POSITION_UNSTABLE
        )
        env.close()

    def _verify_final(self, final_object, expected_reason):
        env, controller, runtime = self.make_env_controller()
        runtime.stage = B1Stage.FINAL_VISUAL_VERIFICATION
        runtime.locked_target_position = np.array([0.50, -0.20, 0.222])
        provider = FakeFrameProvider([frame(final_object, None) for _ in range(3)])
        if expected_reason is None:
            next_stage = controller._handle_final_visual_verification(
                env, runtime, provider, None
            )
            self.assertEqual(next_stage, B1Stage.COMPLETED)
        else:
            with self.assertRaises(B1ControllerFailure) as caught:
                controller._handle_final_visual_verification(
                    env, runtime, provider, None
                )
            self.assertEqual(caught.exception.reason, expected_reason)
        env.close()

    def test_final_visual_success_does_not_require_target_redetection(self) -> None:
        self._verify_final((0.50, -0.20, 0.245), None)

    def test_final_visual_xy_error_is_explicit(self) -> None:
        self._verify_final(
            (0.58, -0.20, 0.245),
            FailureReason.FINAL_VISUAL_PLACE_XY_ERROR,
        )

    def test_final_visual_height_error_is_explicit(self) -> None:
        self._verify_final(
            (0.50, -0.20, 0.30),
            FailureReason.FINAL_VISUAL_PLACE_HEIGHT_ERROR,
        )

    def test_final_object_missing_is_explicit(self) -> None:
        env, controller, runtime = self.make_env_controller()
        runtime.stage = B1Stage.FINAL_VISUAL_VERIFICATION
        runtime.locked_target_position = np.array([0.50, -0.20, 0.222])
        provider = FakeFrameProvider([frame(None, None) for _ in range(3)])
        with self.assertRaises(B1ControllerFailure) as caught:
            controller._handle_final_visual_verification(env, runtime, provider, None)
        self.assertEqual(caught.exception.reason, FailureReason.FINAL_OBJECT_NOT_FOUND)
        env.close()


if __name__ == "__main__":
    unittest.main()
