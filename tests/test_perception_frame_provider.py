from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path
import unittest

import numpy as np

from environments import PandaUTableEnv
from perception import ColorDepthDetector, OverheadRGBDCamera, RGBDPerceptionProvider
from perception.types import DetectionResult, TaskPerceptionFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class ObjectOnlyDetector:
    def __init__(self, detector: ColorDepthDetector) -> None:
        self._detector = detector
        self.config = detector.config

    def detect_object(self, frame):
        return self._detector.detect_object(frame)

    def detect_target(self, frame):
        return DetectionResult(
            detection_id="place_target_0",
            success=False,
            mask=np.zeros(frame.depth.shape, dtype=bool),
            pixel_count=0,
            center_pixel=None,
            position=None,
            confidence=0.0,
            failure_reason="perception_target_not_found",
        )


class PerceptionFrameProviderTests(unittest.TestCase):
    def test_component_api_has_no_privileged_truth_escape_hatch(self) -> None:
        self.assertEqual(
            {field.name for field in fields(TaskPerceptionFrame)},
            {
                "object_detection",
                "target_detection",
                "timestamp",
                "latency_ms",
                "camera_name",
                "image_resolution",
            },
        )
        source = inspect.getsource(RGBDPerceptionProvider.observe)
        for forbidden in (
            "object_body_id",
            "place_target_site_id",
            "current_episode",
            "data.xpos",
            "data.site_xpos",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_component_api_keeps_valid_object_when_target_is_missing(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        detector = ObjectOnlyDetector(ColorDepthDetector(env.config.perception))
        provider = RGBDPerceptionProvider(
            OverheadRGBDCamera(env.model, env.config.camera),
            env.data,
            detector,
        )
        component_frame = provider.observe()
        self.assertTrue(component_frame.object_detection.success)
        self.assertIsNotNone(component_frame.object_detection.position)
        self.assertFalse(component_frame.target_detection.success)
        self.assertEqual(
            component_frame.target_detection.failure_reason,
            "perception_target_not_found",
        )

        combined = provider.estimate()
        self.assertFalse(combined.valid)
        self.assertIsNotNone(combined.object_position)
        self.assertIsNone(combined.target_position)
        self.assertEqual(combined.failure_reason, "perception_target_not_found")
        provider.close()
        env.close()


if __name__ == "__main__":
    unittest.main()
