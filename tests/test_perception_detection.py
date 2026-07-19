from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from environments import PandaUTableEnv, load_config
from perception import ColorDepthDetector, OverheadRGBDCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class PerceptionDetectionTests(unittest.TestCase):
    def test_seed_42_object_and_target_errors(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        with OverheadRGBDCamera(env.model, env.config.camera) as camera:
            frame = camera.capture(env.data)
        detector = ColorDepthDetector(env.config.perception)
        object_detection = detector.detect_object(frame)
        target_detection = detector.detect_target(frame)
        self.assertTrue(object_detection.success, object_detection.failure_reason)
        self.assertTrue(target_detection.success, target_detection.failure_reason)
        object_error = np.linalg.norm(
            np.asarray(object_detection.position) - env.data.xpos[env.object_body_id]
        )
        target_error = np.linalg.norm(
            np.asarray(target_detection.position)
            - env.data.site_xpos[env.place_target_site_id]
        )
        self.assertLess(float(object_error), 0.01)
        self.assertLess(float(target_error), 0.01)
        env.close()

    def test_front_left_and_right_region_detection(self) -> None:
        base = load_config(CONFIG_PATH)
        cases = {
            "front": ((0.70, 0.30, 0.246), (0.50, -0.20, 0.222)),
            "left": ((0.30, 0.57, 0.246), (0.50, -0.20, 0.222)),
            "right": ((0.30, -0.57, 0.246), (0.50, 0.20, 0.222)),
        }
        for region, (pick, place) in cases.items():
            with self.subTest(region=region):
                config = replace(
                    base,
                    pick=replace(base.pick, fixed_position=pick),
                    place=replace(base.place, fixed_position=place),
                )
                env = PandaUTableEnv(config)
                env.reset(seed=42)
                with OverheadRGBDCamera(env.model, config.camera) as camera:
                    frame = camera.capture(env.data)
                result = ColorDepthDetector(config.perception).detect_object(frame)
                self.assertTrue(result.success, result.failure_reason)
                error = np.linalg.norm(
                    np.asarray(result.position) - env.data.xpos[env.object_body_id]
                )
                self.assertLess(float(error), 0.01)
                env.close()

    def test_empty_color_and_invalid_depth_fail_structurally(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        with OverheadRGBDCamera(env.model, env.config.camera) as camera:
            frame = camera.capture(env.data)
        detector = ColorDepthDetector(env.config.perception)
        empty = replace(frame, rgb=np.zeros_like(frame.rgb))
        self.assertEqual(
            detector.detect_object(empty).failure_reason,
            "perception_object_not_found",
        )
        red = np.zeros_like(frame.rgb)
        red[100:110, 100:110] = (200, 20, 20)
        invalid_depth = frame.depth.copy()
        invalid_depth[100:110, 100:110] = np.nan
        invalid = replace(frame, rgb=red, depth=invalid_depth)
        self.assertEqual(
            detector.detect_object(invalid).failure_reason,
            "perception_invalid_depth",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
