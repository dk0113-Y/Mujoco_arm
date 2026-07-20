from __future__ import annotations

from dataclasses import fields
import inspect
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from environments import PandaUTableEnv
from perception import OracleExternalStateProvider, TaskStateEstimate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class OracleExternalStateProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PandaUTableEnv(CONFIG_PATH)
        self.env.reset(seed=42)
        self.provider = OracleExternalStateProvider(self.env)

    def tearDown(self) -> None:
        self.provider.close()
        self.env.close()

    def test_estimate_returns_current_external_truth(self) -> None:
        estimate = self.provider.estimate()

        np.testing.assert_array_equal(
            np.asarray(estimate.object_position),
            self.env.data.xpos[self.env.object_body_id],
        )
        np.testing.assert_array_equal(
            np.asarray(estimate.target_position),
            self.env.data.site_xpos[self.env.place_target_site_id],
        )
        self.assertEqual(estimate.timestamp, float(self.env.data.time))

    def test_estimate_has_oracle_identity_and_non_imaging_metadata(self) -> None:
        estimate = self.provider.estimate()

        self.assertIsInstance(estimate, TaskStateEstimate)
        self.assertEqual(estimate.object_id, "pick_object_0")
        self.assertEqual(estimate.target_id, "place_target_0")
        self.assertEqual(estimate.source, "oracle")
        self.assertTrue(estimate.valid)
        self.assertEqual(estimate.confidence, 1.0)
        self.assertIsNone(estimate.failure_reason)
        self.assertTrue(estimate.object_valid)
        self.assertTrue(estimate.target_valid)
        self.assertEqual(estimate.object_confidence, 1.0)
        self.assertEqual(estimate.target_confidence, 1.0)
        self.assertIsNone(estimate.object_failure_reason)
        self.assertIsNone(estimate.target_failure_reason)
        self.assertEqual(estimate.object_pixel_count, 0)
        self.assertEqual(estimate.target_pixel_count, 0)
        self.assertIsNone(estimate.camera_name)
        self.assertIsNone(estimate.image_resolution)
        self.assertTrue(math.isfinite(estimate.latency_ms))
        self.assertGreaterEqual(estimate.latency_ms, 0.0)

    def test_estimate_schema_contains_no_extra_control_truth(self) -> None:
        estimate = self.provider.estimate()
        allowed_fields = {
            "object_id",
            "target_id",
            "object_position",
            "target_position",
            "timestamp",
            "source",
            "valid",
            "confidence",
            "failure_reason",
            "object_pixel_count",
            "target_pixel_count",
            "latency_ms",
            "camera_name",
            "image_resolution",
            "object_valid",
            "target_valid",
            "object_confidence",
            "target_confidence",
            "object_failure_reason",
            "target_failure_reason",
        }
        self.assertEqual({field.name for field in fields(estimate)}, allowed_fields)
        self.assertEqual(set(vars(estimate)), allowed_fields)

        for forbidden in (
            "object_velocity",
            "future_object_position",
            "object_grasped",
            "grasp_success",
            "drop_detected",
            "placement_success",
            "grasp_pose",
            "optimal_path",
            "ik_reachable",
            "future_collision",
            "rgb",
            "depth",
            "mask",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(hasattr(estimate, forbidden))

    def test_repeated_static_samples_are_identical_except_latency(self) -> None:
        samples = [self.provider.estimate() for _ in range(5)]
        object_positions = np.asarray(
            [sample.object_position for sample in samples], dtype=float
        )
        target_positions = np.asarray(
            [sample.target_position for sample in samples], dtype=float
        )

        np.testing.assert_array_equal(
            object_positions,
            np.repeat(object_positions[:1], len(samples), axis=0),
        )
        np.testing.assert_array_equal(
            target_positions,
            np.repeat(target_positions[:1], len(samples), axis=0),
        )
        self.assertEqual({sample.timestamp for sample in samples}, {self.env.data.time})
        self.assertEqual(
            float(np.max(np.linalg.norm(object_positions - object_positions[0], axis=1))),
            0.0,
        )
        self.assertEqual(
            float(np.max(np.linalg.norm(target_positions - target_positions[0], axis=1))),
            0.0,
        )

    def test_provider_never_constructs_a_renderer_and_close_is_idempotent(self) -> None:
        with patch(
            "mujoco.Renderer",
            side_effect=AssertionError("Oracle provider must not construct a Renderer"),
        ):
            provider = OracleExternalStateProvider(self.env)
            provider.estimate()
            provider.close()
            provider.close()

        source = inspect.getsource(OracleExternalStateProvider)
        for forbidden in (
            "Renderer",
            "OverheadRGBDCamera",
            "RGBDFrame",
            "data.qvel",
            "current_episode",
            ".success(",
            "placement_errors",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
