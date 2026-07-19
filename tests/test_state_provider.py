from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import unittest

import numpy as np

from controllers import FixedDLSPickPlaceController
from environments import PandaUTableEnv, load_config
from perception import (
    ColorDepthDetector,
    OverheadRGBDCamera,
    PrivilegedStateProvider,
    RGBDPerceptionProvider,
)
from perception.types import TaskStateEstimate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class FakeTaskStateProvider:
    source = "perception"

    def __init__(self, object_position, target_position) -> None:
        self.object_position = object_position
        self.target_position = target_position
        self.calls = 0

    def estimate(self) -> TaskStateEstimate:
        self.calls += 1
        return TaskStateEstimate(
            object_id="pick_object_0",
            target_id="place_target_0",
            object_position=self.object_position,
            target_position=self.target_position,
            timestamp=1.0,
            source=self.source,
            valid=True,
            confidence=1.0,
            failure_reason=None,
            object_pixel_count=100,
            target_pixel_count=100,
            latency_ms=0.1,
            camera_name="fake_camera",
            image_resolution=(32, 32),
        )

    def close(self) -> None:
        return None


class RecordingController(FixedDLSPickPlaceController):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.received_positions = None

    def _actions(self, env, object_position, target_position):
        self.received_positions = (object_position.copy(), target_position.copy())
        return []


class StateProviderTests(unittest.TestCase):
    def test_privileged_and_perception_sources_and_repeatability(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        privileged = PrivilegedStateProvider(env).estimate()
        self.assertEqual(privileged.source, "privileged")
        self.assertEqual(privileged.object_id, "pick_object_0")
        self.assertEqual(privileged.target_id, "place_target_0")

        camera = OverheadRGBDCamera(env.model, env.config.camera)
        perception = RGBDPerceptionProvider(
            camera, env.data, ColorDepthDetector(env.config.perception)
        )
        first = perception.estimate()
        second = perception.estimate()
        self.assertEqual(first.source, "perception")
        self.assertTrue(first.valid)
        self.assertEqual(first.object_id, "pick_object_0")
        np.testing.assert_allclose(first.object_position, second.object_position)
        np.testing.assert_allclose(first.target_position, second.target_position)
        perception.close()
        env.close()

    def test_controller_uses_fake_provider_positions(self) -> None:
        config = load_config(CONFIG_PATH).with_modes(observation_source="perception")
        env = PandaUTableEnv(config)
        fake_object = (0.35, 0.30, 0.245)
        fake_target = (0.70, -0.30, 0.222)
        provider = FakeTaskStateProvider(fake_object, fake_target)
        controller = RecordingController(config.controller)
        result = controller.run_episode(env, seed=42, state_provider=provider)
        received_object, received_target = controller.received_positions
        np.testing.assert_allclose(received_object, fake_object)
        np.testing.assert_allclose(received_target, fake_target)
        self.assertGreaterEqual(provider.calls, 2)
        self.assertEqual(result.estimated_object_position, fake_object)
        env.close()

    def test_controller_source_has_no_external_truth_escape_hatch(self) -> None:
        import controllers.fixed_dls_controller as controller_module

        source = inspect.getsource(controller_module)
        for forbidden in (
            "object_body_id",
            "place_target_site_id",
            "current_episode",
            "data.xpos",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_perception_mode_never_defaults_to_privileged_provider(self) -> None:
        config = load_config(CONFIG_PATH).with_modes(observation_source="perception")
        env = PandaUTableEnv(config)
        result = FixedDLSPickPlaceController(config.controller).run_episode(
            env, seed=42
        )
        self.assertFalse(result.success)
        self.assertEqual(
            result.failure_reason, "perception_projection_error"
        )
        self.assertIsNone(result.estimated_object_position)
        env.close()


if __name__ == "__main__":
    unittest.main()
