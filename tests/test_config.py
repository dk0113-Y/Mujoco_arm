from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from environments.config import load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config(CONFIG_PATH)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.pick.mode, "fixed")
        self.assertEqual(config.physics.fixed_mass, 0.10)
        self.assertEqual(config.controller.type, "fixed_dls_b0")
        self.assertGreater(config.b1.initial_perception_frames, 0)

    def test_legacy_b0_config_without_controller_type_or_b1_section_loads(self) -> None:
        source = CONFIG_PATH.read_text(encoding="utf-8")
        legacy, separator, _ = source.partition("\n[b1]\n")
        self.assertTrue(separator)
        legacy = legacy.replace('type = "fixed_dls_b0"\n', "", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_b0.toml"
            path.write_text(legacy, encoding="utf-8")
            config = load_config(path)

        self.assertEqual(config.controller.type, "fixed_dls_b0")
        self.assertEqual(config.b1.initial_perception_frames, 5)
        self.assertEqual(config.b1.final_verification_frames, 5)

    def test_invalid_mode_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(config, pick=replace(config.pick, mode="sometimes"))
        with self.assertRaisesRegex(ValueError, "pick.mode"):
            validate_config(invalid)

    def test_invalid_mass_range_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            physics=replace(config.physics, mass_range=(0.20, 0.05)),
        )
        with self.assertRaisesRegex(ValueError, "mass_range"):
            validate_config(invalid)

    def test_invalid_friction_range_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            physics=replace(
                config.physics,
                friction_min=(1.0, 0.03, 0.001),
                friction_max=(1.2, 0.02, 0.002),
            ),
        )
        with self.assertRaisesRegex(ValueError, "friction range"):
            validate_config(invalid)

    def test_invalid_observation_source_is_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            observation=replace(config.observation, source="fallback"),
        )
        with self.assertRaisesRegex(ValueError, "observation.source"):
            validate_config(invalid)

    def test_invalid_camera_axes_are_rejected(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            camera=replace(config.camera, y_axis_world=config.camera.x_axis_world),
        )
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            validate_config(invalid)

    def test_b1_requires_perception_and_joint_override_is_valid(self) -> None:
        config = load_config(CONFIG_PATH)
        with self.assertRaisesRegex(ValueError, "requires observation.source"):
            config.with_modes(controller_type="sensor_event_b1")
        b1 = config.with_modes(
            controller_type="sensor_event_b1",
            observation_source="perception",
        )
        self.assertEqual(b1.controller.type, "sensor_event_b1")
        self.assertEqual(b1.observation.source, "perception")

    def test_b1_valid_frame_count_cannot_exceed_total(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            b1=replace(
                config.b1,
                minimum_valid_perception_frames=(
                    config.b1.initial_perception_frames + 1
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_config(invalid)

    def test_b1_aperture_thresholds_are_validated(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            b1=replace(config.b1, release_aperture_threshold=0.081),
        )
        with self.assertRaisesRegex(ValueError, "0.08"):
            validate_config(invalid)

    def test_b1_motion_timeout_exceeds_reference_duration(self) -> None:
        config = load_config(CONFIG_PATH)
        invalid = replace(
            config,
            b1=replace(
                config.b1,
                motion_timeout=config.controller.transfer_duration,
            ),
        )
        with self.assertRaisesRegex(ValueError, "reference motion"):
            validate_config(invalid)


if __name__ == "__main__":
    unittest.main()
