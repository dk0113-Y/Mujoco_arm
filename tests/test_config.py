from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
