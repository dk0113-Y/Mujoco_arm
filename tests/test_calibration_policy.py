from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from evaluation.protocol import calibration_parameter_catalog, load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"
BASELINE_PATH = PROJECT_ROOT / "configs" / "baselines" / "b1_vision_calibration_template.toml"


class CalibrationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_allowlist_exists_in_real_dataclasses_and_excludes_protocol_fields(self) -> None:
        catalog = calibration_parameter_catalog(self.protocol)
        self.assertEqual(
            [parameter.path for parameter in catalog],
            list(self.protocol.allowed_calibration_parameters),
        )
        self.assertTrue(all(parameter.calibration_allowed for parameter in catalog))
        self.assertTrue(all(not parameter.modifiable_after_freeze for parameter in catalog))
        forbidden = ("workspace.", "pick.", "place.", "physics.", "camera.", "simulation.")
        self.assertFalse(
            any(parameter.path.startswith(forbidden) for parameter in catalog)
        )

    def test_success_tolerances_are_protocol_protected(self) -> None:
        self.assertNotIn("b1.final_place_xy_tolerance", self.protocol.allowed_calibration_parameters)
        self.assertNotIn("b1.final_place_height_tolerance", self.protocol.allowed_calibration_parameters)
        self.assertEqual(self.protocol.raw["success"]["placement_xy_tolerance"], 0.06)
        self.assertEqual(self.protocol.raw["success"]["placement_height_tolerance"], 0.03)
        self.assertEqual(self.protocol.environment.b1.final_place_xy_tolerance, 0.06)
        self.assertEqual(self.protocol.environment.b1.final_place_height_tolerance, 0.03)
        self.assertEqual(self.protocol.protocol_version, "1.0.1")
        self.assertEqual(self.protocol.metrics_schema_version, "1.0.0")

    def test_development_test_tuning_and_automatic_freeze_are_forbidden(self) -> None:
        self.assertFalse(self.protocol.splits["development"].allows_b1_tuning)
        self.assertFalse(self.protocol.splits["held_out_test"].allows_b1_tuning)
        self.assertFalse(self.protocol.baseline_frozen)
        self.assertFalse(self.protocol.raw["calibration"]["automatic_parameter_search"])
        self.assertTrue(self.protocol.raw["freeze"]["user_controls_git_tag"])

    def test_protocol_load_does_not_modify_baseline_template(self) -> None:
        before = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
        load_protocol(PROTOCOL_PATH)
        after = hashlib.sha256(BASELINE_PATH.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
