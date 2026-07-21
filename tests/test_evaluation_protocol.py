from __future__ import annotations

from pathlib import Path
import unittest

from evaluation.protocol import REQUIRED_CORE_METRICS, load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"


class EvaluationProtocolConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_identity_and_random_task_modes(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol.protocol_id, "evaluation_protocol")
        self.assertEqual(protocol.protocol_version, "1.0.1")
        self.assertEqual(protocol.metrics_schema_version, "1.0.0")
        self.assertEqual(protocol.split_id, "evaluation_protocol_v1")
        self.assertEqual(protocol.environment.pick.mode, "random")
        self.assertEqual(protocol.environment.place.mode, "random")
        self.assertEqual(protocol.environment.physics.mode, "random")

    def test_sizes_metrics_calibration_and_freeze_policy_exist(self) -> None:
        self.assertEqual(self.protocol.splits["calibration"].size, 30)
        self.assertEqual(self.protocol.splits["development"].size, 60)
        self.assertEqual(self.protocol.splits["held_out_test"].size, 100)
        self.assertTrue(REQUIRED_CORE_METRICS.issubset(self.protocol.core_metrics))
        self.assertTrue(self.protocol.calibration_policy_id)
        self.assertIn("freeze", self.protocol.raw)
        self.assertFalse(self.protocol.baseline_frozen)


if __name__ == "__main__":
    unittest.main()
