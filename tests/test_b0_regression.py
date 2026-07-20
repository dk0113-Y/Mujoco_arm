from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
B0_CONTROLLER_PATH = PROJECT_ROOT / "controllers" / "fixed_dls_controller.py"
TASK2_B0_NORMALIZED_SHA256 = (
    "e556924452a9814d3ff0f67319af795bf43cc2e5e17fbf111a10c6b01eafdd11"
)


class B0RegressionTests(unittest.TestCase):
    def test_fixed_dls_controller_matches_task2_baseline_exactly(self) -> None:
        source = B0_CONTROLLER_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.assertEqual(digest, TASK2_B0_NORMALIZED_SHA256)


if __name__ == "__main__":
    unittest.main()
