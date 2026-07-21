from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"
SMOKE_PATH = PROJECT_ROOT / "configs" / "splits" / "evaluation_protocol_v1" / "calibration_smoke.txt"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_calibration.py"


class ProtocolCalibrationSmokeTests(unittest.TestCase):
    def test_two_seed_calibration_run_is_structured_and_never_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calibration_smoke"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--seeds-file",
                    str(SMOKE_PATH),
                    "--output-dir",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=240,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "production_metrics.json").read_text(encoding="utf-8"))
            with (output / "episodes.csv").open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(manifest["calibration_run"])
            self.assertFalse(manifest["baseline_frozen"])
            self.assertFalse(manifest["automatic_parameter_search"])
            self.assertEqual(manifest["split_name"], "calibration_smoke")
            self.assertEqual(len(rows), 4)
            b1_rows = [row for row in rows if row["method_id"] == "b1_vision"]
            self.assertEqual(len(b1_rows), 2)
            self.assertTrue(all(row["final_stage"] for row in b1_rows))
            self.assertTrue(all(row["program_error"] == "" for row in b1_rows))
            self.assertEqual(metrics["requested_episode_count"], 4)
            self.assertFalse((PROJECT_ROOT / "configs" / "baselines" / "b1_vision_v1.toml").exists())


if __name__ == "__main__":
    unittest.main()
