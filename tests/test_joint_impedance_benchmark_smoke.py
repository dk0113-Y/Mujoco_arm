from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from control_benchmarks.outputs import EPISODE_FIELDS, TIMESERIES_FIELDS
from control_benchmarks.runner import run_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "ji_baseline_v1.toml"
REQUIRED_OUTPUTS = {
    "run_manifest.json",
    "episode_metrics.csv",
    "timeseries.csv",
    "summary.json",
    "config_snapshot.toml",
}


def reject_non_finite_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


class JointImpedanceBenchmarkSmokeTests(unittest.TestCase):
    def make_smoke_config(self, directory: Path) -> Path:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        replacements = {
            "maximum_duration = 4.0": "maximum_duration = 0.01",
            "zero_torque_duration = 0.8": "zero_torque_duration = 0.006",
            "compensation_hold_duration = 1.5": "compensation_hold_duration = 0.006",
            "impedance_hold_duration = 2.0": "impedance_hold_duration = 0.006",
            "minimum_jerk_duration = 2.0": "minimum_jerk_duration = 0.006",
            "single_joint_duration = 3.0": "single_joint_duration = 0.006",
            "sine_ramp_duration = 0.5": "sine_ramp_duration = 0.002",
            "multi_joint_duration = 3.0": "multi_joint_duration = 0.006",
        }
        for original, replacement in replacements.items():
            self.assertIn(original, text)
            text = text.replace(original, replacement)
        path = directory / "smoke.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def assert_valid_outputs(self, output: Path) -> None:
        self.assertEqual({item.name for item in output.iterdir()}, REQUIRED_OUTPUTS)
        for name in ("run_manifest.json", "summary.json"):
            with (output / name).open("r", encoding="utf-8") as stream:
                parsed = json.load(
                    stream, parse_constant=reject_non_finite_constant
                )
            self.assertIsInstance(parsed, dict)
        for name, fields in (
            ("episode_metrics.csv", EPISODE_FIELDS),
            ("timeseries.csv", TIMESERIES_FIELDS),
        ):
            with (output / name).open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), tuple(fields))
                rows = list(reader)
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(None not in row for row in rows))
            serialized = json.dumps(rows)
            self.assertNotIn("NaN", serialized)
            self.assertNotIn("Infinity", serialized)

    def test_each_experiment_writes_strict_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_smoke_config(root)
            for experiment in (
                "zero_torque",
                "compensation_hold",
                "impedance_hold",
                "single_joint",
                "multi_joint",
                "all",
            ):
                with self.subTest(experiment=experiment):
                    output = root / experiment
                    summary = run_benchmark(
                        config,
                        experiment=experiment,
                        output=output,
                    )
                    self.assertGreater(summary["episode_count"], 0)
                    self.assertTrue(summary["finite_value_status"])
                    self.assert_valid_outputs(output)

    def test_nonempty_output_directory_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_smoke_config(root)
            output = root / "occupied"
            output.mkdir()
            (output / "marker.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                run_benchmark(
                    config,
                    experiment="zero_torque",
                    output=output,
                )
            self.assertEqual(
                (output / "marker.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_invalid_experiment_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_smoke_config(root)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "Unsupported experiment"):
                run_benchmark(config, experiment="invalid", output=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
