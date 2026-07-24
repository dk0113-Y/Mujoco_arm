from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from control_benchmarks.cartesian_outputs import (
    CARTESIAN_EPISODE_FIELDS,
    CARTESIAN_TIMESERIES_FIELDS,
)
from control_benchmarks.cartesian_runner import (
    CARTESIAN_TERMINATION_REASONS,
    run_cartesian_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "ci_baseline_v1.toml"
REQUIRED_OUTPUTS = {
    "run_manifest.json",
    "episode_metrics.csv",
    "timeseries.csv",
    "summary.json",
    "config_snapshot.toml",
}


def reject_non_finite_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


class CartesianImpedanceBenchmarkSmokeTests(unittest.TestCase):
    def test_cartesian_termination_reason_registry_is_complete(self) -> None:
        required = {
            "tcp_position_error_exceeded",
            "tcp_orientation_error_exceeded",
            "jacobian_rank_deficient",
            "jacobian_condition_exceeded",
            "invalid_orientation",
            "unexpected_contact",
        }
        self.assertTrue(required.issubset(CARTESIAN_TERMINATION_REASONS))

    def make_smoke_config(self, directory: Path) -> Path:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        replacements = {
            "maximum_duration = 5.0": "maximum_duration = 0.012",
            "hold_duration = 2.0": "hold_duration = 0.006",
            "translation_ramp_duration = 0.5": (
                "translation_ramp_duration = 0.002"
            ),
            "translation_duration = 4.0": "translation_duration = 0.008",
            "orientation_ramp_duration = 0.5": (
                "orientation_ramp_duration = 0.002"
            ),
            "orientation_duration = 4.0": "orientation_duration = 0.008",
            "line_duration = 3.0": "line_duration = 0.006",
            "circle_duration = 4.0": "circle_duration = 0.008",
        }
        for original, replacement in replacements.items():
            self.assertIn(original, text)
            text = text.replace(original, replacement)
        path = directory / "ci_smoke.toml"
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
            ("episode_metrics.csv", CARTESIAN_EPISODE_FIELDS),
            ("timeseries.csv", CARTESIAN_TIMESERIES_FIELDS),
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

    def test_each_experiment_writes_all_strict_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_smoke_config(root)
            for experiment in (
                "cartesian_hold",
                "translation_axes",
                "orientation_axes",
                "straight_line",
                "circle",
                "all",
            ):
                with self.subTest(experiment=experiment):
                    output = root / experiment
                    summary = run_cartesian_benchmark(
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
            marker = output / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                run_cartesian_benchmark(
                    config,
                    experiment="circle",
                    output=output,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_invalid_experiment_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.make_smoke_config(root)
            output = root / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "Unsupported experiment"):
                run_cartesian_benchmark(
                    config, experiment="invalid", output=output
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
