from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark.runner as benchmark_runner
from benchmark.methods import (
    BENCHMARK_NAME,
    BENCHMARK_SCHEMA_VERSION,
    FORMAL_METHOD_IDS,
)
from benchmark.pairing import EpisodeFingerprint
from benchmark.runner import BenchmarkRunError, run_benchmark
from benchmark.schemas import (
    EPISODE_METADATA_FIELDS,
    EPISODE_RESULT_FIELDS,
    FAILURE_COUNT_FIELDS,
    PAIRED_RESULT_FIELDS,
)
from evaluation import EpisodeResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_benchmark.py"
EXPECTED_OUTPUT_FILES = {
    "run_manifest.json",
    "config_snapshot.toml",
    "seeds.json",
    "episodes.csv",
    "paired_results.csv",
    "failure_counts.csv",
    "summary.json",
    "run.log",
}
COMPLETED_PAIR_OUTCOMES = {
    "both_success",
    "oracle_only_success",
    "vision_only_success",
    "both_failed",
}


def _strict_json(path: Path):
    def reject_constant(value: str):
        raise ValueError(f"Non-standard JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def _csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return tuple(reader.fieldnames or ()), rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _structured_result(
    fingerprint: EpisodeFingerprint,
    *,
    observation_source: str,
) -> EpisodeResult:
    return EpisodeResult(
        seed=fingerprint.seed,
        pick_mode="fixed",
        place_mode="fixed",
        physics_mode="fixed",
        pick_region=fingerprint.pick_region,
        place_region=fingerprint.place_region,
        sampled_pick_position=fingerprint.pick_position,
        sampled_place_position=fingerprint.place_position,
        sampled_mass=fingerprint.mass,
        sampled_friction=fingerprint.friction,
        success=False,
        failure_reason="motion_stage_timeout",
        final_stage="move_to_pregrasp",
        simulation_time=1.25,
        lift_height=None,
        final_xy_error=0.3,
        final_height_error=0.02,
        collision_count=0,
        exception_message="Expected structured task failure",
        observation_source=observation_source,
        controller_type="sensor_event_b1",
        controller_reported_success=False,
        privileged_ground_truth_success=False,
        false_positive=False,
        false_negative=False,
        stage_durations={"scene_perception": 0.1},
    )


class BenchmarkHeadlessSmokeTests(unittest.TestCase):
    def test_one_seed_real_headless_pair_writes_traceable_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            seeds_path = temporary_root / "smoke_seeds.txt"
            seeds_path.write_text(
                "# One seed used only by the automated headless smoke test.\n42\n",
                encoding="utf-8",
            )
            output_dir = temporary_root / "benchmark_output"
            output_dir.mkdir()
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--config",
                str(CONFIG_PATH),
                "--methods",
                *FORMAL_METHOD_IDS,
                "--seeds-file",
                str(seeds_path),
                "--output-dir",
                str(output_dir),
            ]

            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=(
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}\n"
                ),
            )
            self.assertIn("completed_pairs=1/1", completed.stdout)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                EXPECTED_OUTPUT_FILES,
            )
            for filename in EXPECTED_OUTPUT_FILES:
                with self.subTest(filename=filename):
                    self.assertTrue((output_dir / filename).is_file())

            episode_header, episode_rows = _csv_rows(output_dir / "episodes.csv")
            paired_header, paired_rows = _csv_rows(
                output_dir / "paired_results.csv"
            )
            failure_header, failure_rows = _csv_rows(
                output_dir / "failure_counts.csv"
            )
            manifest = _strict_json(output_dir / "run_manifest.json")
            seeds = _strict_json(output_dir / "seeds.json")
            summary = _strict_json(output_dir / "summary.json")

            self.assertEqual(len(episode_rows), 2)
            self.assertEqual(len(paired_rows), 1)
            self.assertEqual(len(failure_rows), 2)
            self.assertEqual(paired_header, PAIRED_RESULT_FIELDS)
            self.assertEqual(failure_header, FAILURE_COUNT_FIELDS)
            episode_prefix = EPISODE_METADATA_FIELDS + EPISODE_RESULT_FIELDS
            self.assertEqual(episode_header[: len(episode_prefix)], episode_prefix)
            self.assertEqual(
                episode_header[len(episode_prefix) :],
                tuple(sorted(episode_header[len(episode_prefix) :])),
            )
            self.assertEqual(len(episode_header), len(set(episode_header)))

            expected_method_metadata = {
                "b0_oracle": ("oracle", "oracle", "0"),
                "b1_vision": ("vision", "perception", "1"),
            }
            self.assertEqual(
                [row["method_id"] for row in episode_rows],
                list(FORMAL_METHOD_IDS),
            )
            pair_ids = {row["pair_id"] for row in episode_rows}
            fingerprints = {row["episode_fingerprint"] for row in episode_rows}
            self.assertEqual(len(pair_ids), 1)
            self.assertEqual(len(fingerprints), 1)
            self.assertNotIn("", fingerprints)
            for row in episode_rows:
                with self.subTest(method=row["method_id"]):
                    external_source, provider_source, execution_index = (
                        expected_method_metadata[row["method_id"]]
                    )
                    self.assertEqual(row["external_state_source"], external_source)
                    self.assertEqual(row["observation_source"], provider_source)
                    self.assertEqual(row["controller_type"], "sensor_event_b1")
                    self.assertEqual(row["execution_index"], execution_index)
                    self.assertEqual(row["pair_valid"], "True")
                    self.assertEqual(row["program_error"], "")
                    self.assertNotEqual(
                        row["failure_reason"], "unexpected_exception"
                    )
                    self.assertTrue(row["final_stage"])

            pair = paired_rows[0]
            self.assertEqual(pair["pair_id"], next(iter(pair_ids)))
            self.assertEqual(pair["seed"], "42")
            self.assertEqual(pair["pair_valid"], "True")
            self.assertEqual(pair["pair_error"], "")
            self.assertEqual(pair["fingerprint"], next(iter(fingerprints)))
            self.assertIn(pair["outcome_category"], COMPLETED_PAIR_OUTCOMES)
            self.assertNotEqual(pair["outcome_category"], "program_error")
            self.assertNotEqual(pair["outcome_category"], "invalid_pair")

            self.assertEqual(manifest["benchmark_name"], BENCHMARK_NAME)
            self.assertEqual(
                manifest["benchmark_schema_version"], BENCHMARK_SCHEMA_VERSION
            )
            self.assertEqual(manifest["methods"], list(FORMAL_METHOD_IDS))
            self.assertEqual(
                manifest["method_execution_order"], list(FORMAL_METHOD_IDS)
            )
            self.assertEqual(manifest["total_requested_pairs"], 1)
            self.assertEqual(manifest["completed_pairs"], 1)
            self.assertEqual(manifest["invalid_pairs"], 0)
            self.assertEqual(manifest["unhandled_errors"], 0)
            self.assertEqual(manifest["unhandled_error_details"], [])
            self.assertTrue(manifest["pilot"])
            datetime.fromisoformat(manifest["start_time"])
            datetime.fromisoformat(manifest["end_time"])
            self.assertEqual(manifest["config_sha256"], _sha256(CONFIG_PATH))
            self.assertEqual(manifest["seed_file_sha256"], _sha256(seeds_path))
            self.assertEqual(len(manifest["git_commit"]), 40)
            self.assertIsInstance(manifest["git_dirty"], bool)
            self.assertIsInstance(manifest["git_status_short"], list)
            self.assertIsInstance(manifest["submodule_status"], list)
            self.assertTrue(manifest["python_version"])
            self.assertTrue(manifest["mujoco_version"])
            self.assertTrue(manifest["numpy_version"])
            self.assertTrue(manifest["operating_system"])
            self.assertIn("controller.type", manifest["effective_overrides"])
            self.assertEqual(manifest["command"][:2], command[:2])

            self.assertEqual(
                seeds,
                {
                    "duplicates_present": False,
                    "pilot": True,
                    "seed_count": 1,
                    "seeds": [42],
                },
            )
            self.assertEqual(
                (output_dir / "config_snapshot.toml").read_bytes(),
                CONFIG_PATH.read_bytes(),
            )

            self.assertEqual(set(summary["methods"]), set(FORMAL_METHOD_IDS))
            for method_id in FORMAL_METHOD_IDS:
                with self.subTest(summary_method=method_id):
                    method_summary = summary["methods"][method_id]
                    self.assertEqual(method_summary["requested_episodes"], 1)
                    self.assertEqual(method_summary["completed_episodes"], 1)
                    self.assertEqual(method_summary["program_errors"], 0)
                    self.assertEqual(
                        sum(method_summary["failure_reason_counts"].values()), 1
                    )
            paired_summary = summary["paired"]
            self.assertEqual(paired_summary["valid_pair_count"], 1)
            self.assertEqual(paired_summary["invalid_pair_count"], 0)
            self.assertEqual(paired_summary["program_error_pair_count"], 0)
            self.assertEqual(
                sum(
                    paired_summary[name]
                    for name in (
                        "both_success",
                        "oracle_only_success",
                        "vision_only_success",
                        "both_failed",
                    )
                ),
                1,
            )
            self.assertEqual(
                {row["method_id"] for row in failure_rows},
                set(FORMAL_METHOD_IDS),
            )
            self.assertTrue(all(row["count"] == "1" for row in failure_rows))
            log_text = (output_dir / "run.log").read_text(encoding="utf-8")
            self.assertIn("benchmark_start", log_text)
            self.assertIn("benchmark_end", log_text)
            self.assertIn("method=b0_oracle", log_text)
            self.assertIn("method=b1_vision", log_text)

    def test_mismatched_pair_writes_invalid_outputs_then_stops_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            seeds_path = temporary_root / "seeds.txt"
            seeds_path.write_text("42\n", encoding="utf-8")
            output_dir = temporary_root / "mismatch_output"
            oracle_fingerprint = EpisodeFingerprint(
                seed=42,
                pick_position=(0.50, 0.12, 0.246),
                place_position=(0.50, -0.20, 0.222),
                pick_region="front",
                place_region="right",
                mass=0.10,
                friction=(1.0, 0.005, 0.0001),
            )
            vision_fingerprint = EpisodeFingerprint(
                seed=42,
                pick_position=(0.52, 0.12, 0.246),
                place_position=(0.50, -0.20, 0.222),
                pick_region="front",
                place_region="right",
                mass=0.11,
                friction=(1.0, 0.005, 0.0001),
            )

            def fake_execute(
                method,
                _config,
                seed,
                pair_id,
                execution_index,
                _logger,
            ):
                selected = (
                    oracle_fingerprint
                    if method.method_id == "b0_oracle"
                    else vision_fingerprint
                )
                source = "oracle" if method.method_id == "b0_oracle" else "perception"
                return benchmark_runner._EpisodeExecution(
                    pair_id=pair_id,
                    seed=seed,
                    method=method,
                    execution_index=execution_index,
                    result=_structured_result(
                        selected,
                        observation_source=source,
                    ),
                    fingerprint=selected,
                    initial_robot_state=(0.0, 1.0, 2.0),
                )

            repository = {
                "repository_path": str(PROJECT_ROOT),
                "git_commit": "0" * 40,
                "git_branch": "test-branch",
                "git_dirty": False,
                "git_status_short": [],
                "submodule_status": [],
            }
            with patch(
                "benchmark.runner.repository_metadata",
                return_value=repository,
            ), patch(
                "benchmark.runner._execute_episode",
                side_effect=fake_execute,
            ) as execute_episode:
                with self.assertRaises(BenchmarkRunError) as caught:
                    run_benchmark(
                        config_path=CONFIG_PATH,
                        method_ids=FORMAL_METHOD_IDS,
                        seeds_file=seeds_path,
                        output_dir=output_dir,
                    )

            self.assertEqual(execute_episode.call_count, 2)
            error_message = str(caught.exception)
            self.assertIn("stopped after writing traceable outputs", error_message)
            self.assertIn("fingerprints do not match", error_message)
            self.assertIn("pick_position", error_message)
            self.assertIn("mass", error_message)
            self.assertTrue(EXPECTED_OUTPUT_FILES.issubset(
                {path.name for path in output_dir.iterdir()}
            ))

            _, episode_rows = _csv_rows(output_dir / "episodes.csv")
            paired_header, paired_rows = _csv_rows(
                output_dir / "paired_results.csv"
            )
            summary = _strict_json(output_dir / "summary.json")
            manifest = _strict_json(output_dir / "run_manifest.json")

            self.assertEqual(len(episode_rows), 2)
            self.assertEqual(
                [row["method_id"] for row in episode_rows],
                list(FORMAL_METHOD_IDS),
            )
            self.assertTrue(all(row["pair_valid"] == "False" for row in episode_rows))
            self.assertTrue(all(row["program_error"] == "" for row in episode_rows))
            self.assertEqual(
                {row["episode_fingerprint"] for row in episode_rows},
                {oracle_fingerprint.digest, vision_fingerprint.digest},
            )

            self.assertEqual(paired_header, PAIRED_RESULT_FIELDS)
            self.assertEqual(len(paired_rows), 1)
            pair = paired_rows[0]
            self.assertEqual(pair["seed"], "42")
            self.assertEqual(pair["pair_valid"], "False")
            self.assertEqual(pair["outcome_category"], "invalid_pair")
            self.assertIn("fingerprints do not match", pair["pair_error"])
            self.assertIn("pick_position", pair["pair_error"])
            self.assertIn("mass", pair["pair_error"])

            self.assertEqual(manifest["completed_pairs"], 0)
            self.assertEqual(manifest["invalid_pairs"], 1)
            self.assertEqual(manifest["unhandled_errors"], 0)
            self.assertEqual(summary["paired"]["valid_pair_count"], 0)
            self.assertEqual(summary["paired"]["invalid_pair_count"], 1)
            self.assertEqual(summary["paired"]["program_error_pair_count"], 0)
            for method_id in FORMAL_METHOD_IDS:
                with self.subTest(method_id=method_id):
                    method_summary = summary["methods"][method_id]
                    self.assertEqual(method_summary["requested_episodes"], 1)
                    self.assertEqual(method_summary["completed_episodes"], 0)
                    self.assertEqual(method_summary["program_errors"], 0)
                    self.assertIsNone(method_summary["ground_truth_success_rate"])
                    self.assertIsNone(
                        method_summary["controller_reported_success_rate"]
                    )
                    self.assertEqual(method_summary["failure_reason_counts"], {})

    def test_nonempty_output_directory_is_rejected_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            seeds_path = temporary_root / "seeds.txt"
            seeds_path.write_text("42\n", encoding="utf-8")
            output_dir = temporary_root / "existing_output"
            output_dir.mkdir()
            marker = output_dir / "user-data.txt"
            marker.write_text("must remain untouched", encoding="utf-8")

            with patch(
                "benchmark.runner.repository_metadata",
                return_value={"git_dirty": False},
            ), patch("benchmark.runner._execute_episode") as execute_episode:
                with self.assertRaisesRegex(
                    FileExistsError,
                    "Output directory is not empty",
                ):
                    run_benchmark(
                        config_path=CONFIG_PATH,
                        method_ids=FORMAL_METHOD_IDS,
                        seeds_file=seeds_path,
                        output_dir=output_dir,
                    )

            execute_episode.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "must remain untouched")
            self.assertEqual({path.name for path in output_dir.iterdir()}, {marker.name})

    def test_require_clean_git_rejects_dirty_repository_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            seeds_path = temporary_root / "seeds.txt"
            seeds_path.write_text("42\n", encoding="utf-8")
            output_dir = temporary_root / "not_created"

            with patch(
                "benchmark.runner.repository_metadata",
                return_value={
                    "git_dirty": True,
                    "git_status_short": [" M user_file.py"],
                },
            ), patch("benchmark.runner._execute_episode") as execute_episode:
                with self.assertRaisesRegex(
                    BenchmarkRunError,
                    "--require-clean-git",
                ):
                    run_benchmark(
                        config_path=CONFIG_PATH,
                        method_ids=FORMAL_METHOD_IDS,
                        seeds_file=seeds_path,
                        output_dir=output_dir,
                        require_clean_git=True,
                    )

            execute_episode.assert_not_called()
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
