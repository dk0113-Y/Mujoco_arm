from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from benchmark.manifest import sha256_file
from benchmark.pairing import EpisodeFingerprint
from benchmark.runner import _EpisodeExecution, run_benchmark
from environments import load_config
from evaluation import EpisodeResult
from evaluation.protocol import load_protocol
from evaluation.split_analysis import TaskSample
from scripts import build_development_strata as strata
from scripts import run_development as development


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DevelopmentEntrypointTests(unittest.TestCase):
    def test_accepts_only_registered_development_and_frozen_identity(self) -> None:
        protocol, metadata = development.validate_development_request(
            protocol_path=development.PROTOCOL_PATH,
            frozen_config_path=development.FROZEN_CONFIG_PATH,
            seeds_file=development.DEVELOPMENT_SEEDS_PATH,
            freeze_manifest_path=development.FREEZE_MANIFEST_PATH,
            method_ids=("b0_oracle", "b1_vision"),
        )
        self.assertEqual(protocol.splits["development"].size, 60)
        self.assertEqual(metadata["frozen_baseline_id"], "b1_vision_v1")
        self.assertEqual(
            metadata["frozen_config_sha256"],
            development.EXPECTED_FROZEN_CONFIG_SHA256,
        )
        self.assertEqual(
            metadata["verified_behavior_commit"],
            development.EXPECTED_VERIFIED_BEHAVIOR_COMMIT,
        )
        self.assertEqual(
            metadata["freeze_package_commit"],
            development.EXPECTED_FREEZE_PACKAGE_COMMIT,
        )

    def test_rejects_calibration_held_out_and_custom_seed_files(self) -> None:
        protocol = load_protocol(development.PROTOCOL_PATH)
        for path in (
            protocol.splits["calibration"].path,
            protocol.splits["calibration_smoke"].path,
            protocol.splits["held_out_test"].path,
        ):
            with self.subTest(path=path), self.assertRaises(
                development.DevelopmentRunValidationError
            ):
                development.validate_development_request(
                    protocol_path=development.PROTOCOL_PATH,
                    frozen_config_path=development.FROZEN_CONFIG_PATH,
                    seeds_file=path,
                    freeze_manifest_path=development.FREEZE_MANIFEST_PATH,
                    method_ids=("b0_oracle", "b1_vision"),
                )
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "development_copy.txt"
            custom.write_bytes(development.DEVELOPMENT_SEEDS_PATH.read_bytes())
            with self.assertRaises(development.DevelopmentRunValidationError):
                development.validate_development_request(
                    protocol_path=development.PROTOCOL_PATH,
                    frozen_config_path=development.FROZEN_CONFIG_PATH,
                    seeds_file=custom,
                    freeze_manifest_path=development.FREEZE_MANIFEST_PATH,
                    method_ids=("b0_oracle", "b1_vision"),
                )

    def test_rejects_wrong_config_hash_freeze_state_and_method_order(self) -> None:
        arguments = {
            "protocol_path": development.PROTOCOL_PATH,
            "frozen_config_path": development.FROZEN_CONFIG_PATH,
            "seeds_file": development.DEVELOPMENT_SEEDS_PATH,
            "freeze_manifest_path": development.FREEZE_MANIFEST_PATH,
            "method_ids": ("b0_oracle", "b1_vision"),
        }
        real_hash = development.sha256_file

        def mismatched_hash(path: str | Path) -> str:
            if Path(path).resolve() == development.FROZEN_CONFIG_PATH.resolve():
                return "0" * 64
            return real_hash(path)

        with mock.patch.object(development, "sha256_file", side_effect=mismatched_hash):
            with self.assertRaisesRegex(
                development.DevelopmentRunValidationError, "Frozen config SHA-256"
            ):
                development.validate_development_request(**arguments)

        manifest = json.loads(
            development.FREEZE_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        manifest["freeze_state"] = "verified_pending_user_commit"
        with mock.patch.object(development, "_strict_json", return_value=manifest):
            with self.assertRaisesRegex(
                development.DevelopmentRunValidationError, "state"
            ):
                development.validate_development_request(**arguments)

        for methods in (("b1_vision", "b0_oracle"), ("b1_vision",)):
            with self.subTest(methods=methods), self.assertRaisesRegex(
                development.DevelopmentRunValidationError, "exactly"
            ):
                development.validate_development_request(
                    **{**arguments, "method_ids": methods}
                )

    def test_rejects_missing_manifest_and_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with self.assertRaises(development.DevelopmentRunValidationError):
                development.validate_development_request(
                    protocol_path=development.PROTOCOL_PATH,
                    frozen_config_path=development.FROZEN_CONFIG_PATH,
                    seeds_file=development.DEVELOPMENT_SEEDS_PATH,
                    freeze_manifest_path=missing,
                    method_ids=("b0_oracle", "b1_vision"),
                )
            output = Path(directory) / "formal"
            output.mkdir()
            (output / "existing.txt").write_text("user data", encoding="utf-8")
            with mock.patch.object(development, "DEVELOPMENT_OUTPUT_PATH", output):
                with self.assertRaisesRegex(
                    development.DevelopmentRunValidationError, "must be empty"
                ):
                    development.validate_output_directory(output)

    def test_benchmark_manifest_records_frozen_development_identity(self) -> None:
        protocol, metadata = development.validate_development_request(
            protocol_path=development.PROTOCOL_PATH,
            frozen_config_path=development.FROZEN_CONFIG_PATH,
            seeds_file=development.DEVELOPMENT_SEEDS_PATH,
            freeze_manifest_path=development.FREEZE_MANIFEST_PATH,
            method_ids=("b0_oracle", "b1_vision"),
        )

        def fake_execute(method, config, seed, pair_id, execution_index, logger):
            result = EpisodeResult(
                seed=seed,
                pick_mode="random",
                place_mode="random",
                physics_mode="random",
                pick_region="front",
                place_region="left",
                sampled_pick_position=(0.4, 0.1, 0.246),
                sampled_place_position=(0.2, 0.55, 0.222),
                sampled_mass=0.1,
                sampled_friction=(1.0, 0.01, 0.001),
                success=True,
                failure_reason=None,
                final_stage="completed",
                simulation_time=10.0,
                lift_height=0.04,
                final_xy_error=0.0,
                final_height_error=0.0,
                collision_count=0,
                exception_message=None,
                controller_type="sensor_event_b1",
                controller_reported_success=True,
                privileged_ground_truth_success=True,
                false_positive=False,
                false_negative=False,
                initial_perception_frame_count=5,
                initial_valid_frame_count=5,
                pregrasp_perception_frame_count=3,
                pregrasp_valid_frame_count=3,
                grasp_candidate=True,
                trial_lift_completed=True,
                grasp_confirmed=True,
                contact_loss_event_count=0,
                grasp_lost=False,
                final_visual_frame_count=5,
                final_visual_valid_frame_count=5,
            )
            return _EpisodeExecution(
                pair_id=pair_id,
                seed=seed,
                method=method,
                execution_index=execution_index,
                result=result,
                fingerprint=EpisodeFingerprint.from_episode_result(result),
                initial_robot_state=(0.0, 1.0),
            )

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "benchmark.runner._execute_episode", side_effect=fake_execute
        ):
            output = Path(directory) / "development"
            result = run_benchmark(
                config_path=development.FROZEN_CONFIG_PATH,
                method_ids=("b0_oracle", "b1_vision"),
                seeds_file=development.DEVELOPMENT_SEEDS_PATH,
                output_dir=output,
                protocol=protocol,
                split_name="development",
                baseline_frozen=True,
                development_run=True,
                frozen_baseline_metadata=metadata,
            )
            self.assertEqual(result.exit_code, 0)
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["development_run"])
            self.assertTrue(manifest["baseline_frozen"])
            self.assertFalse(manifest["calibration_run"])
            self.assertFalse(manifest["diagnostics_enabled"])
            self.assertFalse(manifest["visualization_enabled"])
            self.assertEqual(manifest["frozen_baseline_id"], "b1_vision_v1")


class DevelopmentStrataTests(unittest.TestCase):
    @staticmethod
    def _samples() -> list[TaskSample]:
        seeds = [int(line) for line in development.DEVELOPMENT_SEEDS_PATH.read_text().splitlines()]
        values: list[TaskSample] = []
        regions = ("front", "left", "right")
        for index, seed in enumerate(seeds):
            fraction = index / 59.0
            pick = regions[index % 3]
            place = regions[(index // 3) % 3]
            pick_position = (0.3 + 0.1 * fraction, 0.1, 0.246)
            place_position = (0.1, 0.55, 0.222)
            values.append(
                TaskSample(
                    seed=seed,
                    pick_region=pick,
                    place_region=place,
                    pick_position=pick_position,
                    place_position=place_position,
                    pick_place_distance=0.2 + fraction,
                    mass=0.05 + 0.15 * fraction,
                    friction=(
                        0.8 + 0.6 * fraction,
                        0.005 + 0.015 * fraction,
                        0.0005 + 0.0015 * fraction,
                    ),
                    settled_object_table_penetration=0.0,
                )
            )
        return values

    def test_strata_uses_fixed_bins_without_controller_or_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            strata, "collect_task_samples", return_value=self._samples()
        ), mock.patch.object(
            strata,
            "repository_metadata",
            return_value={"git_commit": "a" * 40, "git_dirty": False},
        ):
            output = Path(directory) / "strata.json"
            result = strata.build_development_strata(
                protocol_path=strata.PROTOCOL_PATH,
                seeds_file=strata.DEVELOPMENT_SEEDS_PATH,
                output_path=output,
            )
            self.assertEqual(result["seed_count"], 60)
            self.assertFalse(result["controller_outcomes_used"])
            self.assertFalse(result["development_outcomes_used"])
            self.assertFalse(result["held_out_data_read"])
            self.assertFalse(result["renderer_created"])
            self.assertFalse(result["ik_executed"])
            self.assertEqual(
                result["binning"]["mass"]["edges"],
                [0.05, 0.08750000000000001, 0.125, 0.16250000000000003, 0.2],
            )
            self.assertEqual(
                result["binning"]["pick_place_distance"]["cut_points"],
                [0.45, 0.7, 0.95],
            )
            self.assertTrue(output.is_file())

    def test_strata_module_does_not_import_control_or_rendering(self) -> None:
        source = Path(strata.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "SensorEventPickPlaceController",
            "mujoco.Renderer",
            "solve_pose_ik",
            "held_out_test_v1.txt",
            "failure_reason",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
