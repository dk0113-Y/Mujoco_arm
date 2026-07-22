from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from benchmark.development_diagnostics import (
    CANDIDATE_PATH,
    DECISIONS,
    DevelopmentEpisodeRecorder,
    FORMAL_D0_PATH,
    REQUIRED_METHODS,
    REQUIRED_SEEDS,
    SEED_SNAPSHOT_PATH,
    _directory_hashes,
    _replay_comparison,
    _write_artifact_manifests,
    validate_development_diagnostic_request,
)
from benchmark.methods import METHOD_SPECS
from benchmark.runner import _RecordingProvider, _prepare_output_dir
from controllers import B1DiagnosticSnapshot
from environments import PandaUTableEnv, load_config
from evaluation.protocol import load_protocol
from perception import OracleExternalStateProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_v1.toml"


class DevelopmentDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH, validate_splits=False)

    def test_request_accepts_only_fixed_development_candidates_and_order(self) -> None:
        _config, seeds, methods, candidates = validate_development_diagnostic_request(
            protocol=self.protocol,
            config_path=FROZEN_CONFIG_PATH,
            development_run_dir=FORMAL_D0_PATH,
            candidate_file=CANDIDATE_PATH,
            method_ids=REQUIRED_METHODS,
        )
        self.assertEqual(seeds, REQUIRED_SEEDS)
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(tuple(method.method_id for method in methods), REQUIRED_METHODS)
        self.assertEqual([item["seed"] for item in candidates], list(REQUIRED_SEEDS))
        development = set(
            int(line)
            for line in (
                PROJECT_ROOT
                / "configs/splits/evaluation_protocol_v1/development_v1.txt"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        self.assertTrue(set(REQUIRED_SEEDS).issubset(development))

    def test_request_rejects_custom_candidates_seed_snapshot_and_method_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "candidate.json"
            custom.write_bytes(CANDIDATE_PATH.read_bytes())
            with self.assertRaisesRegex(ValueError, "registered formal candidate"):
                validate_development_diagnostic_request(
                    protocol=self.protocol,
                    config_path=FROZEN_CONFIG_PATH,
                    development_run_dir=FORMAL_D0_PATH,
                    candidate_file=custom,
                    method_ids=REQUIRED_METHODS,
                )
            custom_seeds = Path(directory) / "seeds.txt"
            custom_seeds.write_text(SEED_SNAPSHOT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "custom diagnostic seed"):
                validate_development_diagnostic_request(
                    protocol=self.protocol,
                    config_path=FROZEN_CONFIG_PATH,
                    development_run_dir=FORMAL_D0_PATH,
                    candidate_file=CANDIDATE_PATH,
                    method_ids=REQUIRED_METHODS,
                    seed_snapshot_path=custom_seeds,
                )
        with self.assertRaisesRegex(ValueError, "exactly b0_oracle then b1_vision"):
            validate_development_diagnostic_request(
                protocol=self.protocol,
                config_path=FROZEN_CONFIG_PATH,
                development_run_dir=FORMAL_D0_PATH,
                candidate_file=CANDIDATE_PATH,
                method_ids=tuple(reversed(REQUIRED_METHODS)),
            )

    def test_candidate_roles_include_3170_physical_success_visual_failure_control(self) -> None:
        document = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
        item = next(value for value in document["candidates"] if value["seed"] == 3170)
        self.assertEqual(item["pair_category"], "both_success")
        self.assertEqual(item["b1_failure_reason"], "final_object_not_found")
        self.assertEqual(item["selection_role"], "matched_success_control")
        self.assertNotEqual(item["b1_result"], "success")

    def test_output_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nonempty"
            output.mkdir()
            (output / "user.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                _prepare_output_dir(output, overwrite=False)
            self.assertEqual((output / "user.txt").read_text(encoding="utf-8"), "preserve")

    @staticmethod
    def _snapshot(env: PandaUTableEnv, event: str = "episode_reset") -> B1DiagnosticSnapshot:
        observation = env.observation()
        fingers = tuple(float(value) for value in observation["finger_positions"])
        return B1DiagnosticSnapshot(
            event=event,
            simulation_time=float(observation["simulation_time"]),
            stage="scene_perception" if event == "episode_reset" else "close_gripper",
            next_stage=None,
            failure_reason=None,
            grasp_state="gripper_open",
            gripper_aperture=sum(fingers),
            gripper_aperture_velocity=0.0,
            left_finger_position=fingers[0],
            right_finger_position=fingers[1],
            commanded_state="open",
            left_contact=False,
            right_contact=False,
            bilateral_contact=False,
            bilateral_contact_duration=0.0,
            candidate_aperture=None,
            aperture_drop=None,
            commanded_closing_predicate=None,
            minimum_aperture_predicate=None,
            contact_predicate=None,
            lift_predicate=None,
            aperture_retention_predicate=None,
            collision_free_predicate=True,
            combined_predicate=None,
            candidate_hold_steps=0,
            confirmation_hold_steps=0,
            contact_loss_hold_steps=0,
            contact_loss_event_count=0,
            trial_lift_completed=False,
            robot_table_collision=False,
            tcp_position=tuple(float(value) for value in observation["tcp_position"]),
            finger_positions=fingers,
        )

    def test_recorder_fields_are_separated_and_rendering_is_state_invariant(self) -> None:
        env = PandaUTableEnv(load_config(FROZEN_CONFIG_PATH))
        env.reset(seed=2225)
        before = np.concatenate(
            ([env.data.time], env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
        )
        candidate = next(
            item
            for item in json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))["candidates"]
            if item["seed"] == 2225
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = DevelopmentEpisodeRecorder(
                env=env,
                method=METHOD_SPECS["b0_oracle"],
                seed=2225,
                pair_id="test",
                execution_index=0,
                output_dir=Path(directory),
                visualization_enabled=True,
                candidate=candidate,
            )
            recorder.observe(self._snapshot(env))
            recorder.close()
            self.assertFalse(recorder.errors)
            self.assertEqual(len(recorder.trace_rows), 1)
            fields = set(recorder.trace_rows[0])
            self.assertTrue(any(name.startswith("controller_observable.") for name in fields))
            self.assertTrue(any(name.startswith("privileged_diagnostic.") for name in fields))
            self.assertTrue(any(name.startswith("derived_diagnostic.") for name in fields))
            self.assertTrue(recorder._render_state_unchanged)
            self.assertEqual(len(recorder.frame_records), 2)
        after = np.concatenate(
            ([env.data.time], env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy())
        )
        env.close()
        self.assertTrue(np.array_equal(before, after))

    def test_recording_provider_observes_existing_call_once_and_ignores_callback_return(self) -> None:
        env = PandaUTableEnv(load_config(FROZEN_CONFIG_PATH))
        env.reset(seed=2225)
        provider = OracleExternalStateProvider(env)

        class Recorder:
            def __init__(self) -> None:
                self.calls = 0

            def observe_provider(self, **_kwargs):
                self.calls += 1
                return False

        recorder = Recorder()
        wrapper = _RecordingProvider(provider, env, diagnostic_recording=recorder)
        estimate = wrapper.estimate()
        self.assertTrue(estimate.valid)
        self.assertEqual(wrapper.estimate_call_count, 1)
        self.assertEqual(recorder.calls, 1)
        provider.close()
        env.close()

    def test_replay_mismatch_is_not_relaxed(self) -> None:
        with (FORMAL_D0_PATH / "episodes.csv").open("r", encoding="utf-8", newline="") as stream:
            import csv

            row = next(
                item
                for item in csv.DictReader(stream)
                if item["seed"] == "2225" and item["method_id"] == "b0_oracle"
            )
        result = SimpleNamespace(
            seed=2225,
            sampled_pick_position=tuple(json.loads(row["sampled_pick_position"])),
            sampled_place_position=tuple(json.loads(row["sampled_place_position"])),
            sampled_mass=float(row["sampled_mass"]),
            sampled_friction=tuple(json.loads(row["sampled_friction"])),
            pick_region=row["pick_region"],
            place_region=row["place_region"],
            final_stage=row["final_stage"],
            failure_reason=row["failure_reason"] or None,
            controller_reported_success=row["controller_reported_success"] == "True",
            privileged_ground_truth_success=row["privileged_ground_truth_success"] == "True",
            collision_count=int(row["collision_count"]),
            simulation_time=float(row["simulation_time"]),
        )
        execution = SimpleNamespace(
            seed=2225,
            method=SimpleNamespace(method_id="b0_oracle"),
            result=result,
        )
        summary = {
            "seed": 2225,
            "method": "b0_oracle",
            "episode_fingerprint": "deliberate-mismatch",
            "safe_task_success": row["safe_task_success"] == "True",
            "placement_success": row["placement_success"] == "True",
        }
        comparison = _replay_comparison([summary], [execution], FORMAL_D0_PATH)
        self.assertFalse(comparison["all_episodes_match"])
        self.assertFalse(comparison["comparisons"][0]["exact_checks"]["episode_fingerprint"])

    def test_artifact_manifest_hash_self_check_and_no_production_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "evidence.json").write_text("{}", encoding="utf-8")
            manifest = {"production_metrics_generated": False}
            _write_artifact_manifests(output, manifest)
            artifact = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact["self_check_pass"])
            self.assertEqual(
                artifact["artifacts"]["evidence.json"],
                _directory_hashes(output)["evidence.json"],
            )
            self.assertFalse((output / "production_metrics.json").exists())

    def test_decisions_are_only_the_five_declared_mechanism_results(self) -> None:
        self.assertEqual(
            DECISIONS,
            {"M-CENTERING", "M-ORIENTATION", "M-GATE", "M-COUPLED", "M-INCONCLUSIVE"},
        )


if __name__ == "__main__":
    unittest.main()
