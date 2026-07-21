from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from benchmark.calibration_diagnostics import (
    EpisodeDiagnosticRecorder,
    REQUIRED_DIAGNOSTIC_SEEDS,
    REQUIRED_METHODS,
    finalize_structured_review,
    validate_diagnostic_request,
)
from benchmark.methods import METHOD_SPECS
from benchmark.schemas import EPISODE_RESULT_FIELDS
from controllers import B1DiagnosticSnapshot, SensorEventPickPlaceController
from environments import PandaUTableEnv, load_config
from evaluation.protocol import load_protocol
from perception import OracleExternalStateProvider


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"
BASELINE_PATH = (
    PROJECT_ROOT / "configs" / "baselines" / "b1_vision_calibration_template.toml"
)
DIAGNOSTIC_SEEDS_PATH = (
    PROJECT_ROOT / "configs" / "diagnostics" / "b1_round_0_5_seeds.txt"
)


class CalibrationDiagnosticTests(unittest.TestCase):
    @staticmethod
    def _snapshot(env: PandaUTableEnv, *, event: str) -> B1DiagnosticSnapshot:
        observation = env.observation()
        finger_positions = tuple(float(value) for value in observation["finger_positions"])
        aperture = float(sum(finger_positions))
        return B1DiagnosticSnapshot(
            event=event,
            simulation_time=float(observation["simulation_time"]),
            stage="close_gripper",
            next_stage=None,
            failure_reason=None,
            grasp_state="closing",
            gripper_aperture=aperture,
            gripper_aperture_velocity=0.0,
            left_finger_position=finger_positions[0],
            right_finger_position=finger_positions[1],
            commanded_state="closing",
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
            collision_free_predicate=None,
            combined_predicate=None,
            candidate_hold_steps=0,
            confirmation_hold_steps=0,
            contact_loss_hold_steps=0,
            contact_loss_event_count=0,
            trial_lift_completed=False,
            robot_table_collision=False,
            tcp_position=tuple(float(value) for value in observation["tcp_position"]),
            finger_positions=finger_positions,
        )

    def test_request_is_exactly_four_existing_calibration_seeds_and_formal_pair(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        _config, seeds, methods = validate_diagnostic_request(
            protocol=protocol,
            config_path=BASELINE_PATH,
            seeds_path=DIAGNOSTIC_SEEDS_PATH,
            method_ids=REQUIRED_METHODS,
        )
        self.assertEqual(seeds, REQUIRED_DIAGNOSTIC_SEEDS)
        self.assertEqual(tuple(method.method_id for method in methods), REQUIRED_METHODS)
        calibration = set(
            int(line)
            for line in protocol.splits["calibration"].path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )
        self.assertTrue(set(seeds).issubset(calibration))

        with tempfile.TemporaryDirectory() as directory:
            replacement = Path(directory) / "reordered.txt"
            replacement.write_text("3915\n2802\n2957\n1268\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly, in order"):
                validate_diagnostic_request(
                    protocol=protocol,
                    config_path=BASELINE_PATH,
                    seeds_path=replacement,
                    method_ids=REQUIRED_METHODS,
                )
        with self.assertRaisesRegex(ValueError, "methods must be exactly"):
            validate_diagnostic_request(
                protocol=protocol,
                config_path=BASELINE_PATH,
                seeds_path=DIAGNOSTIC_SEEDS_PATH,
                method_ids=tuple(reversed(REQUIRED_METHODS)),
            )

    def test_formal_calibration_runner_still_rejects_diagnostic_seed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_calibration.py"),
                    "--protocol",
                    str(PROTOCOL_PATH),
                    "--baseline-config",
                    str(BASELINE_PATH),
                    "--seeds-file",
                    str(DIAGNOSTIC_SEEDS_PATH),
                    "--output-dir",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("accepts only the registered calibration", completed.stderr)
            self.assertFalse(output.exists())

    def test_diagnostics_do_not_extend_formal_episode_schema(self) -> None:
        self.assertNotIn("candidate_aperture", EPISODE_RESULT_FIELDS)
        self.assertNotIn("confirmation_hold_steps", EPISODE_RESULT_FIELDS)
        self.assertFalse(
            any(
                field.startswith("privileged_diagnostic.")
                or field.startswith("controller_observable.")
                for field in EPISODE_RESULT_FIELDS
            )
        )

    def test_recorder_separates_privileged_fields_and_rendering_preserves_state(self) -> None:
        config = load_config(BASELINE_PATH)
        env = PandaUTableEnv(config)
        env.reset(seed=1268)
        before = np.concatenate(
            (
                np.asarray([env.data.time], dtype=float),
                env.data.qpos.copy(),
                env.data.qvel.copy(),
                env.data.ctrl.copy(),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeDiagnosticRecorder(
                env=env,
                method=METHOD_SPECS["b0_oracle"],
                seed=1268,
                pair_id="test_pair",
                execution_index=0,
                output_dir=Path(directory),
                visualization_enabled=True,
            )
            recorder.observe(self._snapshot(env, event="close_gripper_complete"))
            recorder.close()
            self.assertFalse(recorder.errors)
            self.assertEqual(len(recorder.trace_rows), 1)
            row = recorder.trace_rows[0]
            self.assertIn("controller_observable.gripper_aperture", row)
            self.assertIn("controller_observable.left_contact", row)
            self.assertIn("privileged_diagnostic.object_position_x", row)
            self.assertIn("privileged_diagnostic.object_quaternion_w", row)
            self.assertEqual(len(recorder.frame_records), 2)
            self.assertTrue(
                all((Path(directory) / frame["path"]).is_file() for frame in recorder.frame_records)
            )
        after = np.concatenate(
            (
                np.asarray([env.data.time], dtype=float),
                env.data.qpos.copy(),
                env.data.qvel.copy(),
                env.data.ctrl.copy(),
            )
        )
        env.close()
        self.assertTrue(np.array_equal(before, after))

    def test_final_review_keeps_compatibility_key_and_structured_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "diagnostic_summary.json").write_text(
                '{"run_kind":"calibration_diagnostic_replay","manual_assessment":{}}',
                encoding="utf-8",
            )
            (output / "round_0_5_report.md").write_text(
                "initial report\n", encoding="utf-8"
            )
            (output / "run_manifest.json").write_text("{}", encoding="utf-8")
            review = output / "structured_review.json"
            review.write_text(
                """{
  "assessment_kind": "structured_evidence_review",
  "round_1_decision": "B",
  "parameter_change_recommended_now": false,
  "explicit_parameter_issue": false,
  "geometry_or_algorithm_issue": true,
  "evidence_gap": false,
  "rationale": "contact geometry evidence",
  "next_b2_direction": "geometry robustness",
  "round_1_parameter_adjustments": [],
  "b1_parameters_modified": false,
  "round_1_executed": false,
  "b1_frozen": false,
  "per_seed_findings": {}
}
""",
                encoding="utf-8",
            )
            summary = finalize_structured_review(output, review)
            self.assertIn("manual_assessment", summary)
            self.assertEqual(
                summary["manual_assessment"]["assessment_kind"],
                "structured_evidence_review",
            )
            self.assertEqual(summary["manual_assessment"]["round_1_decision"], "B")
            manifest = json.loads(
                (output / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["structured_review_finalized"])
            self.assertIn("diagnostic_summary.json", manifest["artifact_hashes"])
            self.assertNotIn(
                "human_evidence_review",
                (output / "diagnostic_summary.json").read_text(encoding="utf-8"),
            )

    @staticmethod
    def _run_oracle(observer):
        config = load_config(BASELINE_PATH)
        env = PandaUTableEnv(config)
        actions: list[np.ndarray] = []
        transitions: list[tuple[float, str, str]] = []
        original_step = env.step

        def recording_step(control):
            actions.append(np.asarray(control, dtype=float).copy())
            return original_step(control)

        env.step = recording_step  # type: ignore[method-assign]
        provider = OracleExternalStateProvider(env)
        controller = SensorEventPickPlaceController(config.controller, config.b1)
        original_transition = controller._transition

        def recording_transition(runtime, current_env, next_stage):
            transitions.append(
                (float(current_env.data.time), runtime.stage.value, next_stage.value)
            )
            return original_transition(runtime, current_env, next_stage)

        controller._transition = recording_transition  # type: ignore[method-assign]
        try:
            result = controller.run_episode(
                env,
                seed=2802,
                state_provider=provider,
                diagnostic_observer=observer,
            )
        finally:
            provider.close()
            env.close()
        return result, actions, transitions

    def test_observer_is_immutable_truth_isolated_and_behavior_invariant(self) -> None:
        baseline_result, baseline_actions, baseline_transitions = self._run_oracle(None)
        snapshots: list[B1DiagnosticSnapshot] = []

        def observer(snapshot: B1DiagnosticSnapshot):
            snapshots.append(snapshot)
            return False  # Return values are deliberately ignored by control.

        observed_result, observed_actions, observed_transitions = self._run_oracle(
            observer
        )
        self.assertTrue(snapshots)
        self.assertEqual(len(baseline_actions), len(observed_actions))
        for baseline, observed in zip(baseline_actions, observed_actions):
            self.assertTrue(np.array_equal(baseline, observed))
        self.assertEqual(baseline_transitions, observed_transitions)

        baseline = baseline_result.to_dict()
        observed = observed_result.to_dict()
        for latency_field in (
            "perception_latency_ms",
            "initial_perception_latency_ms",
            "pregrasp_perception_latency_ms",
            "final_visual_latency_ms",
        ):
            baseline.pop(latency_field, None)
            observed.pop(latency_field, None)
        self.assertEqual(baseline, observed)
        self.assertEqual(observed_result.final_stage, "grasp_confirmation")
        self.assertEqual(observed_result.failure_reason, "grasp_not_confirmed")

        confirmation = [
            snapshot
            for snapshot in snapshots
            if snapshot.event == "confirmation_sample"
        ]
        self.assertTrue(confirmation)
        self.assertTrue(any(item.aperture_drop is not None for item in confirmation))
        self.assertTrue(
            any(item.aperture_retention_predicate is False for item in confirmation)
        )
        snapshot_fields = set(asdict(confirmation[-1]))
        self.assertFalse(
            any(
                "object" in field or "privileged" in field
                for field in snapshot_fields
            )
        )
        with self.assertRaises(FrozenInstanceError):
            confirmation[-1].stage = "mutated"  # type: ignore[misc]

    def test_observer_exception_cannot_change_episode_result(self) -> None:
        baseline_result, baseline_actions, _ = self._run_oracle(None)

        def broken_observer(_snapshot):
            raise RuntimeError("diagnostic-only failure")

        observed_result, observed_actions, _ = self._run_oracle(broken_observer)
        self.assertEqual(len(baseline_actions), len(observed_actions))
        for baseline, observed in zip(baseline_actions, observed_actions):
            self.assertTrue(np.array_equal(baseline, observed))
        self.assertEqual(baseline_result.final_stage, observed_result.final_stage)
        self.assertEqual(baseline_result.failure_reason, observed_result.failure_reason)
        self.assertEqual(
            baseline_result.controller_reported_success,
            observed_result.controller_reported_success,
        )
        self.assertEqual(
            baseline_result.privileged_ground_truth_success,
            observed_result.privileged_ground_truth_success,
        )


if __name__ == "__main__":
    unittest.main()
