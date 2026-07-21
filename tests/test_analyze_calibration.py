from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.pairing import EpisodeFingerprint
from scripts.analyze_calibration import (
    CalibrationAnalysisError,
    REQUIRED_INPUT_FILES,
    analyze_calibration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_csv(
    path: Path,
    change,
) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or ())
    change(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path) -> Path:
    output = root / "round_0"
    output.mkdir()
    config = output / "config_snapshot.toml"
    config.write_text("[environment]\nseed = 42\n", encoding="utf-8")
    protocol = output / "protocol_snapshot.toml"
    protocol.write_text(
        "\n".join(
            (
                "[protocol]",
                'protocol_id = "evaluation_protocol"',
                'protocol_version = "1.0.1"',
                'metrics_schema_version = "1.0.0"',
                "",
                "[physics]",
                "mass_range = [0.05, 0.20]",
                "friction_min = [0.80, 0.005, 0.0005]",
                "friction_max = [1.40, 0.02, 0.002]",
                "",
                "[b1]",
                "arrival_position_tolerance = 0.015",
                "arrival_orientation_tolerance = 0.05",
                "settled_joint_velocity_threshold = 0.15",
                "",
                "[calibration]",
                "baseline_frozen = false",
                "automatic_parameter_search = false",
                "",
            )
        ),
        encoding="utf-8",
    )
    seeds = list(range(1000, 1030))
    seed_payload = ("\n".join(str(seed) for seed in seeds) + "\n").encode()
    seed_hash = hashlib.sha256(seed_payload).hexdigest()
    config_hash = _sha256(config)
    protocol_hash = _sha256(protocol)
    manifest = {
        "total_requested_pairs": 30,
        "completed_pairs": 30,
        "invalid_pairs": 0,
        "unhandled_errors": 0,
        "unhandled_error_details": [],
        "protocol_id": "evaluation_protocol",
        "protocol_version": "1.0.1",
        "metrics_schema_version": "1.0.0",
        "split_id": "evaluation_protocol_v1",
        "split_name": "calibration",
        "calibration_run": True,
        "baseline_frozen": False,
        "automatic_parameter_search": False,
        "pilot": False,
        "methods": ["b0_oracle", "b1_vision"],
        "method_execution_order": ["b0_oracle", "b1_vision"],
        "effective_overrides": {},
        "git_dirty": False,
        "git_status_short": [],
        "git_branch": "experiment/test",
        "git_commit": "a" * 40,
        "submodule_status": [" " + "b" * 40 + " models/example"],
        "config_sha256": config_hash,
        "protocol_config_sha256": protocol_hash,
        "seed_file_sha256": seed_hash,
        "start_time": "2026-07-21T00:00:00+00:00",
        "end_time": "2026-07-21T00:01:00+00:00",
        "command": ["python", "scripts/run_calibration.py"],
    }
    _write_json(output / "run_manifest.json", manifest)
    _write_json(
        output / "seeds.json",
        {
            "seeds": seeds,
            "seed_count": 30,
            "duplicates_present": False,
            "pilot": False,
        },
    )

    episode_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    failure_counts = {
        ("b0_oracle", "success"): 30,
        ("b1_vision", "success"): 15,
        ("b1_vision", "grasp_not_confirmed"): 15,
    }
    for index, seed in enumerate(seeds):
        pick_region = ("front", "left", "right")[index % 3]
        place_region = ("left", "right", "front")[index % 3]
        pick = (0.31 + 0.01 * (index % 10), 0.1, 0.246)
        place = (0.5, -0.2, 0.222)
        mass = 0.05 + 0.15 * index / 29
        friction = (
            0.8 + 0.6 * index / 29,
            0.005 + 0.015 * index / 29,
            0.0005 + 0.0015 * index / 29,
        )
        fingerprint = EpisodeFingerprint(
            seed=seed,
            pick_position=pick,
            place_position=place,
            pick_region=pick_region,
            place_region=place_region,
            mass=mass,
            friction=friction,
        ).digest
        pair_id = f"pair_{index:04d}_seed_{seed}"
        b1_success = index < 15
        for method in ("b0_oracle", "b1_vision"):
            success = True if method == "b0_oracle" else b1_success
            failure = "" if success else "grasp_not_confirmed"
            final_stage = "completed" if success else "grasp_confirmation"
            episode_rows.append(
                {
                    "benchmark_name": "benchmark0_oracle_paired_eval",
                    "pair_id": pair_id,
                    "method_id": method,
                    "external_state_source": (
                        "oracle" if method == "b0_oracle" else "vision"
                    ),
                    "execution_index": 2 * index
                    + (1 if method == "b1_vision" else 0),
                    "episode_fingerprint": fingerprint,
                    "pair_valid": True,
                    "program_error": "",
                    "seed": seed,
                    "pick_region": pick_region,
                    "place_region": place_region,
                    "sampled_pick_position": json.dumps(pick, separators=(",", ":")),
                    "sampled_place_position": json.dumps(place, separators=(",", ":")),
                    "sampled_mass": mass,
                    "sampled_friction": json.dumps(
                        friction, separators=(",", ":")
                    ),
                    "failure_reason": failure,
                    "final_stage": final_stage,
                    "simulation_time": 18.0 if success else 13.0,
                    "collision_count": 0,
                    "controller_reported_success": success,
                    "privileged_ground_truth_success": success,
                    "false_positive": False,
                    "false_negative": False,
                    "result_fields_complete": True,
                    "protocol_version": "1.0.1",
                    "split_name": "calibration",
                    "config_sha256": config_hash,
                    "safe_task_success": success,
                    "placement_success": success,
                    "first_attempt_placement_success": success,
                    "collision_episode": False,
                    "unexplained_failure": False,
                    "pick_place_region_pair": f"{pick_region}->{place_region}",
                    "same_region": pick_region == place_region,
                    "pick_place_distance": 0.2 + 0.04 * index,
                    "initial_valid_frame_count": 5,
                    "pregrasp_valid_frame_count": 3,
                    "final_visual_valid_frame_count": 5 if success else 0,
                    "object_position_error": 0.001,
                    "target_position_error": 0.0015,
                    "initial_object_confidence": 0.8,
                    "initial_target_confidence": 0.7,
                    "initial_position_spread": 0.0,
                    "initial_object_position_spread": 0.0,
                    "initial_target_position_spread": 0.0,
                    "pregrasp_correction_magnitude": 0.002,
                    "pregrasp_position_spread": 0.0,
                    "final_visual_xy_error": 0.01 if success else "",
                    "final_visual_height_error": 0.001 if success else "",
                    "key_error.final_visual_position_spread": (
                        0.001 if success else ""
                    ),
                    "grasp_candidate": True,
                    "trial_lift_completed": True,
                    "grasp_confirmed": success,
                    "grasp_lost": False,
                    "gripper_aperture_after_close": (
                        0.05 if success else 0.06
                    ),
                    "bilateral_contact_duration": 8.0 if success else 5.8,
                    "contact_loss_event_count": 0,
                    "key_error.aperture_drop": "",
                    "key_error.move_to_pregrasp_position_error": 0.01,
                    "key_error.move_to_pregrasp_orientation_error": 0.01,
                    "key_error.move_to_pregrasp_joint_speed": 0.1,
                    "stage_duration.scene_perception": 0.008,
                    "stage_duration.grasp_confirmation": (
                        0.03 if success else 4.0
                    ),
                    "stage_duration.completed": 0.0 if success else "",
                }
            )
        paired_rows.append(
            {
                "pair_id": pair_id,
                "seed": seed,
                "pair_valid": True,
                "pair_error": "",
                "fingerprint": fingerprint,
                "outcome_category": (
                    "both_success" if b1_success else "oracle_only_success"
                ),
            }
        )
    _write_csv(output / "episodes.csv", episode_rows)
    _write_csv(output / "paired_results.csv", paired_rows)
    _write_csv(
        output / "failure_counts.csv",
        [
            {
                "method_id": method,
                "failure_reason": reason,
                "count": count,
            }
            for (method, reason), count in failure_counts.items()
        ],
    )
    _write_json(
        output / "summary.json",
        {
            "methods": {
                method: {
                    "requested_episodes": 30,
                    "completed_episodes": 30,
                    "program_errors": 0,
                }
                for method in ("b0_oracle", "b1_vision")
            },
            "paired": {
                "valid_pair_count": 30,
                "invalid_pair_count": 0,
                "program_error_pair_count": 0,
                "both_success": 15,
                "oracle_only_success": 15,
                "vision_only_success": 0,
                "both_failed": 0,
            },
        },
    )
    _write_json(
        output / "production_metrics.json",
        {
            "methods": {
                "b0_oracle": {
                    "requested_episode_count": 30,
                    "valid_episode_count": 30,
                    "invalid_numeric_episode_count": 0,
                },
                "b1_vision": {
                    "requested_episode_count": 30,
                    "valid_episode_count": 30,
                    "invalid_numeric_episode_count": 0,
                    "safe_task_success_count": 15,
                    "first_attempt_placement_success_count": 15,
                    "placement_success_count": 15,
                    "collision_episode_count": 0,
                    "unexplained_failure_count": 0,
                },
            }
        },
    )
    log_lines = []
    for index in range(60):
        log_lines.append(f"INFO episode_start execution={index}")
        log_lines.append(f"INFO episode_end execution={index}")
    (output / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    _write_json(
        output / "manual_assessment.json",
        {
            "assessment_kind": "human_evidence_review",
            "decision": {
                "parameter_change_recommended_now": False,
                "reason": "Structured evidence does not justify a change.",
            },
            "evidence": {"reviewed_seed_ids": [1000, 1015]},
        },
    )
    return output


class CalibrationAnalysisTests(unittest.TestCase):
    def test_generates_b1_only_outputs_and_preserves_run_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = _fixture(Path(directory))
            input_files = (*REQUIRED_INPUT_FILES, "manual_assessment.json")
            before = {
                name: _sha256(output / name) for name in input_files
            }
            analysis = analyze_calibration(output)
            after = {name: _sha256(output / name) for name in input_files}

            self.assertEqual(before, after)
            self.assertEqual(
                analysis["manual_assessment"]["assessment_kind"],
                "structured_evidence_review",
            )
            self.assertEqual(
                analysis["manual_assessment"]["evidence"]["reviewed_seed_ids"],
                [1000, 1015],
            )
            source_assessment = json.loads(
                (output / "manual_assessment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                source_assessment["assessment_kind"], "human_evidence_review"
            )
            self.assertEqual(
                analysis["b1_core_metrics"]["safe_task_success_count"], 15
            )
            self.assertEqual(
                analysis["b0_diagnostic_ground_truth_success_count"], 30
            )
            self.assertEqual(
                analysis["pair_diagnostics"]["both_success"]["count"], 15
            )
            self.assertEqual(
                analysis["pair_diagnostics"]["oracle_only_success"]["count"], 15
            )
            for name in (
                "calibration_analysis.json",
                "calibration_analysis.csv",
                "calibration_round_0_report.md",
            ):
                self.assertTrue((output / name).is_file())
            loaded = json.loads(
                (output / "calibration_analysis.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded["integrity"]["status"], "PASS")
            self.assertEqual(
                loaded["manual_assessment"]["assessment_kind"],
                "structured_evidence_review",
            )
            report = (output / "calibration_round_0_report.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("未修改参数", report)
            self.assertNotIn("NaN", report)

    def test_missing_manual_assessment_uses_structured_review_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = _fixture(Path(directory))
            (output / "manual_assessment.json").unlink()

            analysis = analyze_calibration(output)

            self.assertIn("manual_assessment", analysis)
            self.assertEqual(
                analysis["manual_assessment"]["assessment_kind"],
                "structured_evidence_review",
            )

    def test_rejects_invalid_pair_program_error_missing_seed_and_nan(self) -> None:
        mutations = {
            "invalid_pair": lambda output: _rewrite_csv(
                output / "episodes.csv",
                lambda rows: rows[0].update(pair_valid="False"),
            ),
            "program_error": lambda output: _rewrite_csv(
                output / "episodes.csv",
                lambda rows: rows[0].update(program_error="RuntimeError"),
            ),
            "missing_seed": lambda output: _write_json(
                output / "seeds.json",
                {
                    "seeds": list(range(1000, 1029)),
                    "seed_count": 29,
                    "duplicates_present": False,
                    "pilot": False,
                },
            ),
            "nan": lambda output: _rewrite_csv(
                output / "episodes.csv",
                lambda rows: rows[0].update(simulation_time="NaN"),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = _fixture(Path(directory))
                mutate(output)
                with self.assertRaises(CalibrationAnalysisError):
                    analyze_calibration(output)


if __name__ == "__main__":
    unittest.main()
