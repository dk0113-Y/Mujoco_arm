from __future__ import annotations

import copy
import math
from pathlib import Path
import unittest

from benchmark.pairing import EpisodeFingerprint
from scripts import analyze_development as analysis


def _episode_rows() -> tuple[list[str], list[dict[str, str]], list[int]]:
    seeds = list(range(60))
    rows: list[dict[str, str]] = []
    for pair_index, seed in enumerate(seeds):
        task = {
            "seed": seed,
            "sampled_pick_position": [0.4, 0.1, 0.246],
            "sampled_place_position": [0.1, 0.55, 0.222],
            "pick_region": "front",
            "place_region": "left",
            "sampled_mass": 0.1,
            "sampled_friction": [1.0, 0.01, 0.001],
        }
        digest = EpisodeFingerprint.from_episode_result(task).digest
        for method_index, method in enumerate(analysis.METHOD_IDS):
            rows.append(
                {
                    "pair_id": f"pair_{pair_index:04d}_seed_{seed}",
                    "method_id": method,
                    "execution_index": str(2 * pair_index + method_index),
                    "episode_fingerprint": digest,
                    "pair_valid": "True",
                    "program_error": "",
                    "seed": str(seed),
                    "sampled_pick_position": "[0.4,0.1,0.246]",
                    "sampled_place_position": "[0.1,0.55,0.222]",
                    "pick_region": "front",
                    "place_region": "left",
                    "sampled_mass": "0.1",
                    "sampled_friction": "[1.0,0.01,0.001]",
                    "simulation_time": "10.0",
                    "collision_count": "0",
                    "controller_reported_success": "True",
                    "privileged_ground_truth_success": "True",
                    "placement_success": "True",
                    "safe_task_success": "True",
                    "first_attempt_placement_success": "True",
                    "collision_episode": "False",
                    "unexplained_failure": "False",
                    "result_fields_complete": "True",
                    "failure_reason": "",
                    "final_stage": "completed",
                    "split_name": "development",
                    "config_sha256": analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                    "code_commit": "c" * 40,
                }
            )
    return list(rows[0]), rows, seeds


def _pair_rows(seeds: list[int]) -> list[dict[str, str]]:
    return [
        {
            "seed": str(seed),
            "pair_valid": "True",
            "pair_error": "",
            "outcome_category": "both_success",
        }
        for seed in seeds
    ]


def _strata(seeds: list[int]) -> dict[int, dict[str, object]]:
    return {
        seed: {
            "seed": seed,
            "pick_region": ("front", "left", "right")[seed % 3],
            "place_region": ("left", "right", "front")[seed % 3],
            "region_pair": f"{('front', 'left', 'right')[seed % 3]}->{('left', 'right', 'front')[seed % 3]}",
            "same_cross": "cross_region",
            "mass": 0.05 + 0.002 * seed,
            "friction": [1.0, 0.01, 0.001],
            "pick_place_distance": 0.2 + 0.01 * seed,
            "mass_bin": {"name": f"Q{seed % 4 + 1}"},
            "sliding_friction_bin": {"name": f"Q{seed % 4 + 1}"},
            "torsional_friction_bin": {"name": f"Q{seed % 4 + 1}"},
            "rolling_friction_bin": {"name": f"Q{seed % 4 + 1}"},
            "pick_place_distance_bin": {"name": f"Q{seed % 4 + 1}"},
        }
        for seed in seeds
    }


class StatisticalHelperTests(unittest.TestCase):
    def test_wilson_interval_known_inputs(self) -> None:
        zero = analysis.wilson_interval(0, 10)
        self.assertAlmostEqual(zero["lower"], 0.0)
        self.assertAlmostEqual(zero["upper"], 0.2775327998628892)
        half = analysis.wilson_interval(5, 10)
        self.assertAlmostEqual(half["lower"], 0.236593090512564)
        self.assertAlmostEqual(half["upper"], 0.7634069094874361)
        self.assertEqual(analysis.wilson_interval(0, 0), {"lower": None, "upper": None})

    def test_exact_paired_statistic_known_inputs(self) -> None:
        result = analysis.exact_paired_binomial(9, 1)
        self.assertEqual(result["discordant_pair_count"], 10)
        self.assertAlmostEqual(result["exact_two_sided_p_value"], 0.021484375)
        self.assertEqual(result["label"], "Development exploratory statistic")
        self.assertEqual(analysis.exact_paired_binomial(0, 0)["exact_two_sided_p_value"], 1.0)


class AnalysisClassificationTests(unittest.TestCase):
    def test_pregrasp_frames_are_required_only_after_stage_is_reached(self) -> None:
        early_motion_failure = {
            "failure_reason": "ik_not_converged",
            "final_stage": "move_to_pregrasp",
            "initial_valid_frame_count": "5",
            "pregrasp_valid_frame_count": "0",
            "controller_reported_success": "False",
            "safe_task_success": "False",
        }
        self.assertEqual(
            analysis._perception_group(early_motion_failure),
            "normal_perception_but_later_failure",
        )

        rejected_pregrasp = {
            **early_motion_failure,
            "failure_reason": "pregrasp_reacquisition_failed",
            "final_stage": "pregrasp_reacquisition",
        }
        self.assertEqual(
            analysis._perception_group(rejected_pregrasp),
            "perception_unavailable_or_rejected",
        )

    def test_calibration_comparison_uses_controller_success_count(self) -> None:
        calibration_b1 = {
            "failure_reason_counts": {"success": 17, "grasp_not_confirmed": 13},
            "safe_task_success_count": 17,
            "safe_task_success_rate": 17 / 30,
            "safe_successful_simulation_time_count": 17,
            "safe_successful_simulation_time_median": 18.0,
            "safe_successful_simulation_time_mean": 18.0,
            "safe_successful_simulation_time_minimum": 17.0,
            "safe_successful_simulation_time_maximum": 19.0,
        }
        archive = {
            "calibration": {
                "production_metrics": {
                    "methods": {
                        "b0_oracle": {
                            "safe_task_success_count": 20,
                            "safe_task_success_rate": 20 / 30,
                        },
                        "b1_vision": calibration_b1,
                    }
                },
                "summary": {
                    "paired": {
                        "both_failed": 9,
                        "both_success": 16,
                        "oracle_only_success": 4,
                        "vision_only_success": 1,
                    }
                },
            }
        }
        core = {
            "b0_oracle": {},
            "b1_vision": {"valid_episode_count": 60},
        }
        pair_analysis = {
            "categories": {
                category: {"count": 0} for category in analysis.PAIR_CATEGORIES
            }
        }
        failure_analysis = {
            "total_b1_failures": 28,
            "failure_reasons": {"grasp_not_confirmed": {"count": 28}},
        }

        comparison = analysis._calibration_comparison(
            archive, core, pair_analysis, failure_analysis
        )
        self.assertEqual(
            comparison["development"]["b1_failure_reason_counts"]["success"], 32
        )


class DevelopmentArchiveValidatorTests(unittest.TestCase):
    def test_episode_validator_accepts_complete_coverage(self) -> None:
        fields, rows, seeds = _episode_rows()
        analysis.validate_episode_rows(
            fields,
            rows,
            seeds,
            config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
            code_commit="c" * 40,
        )

    def test_episode_validator_rejects_missing_duplicate_invalid_error_and_nonfinite(self) -> None:
        fields, rows, seeds = _episode_rows()
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "120"):
            analysis.validate_episode_rows(
                fields,
                rows[:-2],
                seeds,
                config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                code_commit="c" * 40,
            )
        duplicate = copy.deepcopy(rows)
        duplicate[2]["seed"] = duplicate[0]["seed"]
        with self.assertRaises(analysis.DevelopmentAnalysisError):
            analysis.validate_episode_rows(
                fields,
                duplicate,
                seeds,
                config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                code_commit="c" * 40,
            )
        invalid = copy.deepcopy(rows)
        invalid[0]["pair_valid"] = "False"
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "invalid pair"):
            analysis.validate_episode_rows(
                fields,
                invalid,
                seeds,
                config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                code_commit="c" * 40,
            )
        error = copy.deepcopy(rows)
        error[0]["program_error"] = "unexpected_exception"
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "program error"):
            analysis.validate_episode_rows(
                fields,
                error,
                seeds,
                config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                code_commit="c" * 40,
            )
        nonfinite = copy.deepcopy(rows)
        nonfinite[0]["simulation_time"] = "NaN"
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "NaN or Inf"):
            analysis.validate_episode_rows(
                fields,
                nonfinite,
                seeds,
                config_sha256=analysis.EXPECTED_FROZEN_CONFIG_SHA256,
                code_commit="c" * 40,
            )

    def test_pair_validator_rejects_missing_duplicate_invalid_and_program_category(self) -> None:
        seeds = list(range(60))
        rows = _pair_rows(seeds)
        analysis.validate_pair_rows(rows, seeds)
        for mutation in ("missing", "duplicate", "invalid", "program"):
            current = copy.deepcopy(rows)
            if mutation == "missing":
                current.pop()
            elif mutation == "duplicate":
                current[1]["seed"] = current[0]["seed"]
            elif mutation == "invalid":
                current[0]["pair_valid"] = "False"
            else:
                current[0]["outcome_category"] = "program_error"
            with self.subTest(mutation=mutation), self.assertRaises(
                analysis.DevelopmentAnalysisError
            ):
                analysis.validate_pair_rows(current, seeds)

    def test_strata_validator_rejects_hash_mismatch_and_dirty_generation(self) -> None:
        seeds = list(range(60))
        document = {
            "split_name": "development",
            "seed_count": 60,
            "input_files": {
                "protocol_sha256": analysis.EXPECTED_PROTOCOL_SHA256,
                "seeds_file_sha256": analysis.EXPECTED_DEVELOPMENT_SPLIT_SHA256,
            },
            "controller_outcomes_used": False,
            "calibration_outcomes_used": False,
            "development_outcomes_used": False,
            "held_out_data_read": False,
            "renderer_created": False,
            "ik_executed": False,
            "generation_git_dirty": False,
            "generation_code_commit": "c" * 40,
            "strata": list(_strata(seeds).values()),
        }
        analysis._strata_map(document, seeds, "c" * 40)
        bad_hash = copy.deepcopy(document)
        bad_hash["input_files"]["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "protocol hash"):
            analysis._strata_map(bad_hash, seeds, "c" * 40)
        dirty = copy.deepcopy(document)
        dirty["generation_git_dirty"] = True
        with self.assertRaisesRegex(analysis.DevelopmentAnalysisError, "generation_git_dirty"):
            analysis._strata_map(dirty, seeds, "c" * 40)


class EvidenceAndCandidateTests(unittest.TestCase):
    @staticmethod
    def _dataset():
        seeds = list(range(12))
        strata = _strata(seeds)
        episodes = {}
        pairs = {}
        for seed in seeds:
            b1_reason = (
                "grasp_not_confirmed"
                if seed < 4
                else "initial_perception_failed"
                if seed < 7
                else "grasp_lost_during_transfer"
                if seed == 7
                else "success"
            )
            b0_reason = "grasp_not_confirmed" if seed < 3 else "success"
            episodes[seed] = {
                "b0_oracle": {
                    "seed": str(seed),
                    "controller_reported_success": str(b0_reason == "success"),
                    "failure_reason": "" if b0_reason == "success" else b0_reason,
                    "safe_task_success": str(b0_reason == "success"),
                },
                "b1_vision": {
                    "seed": str(seed),
                    "controller_reported_success": str(b1_reason == "success"),
                    "failure_reason": "" if b1_reason == "success" else b1_reason,
                    "safe_task_success": str(b1_reason == "success"),
                    "initial_valid_frame_count": "0" if "perception" in b1_reason else "5",
                },
            }
            pairs[seed] = {
                "outcome_category": (
                    "both_failed"
                    if b0_reason != "success" and b1_reason != "success"
                    else "oracle_only_success"
                    if b0_reason == "success" and b1_reason != "success"
                    else "vision_only_success"
                    if b0_reason != "success" and b1_reason == "success"
                    else "both_success"
                )
            }
        return seeds, strata, episodes, pairs

    def test_evidence_matrix_has_required_schema_and_allowed_strength(self) -> None:
        seeds, strata, episodes, pairs = self._dataset()
        b1 = [episodes[seed]["b1_vision"] for seed in seeds]
        categories = {seed: pairs[seed]["outcome_category"] for seed in seeds}
        matrix = analysis._evidence_matrix(
            b1=b1,
            episodes_by_seed=episodes,
            strata=strata,
            pair_category=categories,
        )
        self.assertGreaterEqual(len(matrix["problem_families"]), 5)
        required = {
            "problem_family",
            "observed_failure_count",
            "affected_seeds",
            "share_of_all_b1_failures",
            "pair_category_evidence",
            "regional_pattern",
            "mass_friction_distance_pattern",
            "formal_online_observability",
            "privileged_evidence_only",
            "candidate_mechanism",
            "expected_core_metric",
            "expected_secondary_metrics",
            "safety_risk",
            "possible_regressions",
            "implementation_scope",
            "confounders",
            "evidence_strength",
            "diagnostic_needed",
            "priority",
        }
        for item in matrix["problem_families"]:
            self.assertFalse(required - set(item))
            self.assertIn(
                item["evidence_strength"], {"strong", "moderate", "weak", "insufficient"}
            )

    def test_diagnostic_candidates_are_deterministic_unique_and_bounded(self) -> None:
        seeds, strata, episodes, pairs = self._dataset()
        first = analysis.select_diagnostic_candidates(
            seeds=seeds,
            strata=strata,
            episodes_by_seed=episodes,
            pair_by_seed=pairs,
        )
        second = analysis.select_diagnostic_candidates(
            seeds=seeds,
            strata=strata,
            episodes_by_seed=episodes,
            pair_by_seed=pairs,
        )
        self.assertEqual(first, second)
        candidate_seeds = [item["seed"] for item in first["candidates"]]
        self.assertEqual(len(candidate_seeds), len(set(candidate_seeds)))
        self.assertGreaterEqual(len(candidate_seeds), 6)
        self.assertLessEqual(len(candidate_seeds), 12)
        self.assertEqual(first["d0_5_episodes_run"], 0)
        for item in first["candidates"]:
            self.assertEqual(
                len(item["supported_problem_families"]),
                len(set(item["supported_problem_families"])),
            )
            if "discordant_pair" in item["selection_roles"]:
                self.assertIn(
                    item["pair_category"],
                    {"oracle_only_success", "vision_only_success"},
                )
            if "matched_success_control" in item["selection_roles"]:
                self.assertTrue(
                    analysis._bool(
                        episodes[item["seed"]]["b1_vision"]["safe_task_success"],
                        "safe",
                    )
                )

    def test_source_writes_only_declared_analysis_outputs_and_no_held_out_path(self) -> None:
        source = Path(analysis.__file__).read_text(encoding="utf-8")
        self.assertNotIn("held_out_test_v1.txt", source)
        self.assertNotIn("mujoco.Renderer", source)
        self.assertNotIn("run_episode(", source)
        self.assertIn("formal_raw_files_unchanged", source)
        self.assertIn("pooled_success_rate_reported", source)


if __name__ == "__main__":
    unittest.main()
