from __future__ import annotations

import unittest

from benchmark.summary import build_summary, failure_counts_rows


ORACLE = "b0_oracle"
VISION = "b1_vision"


def episode_row(
    method_id: str,
    *,
    ground_truth_success: bool,
    controller_success: bool,
    simulation_time: float,
    failure_reason: str | None = None,
    collision_count: int = 0,
    pair_valid: bool = True,
    program_error: str | None = None,
) -> dict[str, object]:
    return {
        "method_id": method_id,
        "pair_valid": pair_valid,
        "program_error": program_error,
        "privileged_ground_truth_success": ground_truth_success,
        "controller_reported_success": controller_success,
        "false_positive": controller_success and not ground_truth_success,
        "false_negative": not controller_success and ground_truth_success,
        "collision_count": collision_count,
        "simulation_time": simulation_time,
        "failure_reason": failure_reason,
    }


class BenchmarkSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episode_rows = [
            episode_row(
                ORACLE,
                ground_truth_success=True,
                controller_success=True,
                simulation_time=10.0,
            ),
            episode_row(
                VISION,
                ground_truth_success=True,
                controller_success=True,
                simulation_time=11.0,
            ),
            episode_row(
                ORACLE,
                ground_truth_success=True,
                controller_success=False,
                simulation_time=20.0,
                failure_reason="grasp_candidate_failed",
                collision_count=1,
            ),
            episode_row(
                VISION,
                ground_truth_success=False,
                controller_success=False,
                simulation_time=21.0,
                failure_reason="perception_object_not_found",
            ),
            episode_row(
                ORACLE,
                ground_truth_success=False,
                controller_success=True,
                simulation_time=30.0,
            ),
            episode_row(
                VISION,
                ground_truth_success=True,
                controller_success=False,
                simulation_time=31.0,
                failure_reason="motion_stage_timeout",
                collision_count=1,
            ),
            episode_row(
                ORACLE,
                ground_truth_success=False,
                controller_success=False,
                simulation_time=40.0,
                failure_reason="dropped_object",
                collision_count=2,
            ),
            episode_row(
                VISION,
                ground_truth_success=False,
                controller_success=True,
                simulation_time=41.0,
            ),
            # Invalid pairs and program errors are requested, but are not completed
            # method episodes and must not contaminate rates or failure counts.
            episode_row(
                ORACLE,
                ground_truth_success=True,
                controller_success=True,
                simulation_time=50.0,
                pair_valid=False,
            ),
            episode_row(
                VISION,
                ground_truth_success=True,
                controller_success=True,
                simulation_time=51.0,
                pair_valid=False,
            ),
            episode_row(
                ORACLE,
                ground_truth_success=False,
                controller_success=False,
                simulation_time=0.0,
                program_error="RuntimeError: oracle crashed",
            ),
            episode_row(
                VISION,
                ground_truth_success=False,
                controller_success=False,
                simulation_time=0.0,
                program_error="RuntimeError: vision crashed",
            ),
        ]
        self.paired_rows = [
            {"pair_valid": True, "outcome_category": "both_success"},
            {"pair_valid": True, "outcome_category": "oracle_only_success"},
            {"pair_valid": True, "outcome_category": "vision_only_success"},
            {"pair_valid": True, "outcome_category": "both_failed"},
            {"pair_valid": False, "outcome_category": "invalid_pair"},
            {"pair_valid": False, "outcome_category": "program_error"},
        ]

    def test_method_counts_rates_errors_collisions_and_times(self) -> None:
        summary = build_summary(
            self.episode_rows,
            self.paired_rows,
            (ORACLE, VISION),
            requested_episode_count=6,
        )

        oracle = summary["methods"][ORACLE]
        self.assertEqual(oracle["requested_episodes"], 6)
        self.assertEqual(oracle["completed_episodes"], 4)
        self.assertEqual(oracle["program_errors"], 1)
        self.assertEqual(oracle["ground_truth_success_count"], 2)
        self.assertEqual(oracle["ground_truth_success_rate"], 0.5)
        self.assertEqual(oracle["controller_reported_success_count"], 2)
        self.assertEqual(oracle["controller_reported_success_rate"], 0.5)
        self.assertEqual(oracle["false_positive_count"], 1)
        self.assertEqual(oracle["false_negative_count"], 1)
        self.assertEqual(oracle["collision_episode_count"], 2)
        self.assertEqual(oracle["successful_episode_simulation_time_mean"], 15.0)
        self.assertEqual(oracle["successful_episode_simulation_time_median"], 15.0)
        self.assertEqual(
            oracle["failure_reason_counts"],
            {
                "dropped_object": 1,
                "grasp_candidate_failed": 1,
                "success": 2,
            },
        )

        vision = summary["methods"][VISION]
        self.assertEqual(vision["requested_episodes"], 6)
        self.assertEqual(vision["completed_episodes"], 4)
        self.assertEqual(vision["program_errors"], 1)
        self.assertEqual(vision["ground_truth_success_count"], 2)
        self.assertEqual(vision["ground_truth_success_rate"], 0.5)
        self.assertEqual(vision["controller_reported_success_count"], 2)
        self.assertEqual(vision["controller_reported_success_rate"], 0.5)
        self.assertEqual(vision["false_positive_count"], 1)
        self.assertEqual(vision["false_negative_count"], 1)
        self.assertEqual(vision["collision_episode_count"], 1)
        self.assertEqual(vision["successful_episode_simulation_time_mean"], 21.0)
        self.assertEqual(vision["successful_episode_simulation_time_median"], 21.0)

    def test_paired_outcome_categories_are_counted_without_invalid_pairs(self) -> None:
        paired = build_summary(
            self.episode_rows,
            self.paired_rows,
            (ORACLE, VISION),
        )["paired"]
        self.assertEqual(
            paired,
            {
                "valid_pair_count": 4,
                "both_success": 1,
                "oracle_only_success": 1,
                "vision_only_success": 1,
                "both_failed": 1,
                "invalid_pair_count": 1,
                "program_error_pair_count": 1,
            },
        )

    def test_failure_count_rows_are_sorted_and_exclude_invalid_or_error_rows(self) -> None:
        rows = failure_counts_rows(self.episode_rows, (ORACLE, VISION))
        self.assertEqual(
            rows,
            [
                {
                    "method_id": ORACLE,
                    "failure_reason": "dropped_object",
                    "count": 1,
                },
                {
                    "method_id": ORACLE,
                    "failure_reason": "grasp_candidate_failed",
                    "count": 1,
                },
                {"method_id": ORACLE, "failure_reason": "success", "count": 2},
                {
                    "method_id": VISION,
                    "failure_reason": "motion_stage_timeout",
                    "count": 1,
                },
                {
                    "method_id": VISION,
                    "failure_reason": "perception_object_not_found",
                    "count": 1,
                },
                {"method_id": VISION, "failure_reason": "success", "count": 2},
            ],
        )

    def test_empty_method_has_zero_counts_and_null_rates_and_times(self) -> None:
        method = build_summary([], [], (ORACLE,))["methods"][ORACLE]
        self.assertEqual(method["requested_episodes"], 0)
        self.assertEqual(method["completed_episodes"], 0)
        self.assertEqual(method["program_errors"], 0)
        self.assertIsNone(method["ground_truth_success_rate"])
        self.assertIsNone(method["controller_reported_success_rate"])
        self.assertIsNone(method["successful_episode_simulation_time_mean"])
        self.assertIsNone(method["successful_episode_simulation_time_median"])
        self.assertEqual(method["failure_reason_counts"], {})

    def test_requested_count_defaults_to_rows_for_each_method(self) -> None:
        summary = build_summary(
            self.episode_rows,
            self.paired_rows,
            (ORACLE, VISION),
        )
        self.assertEqual(summary["methods"][ORACLE]["requested_episodes"], 6)
        self.assertEqual(summary["methods"][VISION]["requested_episodes"], 6)


if __name__ == "__main__":
    unittest.main()
