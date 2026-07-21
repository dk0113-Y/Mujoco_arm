from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from evaluation.production_metrics import (
    build_production_metrics,
    derive_episode_protocol_fields,
)
from evaluation.protocol import load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"


def episode(**changes):
    row = {
        "seed": 7,
        "method_id": "b1_vision",
        "pair_valid": True,
        "program_error": None,
        "episode_fingerprint": "a" * 64,
        "pick_region": "front",
        "place_region": "left",
        "sampled_pick_position": (0.5, 0.1, 0.246),
        "sampled_place_position": (0.2, 0.57, 0.222),
        "sampled_mass": 0.1,
        "sampled_friction": (1.0, 0.01, 0.001),
        "final_stage": "completed",
        "simulation_time": 10.0,
        "collision_count": 0,
        "controller_reported_success": True,
        "privileged_ground_truth_success": True,
        "failure_reason": None,
    }
    row.update(changes)
    return row


class SuccessSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_placement_and_safe_success_are_distinct(self) -> None:
        safe = derive_episode_protocol_fields(episode(), self.protocol)
        self.assertTrue(safe["placement_success"])
        self.assertTrue(safe["safe_task_success"])

        collision = derive_episode_protocol_fields(episode(collision_count=1), self.protocol)
        self.assertTrue(collision["placement_success"])
        self.assertFalse(collision["safe_task_success"])
        self.assertTrue(collision["collision_episode"])

        failed = derive_episode_protocol_fields(
            episode(
                privileged_ground_truth_success=False,
                controller_reported_success=False,
                failure_reason="bilateral_contact_missing",
                final_stage="grasp_candidate_check",
            ),
            self.protocol,
        )
        self.assertFalse(failed["placement_success"])
        self.assertFalse(failed["safe_task_success"])
        self.assertFalse(failed["unexplained_failure"])

    def test_program_error_timeout_and_missing_field_are_unsafe_or_unexplained(self) -> None:
        program = derive_episode_protocol_fields(
            episode(program_error="resource cleanup failed"), self.protocol
        )
        self.assertFalse(program["safe_task_success"])
        self.assertTrue(program["unexplained_failure"])
        timeout = derive_episode_protocol_fields(
            episode(
                simulation_time=36.0,
                controller_reported_success=False,
                failure_reason="timeout",
            ),
            self.protocol,
        )
        self.assertFalse(timeout["placement_success"])
        self.assertFalse(timeout["safe_task_success"])
        missing = episode()
        del missing["sampled_mass"]
        self.assertTrue(
            derive_episode_protocol_fields(missing, self.protocol)["unexplained_failure"]
        )


class ProductionMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_core_counts_denominators_rates_and_times(self) -> None:
        rows = [
            episode(simulation_time=10.0),
            episode(seed=8, simulation_time=12.0, collision_count=1),
            episode(
                seed=9,
                final_stage="close_gripper",
                controller_reported_success=False,
                privileged_ground_truth_success=False,
                failure_reason="empty_gripper_closure",
            ),
            episode(seed=10, program_error="unexpected exception"),
            episode(seed=11, pair_valid=False),
            episode(seed=12),
        ]
        del rows[-1]["sampled_mass"]
        pairs = [
            {"outcome_category": "both_success"},
            {"outcome_category": "oracle_only_success"},
        ]
        metrics = build_production_metrics(rows, pairs, protocol=self.protocol)
        self.assertEqual(metrics["requested_episode_count"], 6)
        self.assertEqual(metrics["valid_episode_count"], 4)
        self.assertEqual(metrics["placement_success_count"], 3)
        self.assertEqual(metrics["placement_success_rate"], 0.75)
        self.assertEqual(metrics["safe_task_success_count"], 1)
        self.assertEqual(metrics["safe_task_success_rate"], 0.25)
        self.assertEqual(metrics["first_attempt_placement_success_count"], 3)
        self.assertEqual(metrics["collision_episode_count"], 1)
        self.assertEqual(metrics["collision_episode_rate"], 0.25)
        self.assertEqual(metrics["safe_successful_simulation_time_count"], 1)
        self.assertEqual(metrics["safe_successful_simulation_time_median"], 10.0)
        self.assertEqual(metrics["safe_successful_simulation_time_mean"], 10.0)
        self.assertEqual(metrics["unexplained_failure_count"], 3)
        self.assertEqual(metrics["unexplained_failure_rate"], 0.5)
        self.assertEqual(metrics["oracle_vision_pair_counts"]["both_success"], 1)

    def test_empty_and_non_finite_data_are_explicitly_handled(self) -> None:
        empty = build_production_metrics([], protocol=self.protocol)
        self.assertEqual(empty["valid_episode_count"], 0)
        self.assertIsNone(empty["safe_task_success_rate"])
        self.assertIsNone(empty["safe_successful_simulation_time_median"])
        bad = episode(simulation_time=float("nan"))
        metrics = build_production_metrics([bad], protocol=self.protocol)
        self.assertEqual(metrics["invalid_numeric_episode_count"], 1)
        self.assertEqual(metrics["unexplained_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
