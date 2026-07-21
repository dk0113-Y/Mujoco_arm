from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from benchmark.methods import METHOD_SPECS
from benchmark.pairing import EpisodeFingerprint
from benchmark.runner import _EpisodeExecution, _episode_row
from evaluation import EpisodeResult
from evaluation.protocol import load_protocol
import evaluation.split_analysis as split_analysis
import scripts.generate_protocol_splits as split_script


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"


class ProtocolBenchmarkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)

    def test_protocol_fields_extend_and_preserve_benchmark_episode_row(self) -> None:
        result = EpisodeResult(
            seed=42,
            pick_mode="random",
            place_mode="random",
            physics_mode="random",
            pick_region="front",
            place_region="left",
            sampled_pick_position=(0.5, 0.1, 0.246),
            sampled_place_position=(0.2, 0.57, 0.222),
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
            observation_source="perception",
            controller_type="sensor_event_b1",
            controller_reported_success=True,
            privileged_ground_truth_success=True,
            false_positive=False,
            false_negative=False,
        )
        fingerprint = EpisodeFingerprint.from_episode_result(result)
        execution = _EpisodeExecution(
            pair_id="pair_0000_seed_42",
            seed=42,
            method=METHOD_SPECS["b1_vision"],
            execution_index=1,
            result=result,
            fingerprint=fingerprint,
            pair_valid=True,
        )
        row = _episode_row(
            execution,
            protocol=self.protocol,
            split_name="calibration_smoke",
            config_sha256="b" * 64,
            code_commit="c" * 40,
        )
        self.assertEqual(row["protocol_id"], "evaluation_protocol")
        self.assertEqual(row["protocol_version"], "1.0.0")
        self.assertEqual(row["split_name"], "calibration_smoke")
        self.assertTrue(row["placement_success"])
        self.assertTrue(row["safe_task_success"])
        self.assertFalse(row["unexplained_failure"])
        self.assertEqual(row["method_id"], "b1_vision")
        self.assertEqual(row["episode_fingerprint"], fingerprint.digest)
        self.assertIn("success", row)

    def test_split_selection_source_has_no_controller_or_outcome_filter(self) -> None:
        source = inspect.getsource(split_analysis) + inspect.getsource(split_script)
        for forbidden in (
            "SensorEventPickPlaceController",
            "b0_oracle",
            "b1_vision",
            "failure_reason",
            ".success",
            "solve_pose_ik",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
