from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import unittest

import controllers.sensor_event_controller as b1_controller
import evaluation.perception_evaluator as evaluator
from controllers import B1Stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class B1TruthIsolationTests(unittest.TestCase):
    def test_b1_declares_all_explicit_stages(self) -> None:
        self.assertEqual(
            [stage.value for stage in B1Stage],
            [
                "scene_perception",
                "move_to_pregrasp",
                "pregrasp_reacquisition",
                "descend_to_grasp",
                "close_gripper",
                "grasp_candidate_check",
                "trial_lift",
                "grasp_confirmation",
                "transfer",
                "descend_to_place",
                "release",
                "withdraw",
                "final_visual_verification",
                "completed",
            ],
        )

    def test_b1_controller_source_has_no_external_truth_reads(self) -> None:
        source = inspect.getsource(b1_controller)
        for forbidden in (
            "object_body_id",
            "place_target_site_id",
            "current_episode",
            "data.qpos",
            "data.xpos",
            "data.site_xpos",
            "PrivilegedStateProvider",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_privileged_truth_reads_are_confined_to_evaluator(self) -> None:
        source = inspect.getsource(evaluator)
        self.assertIn("env.object_body_id", source)
        self.assertIn("env.place_target_site_id", source)

    def test_cli_rejects_privileged_b1_before_simulation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_pick_place.py"),
                "--config",
                str(CONFIG_PATH),
                "--controller",
                "sensor_event_b1",
                "--observation-source",
                "privileged",
                "--headless",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires observation.source", completed.stderr)


if __name__ == "__main__":
    unittest.main()
