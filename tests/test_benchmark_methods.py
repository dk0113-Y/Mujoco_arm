from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import controllers
import controllers.sensor_event_controller as sensor_event_controller
from benchmark.methods import (
    FORMAL_METHOD_IDS,
    METHOD_SPECS,
    assert_static_fairness,
    resolve_methods,
)
from controllers import B1Stage, SensorEventPickPlaceController
from environments import load_config
from evaluation.perception_evaluator import build_episode_result
from perception import OracleExternalStateProvider, RGBDPerceptionProvider
import scripts.run_pick_place as run_pick_place


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"

EXPECTED_STAGES = (
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
)


class BenchmarkMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH).with_modes(
            observation_source="perception",
            controller_type="sensor_event_b1",
        )

    def test_only_formal_paired_method_ids_are_registered(self) -> None:
        self.assertEqual(FORMAL_METHOD_IDS, ("b0_oracle", "b1_vision"))
        self.assertEqual(tuple(METHOD_SPECS), FORMAL_METHOD_IDS)
        self.assertEqual(
            tuple(spec.method_id for spec in resolve_methods(list(FORMAL_METHOD_IDS))),
            FORMAL_METHOD_IDS,
        )

        for invalid in (
            [],
            ["b0_oracle"],
            ["b1_vision"],
            ["fixed_dls_b0", "b1_vision"],
            ["b0_oracle", "b0_oracle", "b1_vision"],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_methods(invalid)

    def test_methods_share_controller_class_and_exact_config_objects(self) -> None:
        methods = resolve_methods(list(FORMAL_METHOD_IDS))
        self.assertTrue(
            all(
                method.controller_class is SensorEventPickPlaceController
                for method in methods
            )
        )
        self.assertTrue(
            all(method.controller_type == "sensor_event_b1" for method in methods)
        )

        controllers_for_methods = tuple(
            method.controller_class(self.config.controller, self.config.b1)
            for method in methods
        )
        self.assertTrue(
            all(
                controller.controller_config is self.config.controller
                for controller in controllers_for_methods
            )
        )
        self.assertTrue(
            all(controller.config is self.config.b1 for controller in controllers_for_methods)
        )
        self.assertIs(
            controllers_for_methods[0].controller_config,
            controllers_for_methods[1].controller_config,
        )
        self.assertIs(
            controllers_for_methods[0].config,
            controllers_for_methods[1].config,
        )

    def test_provider_classes_and_external_source_labels_are_exact(self) -> None:
        oracle = METHOD_SPECS["b0_oracle"]
        vision = METHOD_SPECS["b1_vision"]
        self.assertIs(oracle.provider_type, OracleExternalStateProvider)
        self.assertIs(vision.provider_type, RGBDPerceptionProvider)
        self.assertEqual(oracle.external_state_source, "oracle")
        self.assertEqual(vision.external_state_source, "vision")
        self.assertIs(oracle.ground_truth_evaluator, build_episode_result)
        self.assertIs(vision.ground_truth_evaluator, build_episode_result)
        assert_static_fairness(resolve_methods(list(FORMAL_METHOD_IDS)), self.config)

    def test_both_methods_have_the_one_complete_b1_stage_sequence(self) -> None:
        self.assertEqual(tuple(stage.value for stage in B1Stage), EXPECTED_STAGES)
        for method in resolve_methods(list(FORMAL_METHOD_IDS)):
            with self.subTest(method=method.method_id):
                self.assertIs(method.controller_class, SensorEventPickPlaceController)
                self.assertEqual(tuple(stage.value for stage in B1Stage), EXPECTED_STAGES)
        self.assertFalse(hasattr(controllers, "SensorEventOracleController"))

    def test_static_fairness_rejects_method_specific_controller_or_config_modes(self) -> None:
        methods = resolve_methods(list(FORMAL_METHOD_IDS))
        unfair_methods = (
            replace(methods[0], controller_class=controllers.FixedDLSPickPlaceController),
            methods[1],
        )
        with self.assertRaisesRegex(ValueError, "SensorEventPickPlaceController"):
            assert_static_fairness(unfair_methods, self.config)

        privileged = replace(
            self.config,
            observation=replace(self.config.observation, source="privileged"),
        )
        with self.assertRaisesRegex(ValueError, "truth-isolated"):
            assert_static_fairness(methods, privileged)

    def test_normal_single_episode_entrypoint_does_not_expose_oracle(self) -> None:
        source = inspect.getsource(run_pick_place)
        self.assertNotIn("OracleExternalStateProvider", source)
        for arguments in (
            ["run_pick_place.py", "--observation-source", "oracle"],
            ["run_pick_place.py", "--controller", "b0_oracle"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(sys, "argv", arguments),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                run_pick_place.parse_args()

    def test_vision_controller_path_has_no_external_truth_reads(self) -> None:
        source = inspect.getsource(sensor_event_controller)
        for forbidden in (
            "object_body_id",
            "place_target_site_id",
            "current_episode",
            "data.qpos",
            "data.xpos",
            "data.site_xpos",
            "PrivilegedStateProvider",
            "OracleExternalStateProvider",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

        self.assertEqual(
            SensorEventPickPlaceController.external_state_sources,
            frozenset({"perception", "oracle"}),
        )
        self.assertEqual(self.config.observation.source, "perception")


if __name__ == "__main__":
    unittest.main()
