from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controllers.sensor_event_controller import (
    B1ControllerFailure,
    B1Runtime,
    B1Stage,
    EventMotionPlan,
    SensorEventPickPlaceController,
)
from environments.config import load_config
from evaluation import FailureReason


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


def motion_observation(
    position_error: float,
    joint_speed: float,
) -> dict[str, np.ndarray]:
    return {
        "tcp_position": np.array([position_error, 0.0, 0.0]),
        "tcp_orientation": np.eye(3),
        "arm_joint_velocities": np.full(7, joint_speed),
    }


class FakeMotionEnv:
    def __init__(self, observations: list[dict[str, np.ndarray]]) -> None:
        self.data = SimpleNamespace(time=0.0, ctrl=np.zeros(8, dtype=float))
        self.arm_actuator_ids = np.arange(7)
        self._observations = list(observations)
        self.step_count = 0

    def step(self, action: np.ndarray):
        del action
        self.step_count += 1
        self.data.time += 0.01
        return {}, 0.0, False, False, {}

    def observation(self) -> dict[str, np.ndarray]:
        if not self._observations:
            raise AssertionError("Motion loop requested an unexpected observation")
        return self._observations.pop(0)


class B1MotionEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_config = load_config(CONFIG_PATH)

    def make_controller(self, *, hold_steps: int = 3):
        b1_config = replace(
            self.base_config.b1,
            arrival_position_tolerance=0.1,
            arrival_orientation_tolerance=0.1,
            settled_joint_velocity_threshold=1.0,
            arrival_hold_steps=hold_steps,
            motion_timeout=1.0,
        )
        return SensorEventPickPlaceController(
            self.base_config.controller,
            b1_config,
        )

    @staticmethod
    def runtime() -> B1Runtime:
        return B1Runtime(
            stage=B1Stage.MOVE_TO_PREGRASP,
            target_rotation=np.eye(3),
        )

    @staticmethod
    def plan() -> EventMotionPlan:
        return EventMotionPlan(
            start_time=0.0,
            reference_duration=1.0,
            start_control=np.zeros(7),
            target_control=np.ones(7),
            target_position=np.zeros(3),
        )

    def test_strict_then_hysteresis_samples_accumulate_arrival_hold(self) -> None:
        controller = self.make_controller(hold_steps=3)
        runtime = self.runtime()
        env = FakeMotionEnv(
            [
                motion_observation(0.09, 0.9),
                motion_observation(0.12, 1.2),
                motion_observation(0.11, 1.1),
            ]
        )

        with patch.object(controller, "_motion_plan", return_value=self.plan()):
            controller._move_until_arrived(
                env,
                runtime,
                target_position=np.zeros(3),
                reference_duration=1.0,
                step_callback=None,
            )

        self.assertEqual(env.step_count, 3)
        self.assertAlmostEqual(runtime.key_errors["waypoint_error"], 0.11)
        self.assertAlmostEqual(runtime.key_errors["settled_joint_speed"], 1.1)

    def test_pose_outside_hysteresis_at_timeout_is_motion_stage_timeout(self) -> None:
        controller = self.make_controller()
        runtime = self.runtime()
        env = FakeMotionEnv([motion_observation(0.2, 0.0)])

        with patch.object(controller, "_motion_plan", return_value=self.plan()):
            with self.assertRaises(B1ControllerFailure) as caught:
                controller._move_until_arrived(
                    env,
                    runtime,
                    target_position=np.zeros(3),
                    reference_duration=1.0,
                    step_callback=None,
                    timeout=0.0,
                )

        self.assertEqual(caught.exception.reason, FailureReason.MOTION_STAGE_TIMEOUT)
        self.assertAlmostEqual(caught.exception.errors["waypoint_error"], 0.2)

    def test_settling_timeout_with_pose_in_tolerance_is_motion_not_settled(self) -> None:
        controller = self.make_controller()
        runtime = self.runtime()
        env = FakeMotionEnv([motion_observation(0.05, 2.0)])

        with patch.object(controller, "_motion_plan", return_value=self.plan()):
            with self.assertRaises(B1ControllerFailure) as caught:
                controller._move_until_arrived(
                    env,
                    runtime,
                    target_position=np.zeros(3),
                    reference_duration=1.0,
                    step_callback=None,
                    timeout=0.0,
                )

        self.assertEqual(caught.exception.reason, FailureReason.MOTION_NOT_SETTLED)
        self.assertAlmostEqual(caught.exception.errors["waypoint_error"], 0.05)
        self.assertAlmostEqual(caught.exception.errors["settled_joint_speed"], 2.0)


if __name__ == "__main__":
    unittest.main()
