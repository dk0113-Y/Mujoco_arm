from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest

import mujoco
import numpy as np

from environments import PandaUTableEnv
from sensors import GripperFeedbackSensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class GripperFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PandaUTableEnv(CONFIG_PATH)
        self.env.reset(seed=42)
        self.sensor = GripperFeedbackSensor(self.env.model, self.env.data)

    def tearDown(self) -> None:
        self.env.close()

    def _step_gripper(self, control: float, steps: int) -> None:
        self.env.data.ctrl[self.env.gripper_actuator_id] = control
        for _ in range(steps):
            mujoco.mj_step(self.env.model, self.env.data)

    def test_open_feedback_uses_verified_summed_joint_travel(self) -> None:
        feedback = self.sensor.read("open")

        self.assertAlmostEqual(self.sensor.minimum_aperture, 0.0, places=12)
        self.assertAlmostEqual(self.sensor.maximum_aperture, 0.08, places=12)
        self.assertAlmostEqual(
            feedback.aperture,
            feedback.left_finger_position + feedback.right_finger_position,
            places=10,
        )
        self.assertAlmostEqual(feedback.aperture, 0.08, places=6)
        self.assertEqual(feedback.commanded_state, "open")
        self.assertEqual(feedback.timestamp, self.env.data.time)
        self.assertTrue(
            np.all(
                np.isfinite(
                    [
                        feedback.left_finger_position,
                        feedback.right_finger_position,
                        feedback.aperture,
                        feedback.aperture_velocity,
                        feedback.timestamp,
                    ]
                )
            )
        )
        with self.assertRaises(FrozenInstanceError):
            feedback.aperture = 0.0  # type: ignore[misc]

    def test_velocity_tracks_closing_and_opening_motion(self) -> None:
        self._step_gripper(0.0, 25)
        closing = self.sensor.read("closing")
        self.assertLess(closing.aperture_velocity, 0.0)
        self.assertGreater(closing.aperture, 0.0)
        self.assertLess(closing.aperture, self.sensor.maximum_aperture)

        self._step_gripper(0.0, 1_000)
        closed = self.sensor.read("closed")
        self.assertLess(closed.aperture, 1e-5)
        self.assertLess(abs(closed.aperture_velocity), 1e-6)

        self._step_gripper(255.0, 25)
        opening = self.sensor.read("opening")
        self.assertGreater(opening.aperture_velocity, 0.0)
        self.assertGreater(opening.aperture, closed.aperture)

    def test_aperture_clamps_soft_limit_solver_overshoot(self) -> None:
        self.env.data.qpos[self.sensor.left_qpos_address] = -1e-3
        self.env.data.qpos[self.sensor.right_qpos_address] = -1e-3
        feedback = GripperFeedbackSensor(
            self.env.model, self.env.data
        ).read("closed")
        self.assertEqual(feedback.aperture, 0.0)

        self.env.data.qpos[self.sensor.left_qpos_address] = 0.05
        self.env.data.qpos[self.sensor.right_qpos_address] = 0.05
        feedback = GripperFeedbackSensor(
            self.env.model, self.env.data
        ).read("open")
        self.assertEqual(feedback.aperture, 0.08)

    def test_missing_joint_name_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing required object"):
            GripperFeedbackSensor(
                self.env.model,
                self.env.data,
                left_joint_name="missing_left_finger_joint",
            )


if __name__ == "__main__":
    unittest.main()
