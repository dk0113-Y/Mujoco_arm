from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

from control_benchmarks.cartesian_config import load_cartesian_config
from control_benchmarks.kinematics import (
    PandaTcpKinematicsProvider,
    controllability_termination_reason,
    rotation_matrix_log_world,
)
from environments.panda_torque_env import PandaTorqueEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "ci_baseline_v1.toml"


class PandaTcpKinematicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_cartesian_config(CONFIG_PATH)
        self.env = PandaTorqueEnv(self.config)
        self.env.reset(
            qpos=self.config.trajectory.initial_joint_pose,
            qvel=np.zeros(7),
            seed=self.config.seed,
        )
        self.provider = PandaTcpKinematicsProvider(
            self.env.model,
            self.env.arm_dof_addresses,
            site_name=self.config.model.tcp_site,
            rank_tolerance=self.config.kinematics.jacobian_rank_tolerance,
        )

    def tearDown(self) -> None:
        self.env.close()

    def state_at(self, pose: np.ndarray):
        self.env.data.qpos[self.env.arm_qpos_addresses] = pose
        self.env.data.qvel[self.env.arm_dof_addresses] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        return self.provider.compute(self.env.data)

    def test_site_exists_and_relative_hand_transform_is_fixed(self) -> None:
        site_id = self.provider.site_id
        hand_id = int(
            mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
            )
        )
        self.assertGreaterEqual(site_id, 0)
        for pose in self.config.trajectory.hold_joint_poses:
            self.state_at(np.asarray(pose, dtype=float))
            hand_rotation = np.asarray(
                self.env.data.xmat[hand_id], dtype=float
            ).reshape(3, 3)
            hand_position = np.asarray(
                self.env.data.xpos[hand_id], dtype=float
            )
            site_rotation = np.asarray(
                self.env.data.site_xmat[site_id], dtype=float
            ).reshape(3, 3)
            site_position = np.asarray(
                self.env.data.site_xpos[site_id], dtype=float
            )
            local_position = hand_rotation.T @ (
                site_position - hand_position
            )
            local_rotation = hand_rotation.T @ site_rotation
            np.testing.assert_allclose(
                local_position, [0.0, 0.0, 0.103], atol=1e-12
            )
            np.testing.assert_allclose(local_rotation, np.eye(3), atol=1e-12)

    def test_world_pose_quaternion_and_arrays_are_finite(self) -> None:
        state = self.provider.compute(self.env.data)
        self.assertTrue(np.all(np.isfinite(state.position)))
        self.assertTrue(np.all(np.isfinite(state.rotation)))
        self.assertTrue(np.all(np.isfinite(state.quaternion_wxyz)))
        self.assertAlmostEqual(
            float(np.linalg.norm(state.quaternion_wxyz)), 1.0, places=14
        )

    def test_jacobian_shape_order_and_joint_columns(self) -> None:
        state = self.provider.compute(self.env.data)
        self.assertEqual(state.jacobian.shape, (6, 7))
        np.testing.assert_array_equal(
            self.provider.arm_dof_addresses, self.env.arm_dof_addresses
        )
        joint_names = tuple(
            mujoco.mj_id2name(
                self.env.model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
            )
            for joint_id in self.env.arm_joint_ids
        )
        self.assertEqual(
            joint_names, tuple(f"joint{index}" for index in range(1, 8))
        )

    def test_translation_jacobian_matches_central_difference(self) -> None:
        base = np.asarray(
            self.config.trajectory.initial_joint_pose, dtype=float
        )
        epsilon = 1e-6
        analytic = self.state_at(base).jacobian[:3]
        errors = []
        for joint in range(7):
            plus = base.copy()
            minus = base.copy()
            plus[joint] += epsilon
            minus[joint] -= epsilon
            derivative = (
                self.state_at(plus).position - self.state_at(minus).position
            ) / (2.0 * epsilon)
            errors.append(float(np.max(np.abs(derivative - analytic[:, joint]))))
        self.assertLess(max(errors), 2e-7)

    def test_rotation_jacobian_matches_so3_central_difference(self) -> None:
        base = np.asarray(
            self.config.trajectory.initial_joint_pose, dtype=float
        )
        epsilon = 1e-6
        analytic = self.state_at(base).jacobian[3:]
        errors = []
        for joint in range(7):
            plus = base.copy()
            minus = base.copy()
            plus[joint] += epsilon
            minus[joint] -= epsilon
            plus_rotation = self.state_at(plus).rotation
            minus_rotation = self.state_at(minus).rotation
            derivative = rotation_matrix_log_world(
                plus_rotation @ minus_rotation.T
            ) / (2.0 * epsilon)
            errors.append(float(np.max(np.abs(derivative - analytic[:, joint]))))
        self.assertLess(max(errors), 2e-7)

    def test_twist_matches_jacobian_and_mujoco_site_velocity(self) -> None:
        dq = np.linspace(-0.3, 0.3, 7)
        self.env.reset(
            qpos=self.config.trajectory.initial_joint_pose,
            qvel=dq,
            seed=self.config.seed,
        )
        state = self.provider.compute(self.env.data)
        np.testing.assert_allclose(
            state.twist,
            state.jacobian @ dq,
            atol=1e-14,
        )
        np.testing.assert_allclose(state.twist, state.site_twist, atol=1e-14)
        self.assertLessEqual(state.twist_consistency_error, 1e-14)

    def test_virtual_power_identity(self) -> None:
        state = self.provider.compute(self.env.data)
        dq = np.linspace(-0.2, 0.25, 7)
        wrench = np.asarray([1.2, -0.7, 0.4, 0.3, -0.2, 0.9])
        joint_power = float(dq @ (state.jacobian.T @ wrench))
        task_power = float((state.jacobian @ dq) @ wrench)
        self.assertAlmostEqual(joint_power, task_power, places=14)

    def test_rank_condition_and_structured_termination(self) -> None:
        state = self.provider.compute(self.env.data)
        self.assertEqual(state.rank, 6)
        self.assertGreater(state.minimum_singular_value, 0.1)
        self.assertLess(state.condition_number, 20.0)
        self.assertIsNone(
            controllability_termination_reason(
                state,
                minimum_rank=6,
                minimum_singular_value=0.05,
                maximum_condition_number=50.0,
            )
        )
        rank_deficient = replace(state, rank=5)
        self.assertEqual(
            controllability_termination_reason(
                rank_deficient,
                minimum_rank=6,
                minimum_singular_value=0.05,
                maximum_condition_number=50.0,
            ),
            "jacobian_rank_deficient",
        )
        ill_conditioned = replace(state, condition_number=100.0)
        self.assertEqual(
            controllability_termination_reason(
                ill_conditioned,
                minimum_rank=6,
                minimum_singular_value=0.05,
                maximum_condition_number=50.0,
            ),
            "jacobian_condition_exceeded",
        )


if __name__ == "__main__":
    unittest.main()
