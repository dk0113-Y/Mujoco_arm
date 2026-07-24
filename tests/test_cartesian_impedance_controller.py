from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from control_benchmarks.cartesian_config import load_cartesian_config
from control_benchmarks.dynamics import MuJoCoDynamicsProvider
from control_benchmarks.kinematics import (
    PandaTcpKinematicsProvider,
    orientation_error_world,
    quaternion_wxyz_to_rotation,
    rotation_to_quaternion_wxyz,
    rotation_vector_to_matrix,
)
from controllers.cartesian_impedance import CartesianImpedanceController
from environments.panda_torque_env import PandaTorqueEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "ci_baseline_v1.toml"


class CartesianImpedanceControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        stiffness: float = 10.0,
        damping: float = 2.0,
        torque_limit: float = 100.0,
        rate_limit: float = 1000.0,
    ) -> CartesianImpedanceController:
        return CartesianImpedanceController(
            translational_stiffness=np.full(3, stiffness),
            rotational_stiffness=np.full(3, stiffness),
            translational_damping=np.full(3, damping),
            rotational_damping=np.full(3, damping),
            torque_limits=np.full(7, torque_limit),
            torque_rate_limits=np.full(7, rate_limit),
        )

    def compute(
        self,
        controller: CartesianImpedanceController,
        *,
        target_position: np.ndarray | None = None,
        target_rotation: np.ndarray | None = None,
        linear_velocity: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
        target_linear_velocity: np.ndarray | None = None,
        target_angular_velocity: np.ndarray | None = None,
        jacobian: np.ndarray | None = None,
        compensation: np.ndarray | None = None,
        dt: float = 0.01,
    ):
        zero3 = np.zeros(3)
        zero7 = np.zeros(7)
        identity_jacobian = np.zeros((6, 7))
        identity_jacobian[:, :6] = np.eye(6)
        return controller.compute(
            position=zero3,
            rotation=np.eye(3),
            linear_velocity=(
                zero3 if linear_velocity is None else linear_velocity
            ),
            angular_velocity=(
                zero3 if angular_velocity is None else angular_velocity
            ),
            target_position=(
                zero3 if target_position is None else target_position
            ),
            target_rotation=(
                np.eye(3) if target_rotation is None else target_rotation
            ),
            target_linear_velocity=(
                zero3
                if target_linear_velocity is None
                else target_linear_velocity
            ),
            target_angular_velocity=(
                zero3
                if target_angular_velocity is None
                else target_angular_velocity
            ),
            jacobian=(
                identity_jacobian if jacobian is None else jacobian
            ),
            dynamics_compensation=(
                zero7 if compensation is None else compensation
            ),
            dt=dt,
        )

    def test_zero_pose_and_twist_error_has_zero_task_torque(self) -> None:
        torque, diagnostics = self.compute(self.make_controller())
        np.testing.assert_allclose(diagnostics.task_wrench, 0.0)
        np.testing.assert_allclose(diagnostics.task_torque, 0.0)
        np.testing.assert_allclose(torque, 0.0)

    def test_translation_target_directions_are_positive(self) -> None:
        for axis in range(3):
            controller = self.make_controller(rate_limit=1e9)
            target = np.zeros(3)
            target[axis] = 0.01
            _, diagnostics = self.compute(
                controller, target_position=target
            )
            self.assertGreater(diagnostics.task_wrench[axis], 0.0)
            self.assertGreater(diagnostics.task_torque[axis], 0.0)

    def test_rotation_target_directions_are_positive(self) -> None:
        for axis in range(3):
            controller = self.make_controller(rate_limit=1e9)
            vector = np.zeros(3)
            vector[axis] = 0.02
            _, diagnostics = self.compute(
                controller,
                target_rotation=rotation_vector_to_matrix(vector),
            )
            self.assertGreater(diagnostics.task_wrench[3 + axis], 0.0)
            self.assertGreater(diagnostics.task_torque[3 + axis], 0.0)

    def test_jacobian_transpose_mapping_is_exact(self) -> None:
        rng = np.random.default_rng(123)
        jacobian = rng.normal(size=(6, 7))
        controller = CartesianImpedanceController(
            translational_stiffness=np.ones(3),
            rotational_stiffness=np.ones(3),
            translational_damping=np.zeros(3),
            rotational_damping=np.zeros(3),
            torque_limits=np.full(7, 1e6),
            torque_rate_limits=np.full(7, 1e9),
        )
        target_position = np.asarray([0.1, -0.2, 0.3])
        target_rotation = rotation_vector_to_matrix(
            np.asarray([0.03, -0.02, 0.01])
        )
        _, diagnostics = self.compute(
            controller,
            target_position=target_position,
            target_rotation=target_rotation,
            jacobian=jacobian,
        )
        np.testing.assert_allclose(
            diagnostics.task_torque,
            jacobian.T @ diagnostics.task_wrench,
            atol=1e-14,
        )

    def test_dynamics_compensation_is_added_exactly_once(self) -> None:
        compensation = np.arange(1.0, 8.0)
        torque, diagnostics = self.compute(
            self.make_controller(rate_limit=1e9),
            compensation=compensation,
        )
        np.testing.assert_allclose(diagnostics.task_torque, 0.0)
        np.testing.assert_allclose(diagnostics.raw_torque, compensation)
        np.testing.assert_allclose(torque, compensation)

    def test_absolute_torque_clipping(self) -> None:
        controller = self.make_controller(
            stiffness=1000.0, torque_limit=0.5, rate_limit=1e9
        )
        _, diagnostics = self.compute(
            controller, target_position=np.ones(3)
        )
        self.assertTrue(np.any(diagnostics.saturation_mask))
        self.assertTrue(np.all(np.abs(diagnostics.final_torque) <= 0.5))

    def test_torque_rate_limiting_and_reset(self) -> None:
        controller = self.make_controller(
            stiffness=1000.0, torque_limit=1000.0, rate_limit=5.0
        )
        torque, diagnostics = self.compute(
            controller, target_position=np.ones(3), dt=0.01
        )
        self.assertTrue(np.any(diagnostics.rate_limit_mask))
        self.assertLessEqual(float(np.max(np.abs(torque))), 0.05 + 1e-12)
        self.assertGreater(float(np.linalg.norm(controller.previous_torque)), 0.0)
        controller.reset()
        np.testing.assert_allclose(controller.previous_torque, 0.0)

    def test_quaternion_target_sign_flip_keeps_control_output(self) -> None:
        quaternion = rotation_to_quaternion_wxyz(
            rotation_vector_to_matrix(np.asarray([0.1, -0.05, 0.02]))
        )
        controller = self.make_controller(rate_limit=1e9)
        first, _ = self.compute(
            controller,
            target_rotation=quaternion_wxyz_to_rotation(quaternion),
        )
        controller.reset()
        second, _ = self.compute(
            controller,
            target_rotation=quaternion_wxyz_to_rotation(-quaternion),
        )
        np.testing.assert_allclose(first, second, atol=1e-14)

    def test_non_finite_and_invalid_rotation_inputs_are_rejected(self) -> None:
        target = np.zeros(3)
        target[0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.compute(self.make_controller(), target_position=target)
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.compute(
                self.make_controller(), target_rotation=np.ones((3, 3))
            )


class CartesianClosedLoopDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_cartesian_config(CONFIG_PATH)

    def run_response(
        self, *, translation_axis: int | None, rotation_axis: int | None
    ) -> tuple[float, float]:
        env = PandaTorqueEnv(self.config)
        try:
            env.reset(
                qpos=self.config.trajectory.initial_joint_pose,
                qvel=np.zeros(7),
                seed=self.config.seed,
            )
            provider = PandaTcpKinematicsProvider(
                env.model,
                env.arm_dof_addresses,
                site_name=self.config.model.tcp_site,
                rank_tolerance=(
                    self.config.kinematics.jacobian_rank_tolerance
                ),
            )
            initial = provider.compute(env.data)
            controller = CartesianImpedanceController(
                translational_stiffness=np.asarray(
                    self.config.controller.translational_stiffness
                ),
                rotational_stiffness=np.asarray(
                    self.config.controller.rotational_stiffness
                ),
                translational_damping=np.asarray(
                    self.config.controller.translational_damping
                ),
                rotational_damping=np.asarray(
                    self.config.controller.rotational_damping
                ),
                torque_limits=np.asarray(
                    self.config.controller.torque_limits
                ),
                torque_rate_limits=np.asarray(
                    self.config.controller.torque_rate_limits
                ),
            )
            dynamics = MuJoCoDynamicsProvider(
                env.model,
                env.arm_qpos_addresses,
                env.arm_dof_addresses,
                mode=self.config.controller.dynamics_compensation_mode,
            )
            # Establish the same controller's gravity-compensation command
            # before isolating a small Cartesian target direction.  The formal
            # benchmark intentionally logs the zero-torque startup transient;
            # this direction test measures the local closed-loop response.
            for _ in range(500):
                state = provider.compute(env.data)
                terms = dynamics.compute(env.data)
                torque, _ = controller.compute(
                    position=state.position,
                    rotation=state.rotation,
                    linear_velocity=state.linear_velocity,
                    angular_velocity=state.angular_velocity,
                    target_position=initial.position,
                    target_rotation=initial.rotation,
                    target_linear_velocity=np.zeros(3),
                    target_angular_velocity=np.zeros(3),
                    jacobian=state.jacobian,
                    dynamics_compensation=terms.compensation,
                    dt=env.control_period,
                )
                env.step(torque)
            settled = provider.compute(env.data)
            target_position = settled.position.copy()
            target_rotation = settled.rotation.copy()
            if translation_axis is not None:
                target_position[translation_axis] += 0.005
                initial_error = 0.005
            else:
                vector = np.zeros(3)
                vector[int(rotation_axis)] = 0.03
                target_rotation = (
                    rotation_vector_to_matrix(vector) @ settled.rotation
                )
                initial_error = 0.03
            for _ in range(250):
                state = provider.compute(env.data)
                terms = dynamics.compute(env.data)
                torque, _ = controller.compute(
                    position=state.position,
                    rotation=state.rotation,
                    linear_velocity=state.linear_velocity,
                    angular_velocity=state.angular_velocity,
                    target_position=target_position,
                    target_rotation=target_rotation,
                    target_linear_velocity=np.zeros(3),
                    target_angular_velocity=np.zeros(3),
                    jacobian=state.jacobian,
                    dynamics_compensation=terms.compensation,
                    dt=env.control_period,
                )
                env.step(torque)
            final = provider.compute(env.data)
            if translation_axis is not None:
                final_error = abs(
                    target_position[translation_axis]
                    - final.position[translation_axis]
                )
            else:
                final_error = float(
                    np.linalg.norm(
                        orientation_error_world(
                            final.rotation, target_rotation
                        )
                    )
                )
            return initial_error, final_error
        finally:
            env.close()

    def test_world_translation_axes_reduce_short_time_error(self) -> None:
        for axis in range(3):
            initial, final = self.run_response(
                translation_axis=axis, rotation_axis=None
            )
            self.assertLess(final, initial, f"world translation axis {axis}")

    def test_world_rotation_axes_reduce_short_time_error(self) -> None:
        for axis in range(3):
            initial, final = self.run_response(
                translation_axis=None, rotation_axis=axis
            )
            self.assertLess(final, initial, f"world rotation axis {axis}")


if __name__ == "__main__":
    unittest.main()
