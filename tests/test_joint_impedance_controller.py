from __future__ import annotations

import unittest

import numpy as np

from controllers.joint_impedance import JointImpedanceController


class JointImpedanceControllerTests(unittest.TestCase):
    def make_controller(
        self,
        *,
        stiffness: float = 10.0,
        damping: float = 2.0,
        torque_limit: float = 100.0,
        rate_limit: float = 1000.0,
    ) -> JointImpedanceController:
        return JointImpedanceController(
            stiffness=np.full(7, stiffness),
            damping=np.full(7, damping),
            torque_limits=np.full(7, torque_limit),
            torque_rate_limits=np.full(7, rate_limit),
        )

    def compute(
        self,
        controller: JointImpedanceController,
        *,
        q: np.ndarray | None = None,
        dq: np.ndarray | None = None,
        q_target: np.ndarray | None = None,
        dq_target: np.ndarray | None = None,
        compensation: np.ndarray | None = None,
        dt: float = 0.01,
    ):
        zero = np.zeros(7)
        return controller.compute(
            q=zero if q is None else q,
            dq=zero if dq is None else dq,
            q_target=zero if q_target is None else q_target,
            dq_target=zero if dq_target is None else dq_target,
            dynamics_compensation=(
                zero if compensation is None else compensation
            ),
            dt=dt,
        )

    def test_zero_error_has_zero_feedback(self) -> None:
        torque, diagnostics = self.compute(self.make_controller())
        np.testing.assert_allclose(torque, 0.0)
        np.testing.assert_allclose(diagnostics.feedback_torque, 0.0)

    def test_position_error_sign_is_correct(self) -> None:
        controller = self.make_controller()
        _, positive = self.compute(controller, q_target=np.ones(7) * 0.1)
        controller.reset()
        _, negative = self.compute(controller, q_target=-np.ones(7) * 0.1)
        self.assertTrue(np.all(positive.feedback_torque > 0.0))
        self.assertTrue(np.all(negative.feedback_torque < 0.0))

    def test_damping_opposes_velocity(self) -> None:
        _, diagnostics = self.compute(
            self.make_controller(), dq=np.ones(7) * 0.2
        )
        self.assertTrue(np.all(diagnostics.feedback_torque < 0.0))

    def test_dynamics_compensation_is_added_exactly_once(self) -> None:
        compensation = np.arange(1.0, 8.0)
        torque, diagnostics = self.compute(
            self.make_controller(), compensation=compensation
        )
        np.testing.assert_allclose(diagnostics.raw_torque, compensation)
        np.testing.assert_allclose(torque, compensation)

    def test_absolute_torque_clipping(self) -> None:
        controller = self.make_controller(
            stiffness=1000.0, torque_limit=3.0, rate_limit=1e9
        )
        torque, diagnostics = self.compute(
            controller, q_target=np.ones(7), dt=0.01
        )
        np.testing.assert_allclose(torque, 3.0)
        self.assertTrue(np.all(diagnostics.saturation_mask))

    def test_torque_rate_limiting(self) -> None:
        controller = self.make_controller(
            stiffness=1000.0, torque_limit=1000.0, rate_limit=5.0
        )
        torque, diagnostics = self.compute(
            controller, q_target=np.ones(7), dt=0.01
        )
        np.testing.assert_allclose(torque, 0.05)
        self.assertTrue(np.all(diagnostics.rate_limit_mask))

    def test_reset_clears_previous_torque(self) -> None:
        controller = self.make_controller(
            stiffness=1000.0, torque_limit=1000.0, rate_limit=5.0
        )
        self.compute(controller, q_target=np.ones(7), dt=0.01)
        np.testing.assert_allclose(controller.previous_torque, 0.05)
        controller.reset()
        np.testing.assert_allclose(controller.previous_torque, 0.0)

    def test_rejects_invalid_configuration_dimensions_and_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            JointImpedanceController(
                stiffness=np.ones(6),
                damping=np.ones(7),
                torque_limits=np.ones(7),
                torque_rate_limits=np.ones(7),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            JointImpedanceController(
                stiffness=np.zeros(7),
                damping=np.ones(7),
                torque_limits=np.ones(7),
                torque_rate_limits=np.ones(7),
            )

    def test_compute_rejects_non_finite_inputs(self) -> None:
        target = np.zeros(7)
        target[0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            self.compute(self.make_controller(), q_target=target)


if __name__ == "__main__":
    unittest.main()
