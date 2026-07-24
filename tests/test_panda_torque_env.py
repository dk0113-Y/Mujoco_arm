from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from control_benchmarks.config import load_control_config
from control_benchmarks.dynamics import MuJoCoDynamicsProvider
from environments.panda_torque_env import PandaTorqueEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "control" / "ji_baseline_v1.toml"


class PandaTorqueEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_control_config(CONFIG_PATH)
        self.env = PandaTorqueEnv(self.config)

    def tearDown(self) -> None:
        self.env.close()

    def test_reset_is_repeatable(self) -> None:
        first = self.env.reset(seed=123)
        first_sequence = []
        for _ in range(3):
            observation, _ = self.env.step(np.full(7, 0.25))
            first_sequence.append(observation["joint_positions"].copy())
        second = self.env.reset(seed=123)
        second_sequence = []
        for _ in range(3):
            observation, _ = self.env.step(np.full(7, 0.25))
            second_sequence.append(observation["joint_positions"].copy())
        np.testing.assert_array_equal(
            first["joint_positions"], second["joint_positions"]
        )
        np.testing.assert_array_equal(first_sequence, second_sequence)

    def test_step_accepts_only_seven_torques(self) -> None:
        self.env.reset()
        for invalid in (np.zeros(6), np.zeros(8), np.zeros((7, 1))):
            with self.subTest(shape=invalid.shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    self.env.step(invalid)

    def test_step_rejects_nan_and_inf(self) -> None:
        self.env.reset()
        for value in (float("nan"), float("inf"), float("-inf")):
            action = np.zeros(7)
            action[3] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    self.env.step(action)

    def test_action_is_direct_torque_not_position_target(self) -> None:
        self.env.reset()
        observation, diagnostics = self.env.step(np.ones(7))
        np.testing.assert_allclose(self.env.data.ctrl[:7], 1.0)
        np.testing.assert_allclose(diagnostics["actuator_force"], 1.0)
        np.testing.assert_allclose(
            observation["applied_generalized_force"], 1.0
        )

    def test_observation_and_diagnostics_fields_are_complete(self) -> None:
        observation = self.env.reset()
        expected_observation = {
            "joint_positions",
            "joint_velocities",
            "simulation_time",
            "actuator_force",
            "applied_generalized_force",
            "joint_limit_margin",
            "control_cycle",
        }
        self.assertEqual(set(observation), expected_observation)
        _, diagnostics = self.env.step(np.zeros(7))
        expected_diagnostics = {
            "commanded_torque",
            "rate_limited_torque",
            "clipped_torque",
            "actuator_force",
            "saturation_mask",
            "torque_rate_limit_mask",
            "joint_limit_mask",
            "velocity_limit_mask",
            "tracking_error_mask",
            "simulation_instability_mask",
            "finite_value_status",
            "termination_reason",
        }
        self.assertEqual(set(diagnostics), expected_diagnostics)

    def test_control_period_is_model_timestep_times_substeps(self) -> None:
        self.assertAlmostEqual(self.env.control_period, 0.002, places=12)
        self.assertAlmostEqual(
            self.env.control_period,
            self.env.model.opt.timestep * self.config.simulation.substeps,
            places=12,
        )

    def test_rate_and_absolute_limits_are_observable(self) -> None:
        high_rate_controller = replace(
            self.config.controller,
            torque_rate_limits=(1e9,) * 7,
        )
        fast_violation = replace(
            self.config.safety,
            sustained_violation_duration=self.env.control_period,
        )
        config = replace(
            self.config,
            controller=high_rate_controller,
            safety=fast_violation,
        )
        env = PandaTorqueEnv(config)
        try:
            env.reset()
            _, diagnostics = env.step(np.full(7, 1e6))
            self.assertTrue(np.all(diagnostics["saturation_mask"]))
            np.testing.assert_allclose(
                np.abs(diagnostics["clipped_torque"]),
                self.config.controller.torque_limits,
            )
            self.assertEqual(
                diagnostics["termination_reason"],
                "torque_saturation_sustained",
            )
        finally:
            env.close()

    def test_velocity_safety_terminates_structurally(self) -> None:
        safety = replace(
            self.config.safety,
            joint_velocity_limits=(1e-8,) * 7,
        )
        env = PandaTorqueEnv(replace(self.config, safety=safety))
        try:
            env.reset()
            _, diagnostics = env.step(np.zeros(7))
            self.assertEqual(
                diagnostics["termination_reason"], "joint_velocity_limit"
            )
            self.assertTrue(np.any(diagnostics["velocity_limit_mask"]))
        finally:
            env.close()


class DynamicsProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_control_config(CONFIG_PATH)
        self.env = PandaTorqueEnv(self.config)
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_zero_velocity_separates_gravity_from_velocity_bias(self) -> None:
        provider = MuJoCoDynamicsProvider(
            self.env.model,
            self.env.arm_qpos_addresses,
            self.env.arm_dof_addresses,
        )
        terms = provider.compute(self.env.data)
        np.testing.assert_allclose(terms.coriolis_centrifugal, 0.0, atol=1e-12)
        np.testing.assert_allclose(terms.compensation, terms.gravity)
        self.assertGreater(np.linalg.norm(terms.gravity), 1.0)

    def test_nonzero_velocity_reconstructs_bias_and_keeps_passive_separate(self) -> None:
        dq = np.linspace(-0.3, 0.3, 7)
        self.env.reset(qvel=dq)
        provider = MuJoCoDynamicsProvider(
            self.env.model,
            self.env.arm_qpos_addresses,
            self.env.arm_dof_addresses,
        )
        terms = provider.compute(self.env.data)
        np.testing.assert_allclose(
            terms.gravity + terms.coriolis_centrifugal,
            terms.compensation,
            atol=1e-12,
        )
        np.testing.assert_allclose(terms.passive, -dq, atol=1e-12)

    def test_compensation_modes_are_explicit(self) -> None:
        outputs = {}
        for mode in ("none", "gravity", "gravity_coriolis"):
            provider = MuJoCoDynamicsProvider(
                self.env.model,
                self.env.arm_qpos_addresses,
                self.env.arm_dof_addresses,
                mode=mode,
            )
            outputs[mode] = provider.compute(self.env.data)
        np.testing.assert_allclose(outputs["none"].compensation, 0.0)
        np.testing.assert_allclose(
            outputs["gravity"].compensation, outputs["gravity"].gravity
        )
        np.testing.assert_allclose(
            outputs["gravity_coriolis"].compensation,
            outputs["gravity_coriolis"].gravity
            + outputs["gravity_coriolis"].coriolis_centrifugal,
        )


if __name__ == "__main__":
    unittest.main()
