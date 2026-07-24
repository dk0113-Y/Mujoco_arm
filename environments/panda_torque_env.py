"""Isolated direct-torque Panda environment.

This environment intentionally has no inheritance or action compatibility with
``PandaUTableEnv``.  Its only arm action is a seven-element vector in N·m.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from control_benchmarks.config import (
    ControlBenchmarkConfig,
    load_control_config,
)


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))
GRIPPER_ACTUATOR_NAME = "actuator8"


def _required_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"Torque model is missing required object: {name}")
    return object_id


def load_torque_model(path: str | Path) -> mujoco.MjModel:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Torque-control MJCF does not exist: {model_path}")
    return mujoco.MjModel.from_xml_path(str(model_path))


class PandaTorqueEnv:
    """Headless deterministic seven-joint direct torque simulation."""

    def __init__(
        self, config: ControlBenchmarkConfig | str | Path
    ) -> None:
        self.config = (
            load_control_config(config)
            if isinstance(config, (str, Path))
            else config
        )
        self.model = load_torque_model(self.config.model.path)
        self.data = mujoco.MjData(self.model)
        self.arm_joint_ids = np.asarray(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self.arm_actuator_ids = np.asarray(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ],
            dtype=int,
        )
        self.gripper_actuator_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self.arm_qpos_addresses = np.asarray(
            self.model.jnt_qposadr[self.arm_joint_ids], dtype=int
        )
        self.arm_dof_addresses = np.asarray(
            self.model.jnt_dofadr[self.arm_joint_ids], dtype=int
        )
        self.arm_joint_ranges = np.asarray(
            self.model.jnt_range[self.arm_joint_ids], dtype=float
        ).copy()
        self.control_period = (
            float(self.model.opt.timestep) * self.config.simulation.substeps
        )
        expected_period = self.config.simulation.control_period
        if not math.isclose(
            self.control_period, expected_period, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "Configured control period does not equal model timestep * "
                f"substeps: {expected_period} != {self.control_period}"
            )
        self._validate_actuator_contract()
        self._previous_torque = np.zeros(7, dtype=float)
        self._tracking_target: np.ndarray | None = None
        self._saturation_streak = 0
        self._rate_limit_streak = 0
        self._cycle = 0
        self._episode_start_time = 0.0
        self._termination_reason: str | None = None
        self.current_seed: int | None = None
        self.rng = np.random.default_rng(self.config.seed)
        self._closed = False

    def _validate_actuator_contract(self) -> None:
        if self.model.nu != 8:
            raise RuntimeError(f"Torque model must have exactly 8 actuators, got {self.model.nu}")
        if not np.array_equal(self.arm_actuator_ids, np.arange(7, dtype=int)):
            raise RuntimeError("Arm direct motors must be the first seven actuators")
        if self.gripper_actuator_id != 7:
            raise RuntimeError("The independent gripper actuator must have id 7")
        for index, (actuator_id, joint_id) in enumerate(
            zip(self.arm_actuator_ids, self.arm_joint_ids)
        ):
            if int(self.model.actuator_trntype[actuator_id]) != int(
                mujoco.mjtTrn.mjTRN_JOINT
            ):
                raise RuntimeError(f"actuator{index + 1} is not a joint transmission")
            if int(self.model.actuator_trnid[actuator_id, 0]) != int(joint_id):
                raise RuntimeError(
                    f"actuator{index + 1} is not mapped to joint{index + 1}"
                )
            if not np.isclose(self.model.actuator_gear[actuator_id, 0], 1.0):
                raise RuntimeError(f"actuator{index + 1} must have gear=1")
            if not np.isclose(self.model.actuator_gainprm[actuator_id, 0], 1.0):
                raise RuntimeError(f"actuator{index + 1} must have fixed gain 1")
            if np.any(
                np.abs(self.model.actuator_biasprm[actuator_id, :3]) > 1e-12
            ):
                raise RuntimeError(f"actuator{index + 1} must have zero bias")
        mjcf_limits = np.asarray(
            self.model.actuator_forcerange[self.arm_actuator_ids], dtype=float
        )
        expected_limits = np.asarray(
            self.config.controller.torque_limits, dtype=float
        )
        if np.any(mjcf_limits[:, 0] > -expected_limits + 1e-12) or np.any(
            mjcf_limits[:, 1] < expected_limits - 1e-12
        ):
            raise RuntimeError("Configured torque limits exceed MJCF force ranges")

    @property
    def terminated(self) -> bool:
        return self._termination_reason is not None

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def set_tracking_target(self, q_target: np.ndarray | None) -> None:
        if q_target is None:
            self._tracking_target = None
            return
        target = np.asarray(q_target, dtype=float)
        if target.shape != (7,):
            raise ValueError(f"q_target must have shape (7,), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("q_target contains NaN or Inf")
        self._tracking_target = target.copy()

    def reset(
        self,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed PandaTorqueEnv")
        if seed is None:
            seed = self.config.seed
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self.current_seed = int(seed)
        self.rng = np.random.default_rng(self.current_seed)
        mujoco.mj_resetData(self.model, self.data)
        home_id = _required_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.data.qpos[:] = self.model.key_qpos[home_id]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.model.key_ctrl[home_id]

        if qpos is not None:
            qpos_value = np.asarray(qpos, dtype=float)
            if qpos_value.shape != (7,):
                raise ValueError(f"qpos must have shape (7,), got {qpos_value.shape}")
            if not np.all(np.isfinite(qpos_value)):
                raise ValueError("qpos contains NaN or Inf")
            self.data.qpos[self.arm_qpos_addresses] = qpos_value
        if qvel is not None:
            qvel_value = np.asarray(qvel, dtype=float)
            if qvel_value.shape != (7,):
                raise ValueError(f"qvel must have shape (7,), got {qvel_value.shape}")
            if not np.all(np.isfinite(qvel_value)):
                raise ValueError("qvel contains NaN or Inf")
            self.data.qvel[self.arm_dof_addresses] = qvel_value

        q = np.asarray(self.data.qpos[self.arm_qpos_addresses], dtype=float)
        margin = self._joint_limit_margin(q)
        if np.any(margin <= self.config.safety.joint_limit_margin):
            raise ValueError("reset qpos violates a configured soft joint limit")
        velocity = np.asarray(self.data.qvel[self.arm_dof_addresses], dtype=float)
        if np.any(
            np.abs(velocity)
            >= np.asarray(self.config.safety.joint_velocity_limits, dtype=float)
        ):
            raise ValueError("reset qvel violates a configured velocity limit")

        self.data.ctrl[self.arm_actuator_ids] = 0.0
        self._previous_torque[:] = 0.0
        self._tracking_target = None
        self._saturation_streak = 0
        self._rate_limit_streak = 0
        self._cycle = 0
        self._termination_reason = None
        self._episode_start_time = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)
        if not self._finite_state():
            raise RuntimeError("reset produced a non-finite MuJoCo state")
        return self.observation()

    def _joint_limit_margin(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(q - self.arm_joint_ranges[:, 0], self.arm_joint_ranges[:, 1] - q)

    def _finite_state(self) -> bool:
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.actuator_force,
            self.data.qfrc_actuator,
        )
        return all(np.all(np.isfinite(array)) for array in arrays)

    def observation(self) -> dict[str, Any]:
        q = np.asarray(self.data.qpos[self.arm_qpos_addresses], dtype=float).copy()
        return {
            "joint_positions": q,
            "joint_velocities": np.asarray(
                self.data.qvel[self.arm_dof_addresses], dtype=float
            ).copy(),
            "simulation_time": float(self.data.time),
            "actuator_force": np.asarray(
                self.data.actuator_force[self.arm_actuator_ids], dtype=float
            ).copy(),
            "applied_generalized_force": np.asarray(
                self.data.qfrc_actuator[self.arm_dof_addresses], dtype=float
            ).copy(),
            "joint_limit_margin": self._joint_limit_margin(q),
            "control_cycle": int(self._cycle),
        }

    def step(
        self, joint_torque: np.ndarray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot step a closed PandaTorqueEnv")
        if self.terminated:
            raise RuntimeError(
                f"Cannot step a terminated episode: {self._termination_reason}"
            )
        commanded = np.asarray(joint_torque, dtype=float)
        if commanded.shape != (7,):
            raise ValueError(
                f"joint_torque must have shape (7,), got {commanded.shape}"
            )
        if not np.all(np.isfinite(commanded)):
            raise ValueError("joint_torque contains NaN or Inf")

        rate_limits = np.asarray(
            self.config.controller.torque_rate_limits, dtype=float
        )
        maximum_delta = rate_limits * self.control_period
        rate_limited = np.clip(
            commanded,
            self._previous_torque - maximum_delta,
            self._previous_torque + maximum_delta,
        )
        rate_mask = np.abs(rate_limited - commanded) > 1e-12
        torque_limits = np.asarray(
            self.config.controller.torque_limits, dtype=float
        )
        clipped = np.clip(rate_limited, -torque_limits, torque_limits)
        saturation_mask = np.abs(clipped - rate_limited) > 1e-12
        self._previous_torque = clipped.copy()
        self.data.ctrl[self.arm_actuator_ids] = clipped

        for _ in range(self.config.simulation.substeps):
            mujoco.mj_step(self.model, self.data)
        self._cycle += 1

        observation = self.observation()
        q = observation["joint_positions"]
        dq = observation["joint_velocities"]
        joint_margin = observation["joint_limit_margin"]
        joint_limit_mask = (
            joint_margin <= self.config.safety.joint_limit_margin
        )
        velocity_limit_mask = np.abs(dq) >= np.asarray(
            self.config.safety.joint_velocity_limits, dtype=float
        )
        tracking_error_mask = (
            np.zeros(7, dtype=bool)
            if self._tracking_target is None
            else np.abs(self._tracking_target - q)
            > np.asarray(self.config.safety.maximum_tracking_error, dtype=float)
        )
        finite = self._finite_state()
        acceleration_mask = (
            np.abs(self.data.qacc[self.arm_dof_addresses])
            >= self.config.safety.simulation_instability_acceleration
        )

        self._saturation_streak = (
            self._saturation_streak + 1 if np.any(saturation_mask) else 0
        )
        self._rate_limit_streak = (
            self._rate_limit_streak + 1 if np.any(rate_mask) else 0
        )
        sustained_steps = max(
            1,
            math.ceil(
                self.config.safety.sustained_violation_duration
                / self.control_period
            ),
        )
        elapsed = float(self.data.time - self._episode_start_time)
        if not finite:
            self._termination_reason = "non_finite_state"
        elif np.any(joint_limit_mask):
            self._termination_reason = "joint_position_limit"
        elif np.any(velocity_limit_mask):
            self._termination_reason = "joint_velocity_limit"
        elif np.any(tracking_error_mask):
            self._termination_reason = "tracking_error_exceeded"
        elif self._saturation_streak >= sustained_steps:
            self._termination_reason = "torque_saturation_sustained"
        elif self._rate_limit_streak >= sustained_steps:
            self._termination_reason = "torque_rate_limit_sustained"
        elif np.any(acceleration_mask):
            self._termination_reason = "simulation_instability"
        elif elapsed + 1e-12 >= self.config.simulation.maximum_duration:
            self._termination_reason = "timeout"

        diagnostics = {
            "commanded_torque": commanded.copy(),
            "rate_limited_torque": rate_limited.copy(),
            "clipped_torque": clipped.copy(),
            "actuator_force": observation["actuator_force"].copy(),
            "saturation_mask": saturation_mask.copy(),
            "torque_rate_limit_mask": rate_mask.copy(),
            "joint_limit_mask": joint_limit_mask.copy(),
            "velocity_limit_mask": velocity_limit_mask.copy(),
            "tracking_error_mask": tracking_error_mask.copy(),
            "simulation_instability_mask": acceleration_mask.copy(),
            "finite_value_status": bool(finite),
            "termination_reason": self._termination_reason,
        }
        return observation, diagnostics

    def close(self) -> None:
        self._closed = True
