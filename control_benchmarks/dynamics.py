"""Verified MuJoCo dynamics terms for physical direct-motor compensation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class DynamicsTerms:
    gravity: np.ndarray
    coriolis_centrifugal: np.ndarray
    passive: np.ndarray
    constraint: np.ndarray
    compensation: np.ndarray
    mode: str


class MuJoCoDynamicsProvider:
    """Split gravity and velocity bias without mutating the stepping plant.

    MuJoCo 3.10 defines ``qfrc_bias`` as gravity plus Coriolis/centrifugal
    generalized force.  It excludes passive and constraint forces.  This
    provider re-evaluates the current state in scratch ``MjData`` instances,
    obtains gravity at identical q with zero velocity, and uses the verified
    difference for Coriolis/centrifugal terms.
    """

    MODES = frozenset({"none", "gravity", "gravity_coriolis"})

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_qpos_addresses: np.ndarray,
        arm_dof_addresses: np.ndarray,
        *,
        mode: str = "gravity_coriolis",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported dynamics compensation mode: {mode!r}")
        if np.any(np.asarray(model.body_gravcomp, dtype=float) != 0.0):
            raise ValueError(
                "Dynamics provider requires body_gravcomp=0 to avoid duplicate "
                "gravity compensation"
            )
        self.model = model
        self.arm_qpos_addresses = np.asarray(arm_qpos_addresses, dtype=int).copy()
        self.arm_dof_addresses = np.asarray(arm_dof_addresses, dtype=int).copy()
        if self.arm_qpos_addresses.shape != (7,) or self.arm_dof_addresses.shape != (
            7,
        ):
            raise ValueError("arm address arrays must both have shape (7,)")
        self.mode = mode
        self._moving_data = mujoco.MjData(model)
        self._zero_velocity_data = mujoco.MjData(model)

    def compute(self, data: mujoco.MjData) -> DynamicsTerms:
        for scratch in (self._moving_data, self._zero_velocity_data):
            scratch.qpos[:] = data.qpos
            scratch.time = data.time
            scratch.ctrl[:] = 0.0
            scratch.qfrc_applied[:] = 0.0
            scratch.xfrc_applied[:] = 0.0
        self._moving_data.qvel[:] = data.qvel
        self._zero_velocity_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._moving_data)
        mujoco.mj_forward(self.model, self._zero_velocity_data)

        dofs = self.arm_dof_addresses
        gravity = np.asarray(
            self._zero_velocity_data.qfrc_bias[dofs], dtype=float
        ).copy()
        full_bias = np.asarray(
            self._moving_data.qfrc_bias[dofs], dtype=float
        ).copy()
        coriolis_centrifugal = full_bias - gravity
        passive = np.asarray(
            self._moving_data.qfrc_passive[dofs], dtype=float
        ).copy()
        constraint = np.asarray(data.qfrc_constraint[dofs], dtype=float).copy()
        if self.mode == "none":
            compensation = np.zeros(7, dtype=float)
        elif self.mode == "gravity":
            compensation = gravity.copy()
        else:
            compensation = full_bias.copy()
        arrays = (
            gravity,
            coriolis_centrifugal,
            passive,
            constraint,
            compensation,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise FloatingPointError("MuJoCo dynamics terms contain NaN or Inf")
        return DynamicsTerms(
            gravity=gravity,
            coriolis_centrifugal=coriolis_centrifugal,
            passive=passive,
            constraint=constraint,
            compensation=compensation,
            mode=self.mode,
        )
