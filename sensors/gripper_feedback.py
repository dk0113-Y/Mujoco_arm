from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GripperFeedback:
    """Encoder-like feedback exposed by the simulated parallel gripper."""

    left_finger_position: float
    right_finger_position: float
    aperture: float
    aperture_velocity: float
    commanded_state: str
    timestamp: float


def _required_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing required object: {name}")
    return object_id


class GripperFeedbackSensor:
    """Read Panda finger encoders as a verified physical jaw aperture.

    The pinned Panda model has two equal-range slide joints whose positive world
    axes oppose one another.  Its joint equality constrains both positions to be
    equal, so moving either joint positively increases the jaw separation and
    the movable aperture is ``q_left + q_right``.  Construction validates that
    contract instead of silently applying it to an incompatible model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        left_joint_name: str = "finger_joint1",
        right_joint_name: str = "finger_joint2",
    ) -> None:
        self.model = model
        self.data = data
        self.left_joint_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, left_joint_name
        )
        self.right_joint_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, right_joint_name
        )
        if self.left_joint_id == self.right_joint_id:
            raise RuntimeError("Left and right finger joints must be distinct")

        self._validate_joint_contract()
        self.left_qpos_address = int(model.jnt_qposadr[self.left_joint_id])
        self.right_qpos_address = int(model.jnt_qposadr[self.right_joint_id])
        self.left_dof_address = int(model.jnt_dofadr[self.left_joint_id])
        self.right_dof_address = int(model.jnt_dofadr[self.right_joint_id])

        joint_ranges = np.asarray(
            model.jnt_range[[self.left_joint_id, self.right_joint_id]], dtype=float
        )
        self.minimum_aperture = float(np.sum(joint_ranges[:, 0]))
        self.maximum_aperture = float(np.sum(joint_ranges[:, 1]))

    def _validate_joint_contract(self) -> None:
        joint_ids = np.asarray(
            [self.left_joint_id, self.right_joint_id], dtype=int
        )
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
        if any(
            int(self.model.jnt_type[joint_id]) != slide_type
            for joint_id in joint_ids
        ):
            raise RuntimeError("Panda finger joints must both be slide joints")
        if not np.all(self.model.jnt_limited[joint_ids]):
            raise RuntimeError("Panda finger joints must have finite limits")

        joint_ranges = np.asarray(self.model.jnt_range[joint_ids], dtype=float)
        if not np.all(np.isfinite(joint_ranges)):
            raise RuntimeError("Panda finger joint ranges must be finite")
        if not np.all(joint_ranges[:, 1] > joint_ranges[:, 0]):
            raise RuntimeError("Panda finger joint ranges must be non-empty")
        if not np.allclose(joint_ranges[0], joint_ranges[1], atol=1e-12):
            raise RuntimeError("Panda finger joint ranges must match")
        if joint_ranges[0, 0] < -1e-12:
            raise RuntimeError("Panda finger joint lower limit must be non-negative")

        expected_polynomial = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        has_identity_equality = False
        for equality_id in range(self.model.neq):
            if int(self.model.eq_type[equality_id]) != int(
                mujoco.mjtEq.mjEQ_JOINT
            ):
                continue
            object_ids = {
                int(self.model.eq_obj1id[equality_id]),
                int(self.model.eq_obj2id[equality_id]),
            }
            if object_ids != {self.left_joint_id, self.right_joint_id}:
                continue
            polynomial = np.asarray(self.model.eq_data[equality_id, :5], dtype=float)
            if np.allclose(polynomial, expected_polynomial, atol=1e-12):
                has_identity_equality = True
                break
        if not has_identity_equality:
            raise RuntimeError(
                "Panda finger joints must have an identity joint equality"
            )

        validation_data = mujoco.MjData(self.model)
        qpos_addresses = self.model.jnt_qposadr[joint_ids].astype(int)
        body_ids = self.model.jnt_bodyid[joint_ids].astype(int)

        validation_data.qpos[qpos_addresses] = joint_ranges[:, 0]
        mujoco.mj_forward(self.model, validation_data)
        world_axes = np.asarray(validation_data.xaxis[joint_ids], dtype=float)
        axis_norms = np.linalg.norm(world_axes, axis=1)
        if not np.allclose(axis_norms, 1.0, atol=1e-9):
            raise RuntimeError("Panda finger joint world axes are invalid")
        if float(np.dot(world_axes[0], world_axes[1])) > -1.0 + 1e-9:
            raise RuntimeError(
                "Panda finger joints must move in opposite physical directions"
            )
        closed_separation = float(
            np.linalg.norm(
                validation_data.xpos[body_ids[0]]
                - validation_data.xpos[body_ids[1]]
            )
        )

        validation_data.qpos[qpos_addresses] = joint_ranges[:, 1]
        mujoco.mj_forward(self.model, validation_data)
        open_separation = float(
            np.linalg.norm(
                validation_data.xpos[body_ids[0]]
                - validation_data.xpos[body_ids[1]]
            )
        )
        expected_increase = float(
            np.sum(joint_ranges[:, 1] - joint_ranges[:, 0])
        )
        if not np.isclose(
            open_separation - closed_separation,
            expected_increase,
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                "Finger body separation does not match the summed joint travel"
            )

    def read(self, commanded_state: str = "unknown") -> GripperFeedback:
        if not isinstance(commanded_state, str) or not commanded_state:
            raise ValueError("commanded_state must be a non-empty string")

        left_position = float(self.data.qpos[self.left_qpos_address])
        right_position = float(self.data.qpos[self.right_qpos_address])
        aperture_velocity = float(
            self.data.qvel[self.left_dof_address]
            + self.data.qvel[self.right_dof_address]
        )
        timestamp = float(self.data.time)
        values = (
            left_position,
            right_position,
            aperture_velocity,
            timestamp,
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Gripper feedback contains NaN or Inf")

        aperture = float(
            np.clip(
                left_position + right_position,
                self.minimum_aperture,
                self.maximum_aperture,
            )
        )
        return GripperFeedback(
            left_finger_position=left_position,
            right_finger_position=right_position,
            aperture=aperture,
            aperture_velocity=aperture_velocity,
            commanded_state=commanded_state,
            timestamp=timestamp,
        )
