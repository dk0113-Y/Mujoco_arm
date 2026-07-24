"""World-frame kinematics for the project-defined Panda CI TCP site."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


JOINT_COUNT = 7
TASK_DIMENSION = 6
QUATERNION_CONVENTION = "wxyz"
TWIST_ORDER = ("linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z")


def _finite_array(
    value: np.ndarray, shape: tuple[int, ...], name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def normalize_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = _finite_array(value, (4,), "quaternion").copy()
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(float).eps:
        raise ValueError("quaternion norm must be positive")
    quaternion /= norm
    # Canonical hemisphere makes logging deterministic.  Rotation conversion
    # and all control errors remain invariant to an input sign flip.
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    elif abs(quaternion[0]) <= 1e-15:
        for component in quaternion[1:]:
            if abs(component) > 1e-15:
                if component < 0.0:
                    quaternion *= -1.0
                break
    return quaternion


def quaternion_wxyz_to_rotation(value: np.ndarray) -> np.ndarray:
    quaternion = normalize_quaternion_wxyz(value)
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def rotation_to_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    rotation = validate_rotation_matrix(value, "rotation")
    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
    return normalize_quaternion_wxyz(quaternion)


def validate_rotation_matrix(value: np.ndarray, name: str) -> np.ndarray:
    rotation = _finite_array(value, (3, 3), name)
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0
    ):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return rotation


def rotation_vector_to_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    vector = _finite_array(rotation_vector, (3,), "rotation_vector")
    angle = float(np.linalg.norm(vector))
    if angle <= 1e-15:
        # First-order branch preserves the exact identity at zero and avoids
        # dividing by a tiny angle.
        return np.eye(3, dtype=float)
    axis = vector / angle
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * (cross @ cross)
    )


def rotation_matrix_log_world(value: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector for a world-frame rotation."""

    rotation = validate_rotation_matrix(value, "relative_rotation")
    quaternion = rotation_to_quaternion_wxyz(rotation)
    vector = quaternion[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-15:
        return np.zeros(3, dtype=float)
    angle = 2.0 * float(np.arctan2(vector_norm, quaternion[0]))
    result = angle * vector / vector_norm
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("rotation logarithm produced NaN or Inf")
    return result


def orientation_error_world(
    current_rotation: np.ndarray, target_rotation: np.ndarray
) -> np.ndarray:
    """World-frame error taking current orientation toward the target.

    The convention is ``Log(R_target R_current.T)``.  A positive small target
    rotation about a world axis therefore produces a positive component on
    that axis, consistent with the world angular Jacobian returned by MuJoCo.
    """

    current = validate_rotation_matrix(current_rotation, "current_rotation")
    target = validate_rotation_matrix(target_rotation, "target_rotation")
    return rotation_matrix_log_world(target @ current.T)


@dataclass(frozen=True)
class PandaTcpKinematics:
    position: np.ndarray
    rotation: np.ndarray
    quaternion_wxyz: np.ndarray
    jacobian: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    twist: np.ndarray
    site_twist: np.ndarray
    singular_values: np.ndarray
    rank: int
    minimum_singular_value: float
    condition_number: float
    twist_consistency_error: float


class PandaTcpKinematicsProvider:
    """Compute CI TCP pose, world Jacobian, twist, and controllability."""

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_dof_addresses: np.ndarray,
        *,
        site_name: str,
        rank_tolerance: float,
    ) -> None:
        self.model = model
        self.arm_dof_addresses = np.asarray(
            arm_dof_addresses, dtype=int
        ).copy()
        if self.arm_dof_addresses.shape != (JOINT_COUNT,):
            raise ValueError("arm_dof_addresses must have shape (7,)")
        if len(set(int(value) for value in self.arm_dof_addresses)) != JOINT_COUNT:
            raise ValueError("arm_dof_addresses must be unique")
        if not np.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive")
        self.rank_tolerance = float(rank_tolerance)
        self.site_name = str(site_name)
        self.site_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.site_name)
        )
        if self.site_id < 0:
            raise RuntimeError(
                f"Torque model is missing required CI TCP site: {self.site_name}"
            )

    def compute(self, data: mujoco.MjData) -> PandaTcpKinematics:
        # mj_step ends after integration; derived position/velocity fields can
        # still describe the pre-integration stage.  Refresh them so the site
        # pose, Jacobian, cvel, and the current q/dq all refer to one state.
        mujoco.mj_forward(self.model, data)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)
        dofs = self.arm_dof_addresses
        jacobian = np.vstack((jacp[:, dofs], jacr[:, dofs]))
        dq = np.asarray(data.qvel[dofs], dtype=float)
        twist = jacobian @ dq
        object_velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_SITE,
            self.site_id,
            object_velocity,
            0,
        )
        site_twist = np.concatenate(
            (object_velocity[3:6], object_velocity[0:3])
        )
        twist_error = float(np.max(np.abs(twist - site_twist)))
        if twist_error > 1e-9:
            raise RuntimeError(
                "CI TCP J@dq does not match MuJoCo world site velocity: "
                f"{twist_error}"
            )

        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.count_nonzero(singular_values > self.rank_tolerance))
        minimum = float(singular_values[-1])
        denominator = max(minimum, np.finfo(float).tiny)
        condition = float(singular_values[0] / denominator)
        position = np.asarray(data.site_xpos[self.site_id], dtype=float).copy()
        rotation = np.asarray(
            data.site_xmat[self.site_id], dtype=float
        ).reshape(3, 3).copy()
        quaternion = rotation_to_quaternion_wxyz(rotation)
        arrays = (
            position,
            rotation,
            quaternion,
            jacobian,
            twist,
            site_twist,
            singular_values,
        )
        scalars = (minimum, condition, twist_error)
        if not all(np.all(np.isfinite(value)) for value in arrays) or not all(
            np.isfinite(value) for value in scalars
        ):
            raise FloatingPointError("CI TCP kinematics contains NaN or Inf")
        return PandaTcpKinematics(
            position=position,
            rotation=rotation,
            quaternion_wxyz=quaternion,
            jacobian=jacobian,
            linear_velocity=twist[:3].copy(),
            angular_velocity=twist[3:].copy(),
            twist=twist.copy(),
            site_twist=site_twist.copy(),
            singular_values=singular_values.copy(),
            rank=rank,
            minimum_singular_value=minimum,
            condition_number=condition,
            twist_consistency_error=twist_error,
        )


def controllability_termination_reason(
    kinematics: PandaTcpKinematics,
    *,
    minimum_rank: int,
    minimum_singular_value: float,
    maximum_condition_number: float,
) -> str | None:
    if kinematics.rank < minimum_rank:
        return "jacobian_rank_deficient"
    if (
        kinematics.minimum_singular_value < minimum_singular_value
        or kinematics.condition_number > maximum_condition_number
    ):
        return "jacobian_condition_exceeded"
    return None
