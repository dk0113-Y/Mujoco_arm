from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Callable

import mujoco
import numpy as np

from environments.config import ControllerConfig
from environments.panda_u_table_env import InvalidResetError, PandaUTableEnv
from evaluation.episode_result import EpisodeResult, FailureReason


class IKNotConvergedError(RuntimeError):
    pass


class ControllerFailure(RuntimeError):
    def __init__(
        self,
        reason: FailureReason,
        message: str,
        *,
        errors: dict[str, float] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.errors = errors or {}


@dataclass(frozen=True)
class Action:
    kind: str
    stage: str
    duration: float
    target_position: np.ndarray | None = None
    gripper_control: float | None = None


@dataclass
class MotionPlan:
    start_time: float
    duration: float
    start_control: np.ndarray
    target_control: np.ndarray
    target_position: np.ndarray


def limit_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or norm <= max_norm:
        return vector
    return vector * (max_norm / norm)


def rotation_error_world(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    error = np.zeros(3, dtype=float)
    for axis_index in range(3):
        error += np.cross(
            current_rotation[:, axis_index], target_rotation[:, axis_index]
        )
    return 0.5 * error


def solve_pose_ik(
    *,
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    tcp_site_id: int,
    arm_qpos_addresses: np.ndarray,
    arm_dof_addresses: np.ndarray,
    arm_joint_ranges: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    config: ControllerConfig,
) -> tuple[np.ndarray, float, float, int]:
    """Solve fixed-damping DLS IK with the original baseline equation."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = initial_qpos
    ik_data.qvel[:] = 0.0
    mujoco.mj_forward(model, ik_data)

    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)
    position_error_norm = float("inf")
    orientation_error_norm = float("inf")

    for iteration in range(1, config.ik_max_iterations + 1):
        mujoco.mj_forward(model, ik_data)
        current_position = ik_data.site_xpos[tcp_site_id].copy()
        current_rotation = ik_data.site_xmat[tcp_site_id].reshape(3, 3).copy()
        position_error = target_position - current_position
        orientation_error = rotation_error_world(current_rotation, target_rotation)
        position_error_norm = float(np.linalg.norm(position_error))
        orientation_error_norm = float(np.linalg.norm(orientation_error))
        if (
            position_error_norm <= config.ik_position_tolerance
            and orientation_error_norm <= config.orientation_tolerance
        ):
            return (
                ik_data.qpos[arm_qpos_addresses].copy(),
                position_error_norm,
                orientation_error_norm,
                iteration,
            )

        mujoco.mj_jacSite(
            model,
            ik_data,
            jacobian_position,
            jacobian_rotation,
            tcp_site_id,
        )
        position_jacobian = jacobian_position[:, arm_dof_addresses]
        rotation_jacobian = jacobian_rotation[:, arm_dof_addresses]
        task_jacobian = np.vstack(
            (position_jacobian, config.orientation_weight * rotation_jacobian)
        )
        task_error = np.concatenate(
            (position_error, config.orientation_weight * orientation_error)
        )

        # Baseline fixed DLS: dq = J.T @ solve(J @ J.T + lambda^2 I, e)
        regularized_matrix = (
            task_jacobian @ task_jacobian.T
            + (config.ik_damping**2) * np.eye(6)
        )
        joint_step = task_jacobian.T @ np.linalg.solve(
            regularized_matrix, task_error
        )
        joint_step = config.ik_step_gain * joint_step
        joint_step = limit_vector_norm(joint_step, config.ik_max_joint_step)
        next_joint_position = ik_data.qpos[arm_qpos_addresses] + joint_step
        ik_data.qpos[arm_qpos_addresses] = np.clip(
            next_joint_position,
            arm_joint_ranges[:, 0],
            arm_joint_ranges[:, 1],
        )

    raise IKNotConvergedError(
        "Fixed-DLS IK did not converge: "
        f"position_error={position_error_norm:.6f} m, "
        f"orientation_error={orientation_error_norm:.6f} rad"
    )


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


class FixedDLSPickPlaceController:
    """Original fixed-DLS math plus the scripted pick/place state machine."""

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config

    def _make_motion_plan(
        self,
        env: PandaUTableEnv,
        action: Action,
        target_rotation: np.ndarray,
        key_errors: dict[str, float],
    ) -> MotionPlan:
        if action.target_position is None:
            raise ValueError(f"Motion action {action.stage} has no target position")
        try:
            target, position_error, orientation_error, _ = solve_pose_ik(
                model=env.model,
                initial_qpos=env.data.qpos.copy(),
                tcp_site_id=env.tcp_site_id,
                arm_qpos_addresses=env.arm_qpos_addresses,
                arm_dof_addresses=env.arm_dof_addresses,
                arm_joint_ranges=env.arm_joint_ranges,
                target_position=action.target_position,
                target_rotation=target_rotation,
                config=self.config,
            )
        except IKNotConvergedError as exc:
            raise ControllerFailure(
                FailureReason.IK_NOT_CONVERGED, str(exc)
            ) from exc
        key_errors["ik_position_error"] = position_error
        key_errors["ik_orientation_error"] = orientation_error
        target = np.clip(
            target,
            env.arm_ctrl_ranges[:, 0],
            env.arm_ctrl_ranges[:, 1],
        )
        return MotionPlan(
            start_time=float(env.data.time),
            duration=action.duration,
            start_control=env.data.ctrl[env.arm_actuator_ids].copy(),
            target_control=target,
            target_position=action.target_position.copy(),
        )

    @staticmethod
    def _apply_motion(env: PandaUTableEnv, plan: MotionPlan) -> None:
        elapsed = float(env.data.time - plan.start_time)
        interpolation = smoothstep(elapsed / plan.duration)
        env.data.ctrl[env.arm_actuator_ids] = plan.start_control + interpolation * (
            plan.target_control - plan.start_control
        )

    def _actions(
        self,
        env: PandaUTableEnv,
        object_position: np.ndarray,
        target_position: np.ndarray,
    ) -> list[Action]:
        above_pick = object_position + np.array(
            [0.0, 0.0, self.config.waypoint_height]
        )
        grasp = object_position + np.array([0.0, 0.0, self.config.grasp_z_offset])
        lift = object_position + np.array([0.0, 0.0, self.config.lift_height])
        above_target = np.array(
            [target_position[0], target_position[1], lift[2]], dtype=float
        )
        place = np.array(
            [
                target_position[0],
                target_position[1],
                target_position[2] + env.config.workspace.object_half_size,
            ],
            dtype=float,
        )
        return [
            Action("motion", "move_above_pick", self.config.approach_duration, above_pick),
            Action("motion", "descend_to_pick", self.config.descent_duration, grasp),
            Action(
                "gripper",
                "close_gripper",
                self.config.gripper_duration,
                gripper_control=self.config.gripper_close_control,
            ),
            Action("motion", "lift_object", self.config.lift_duration, lift),
            Action(
                "motion",
                "move_above_target",
                self.config.transfer_duration,
                above_target,
            ),
            Action("motion", "descend_to_place", self.config.descent_duration, place),
            Action(
                "gripper",
                "open_gripper",
                self.config.gripper_duration,
                gripper_control=self.config.gripper_open_control,
            ),
            Action(
                "motion",
                "withdraw",
                self.config.withdraw_duration,
                above_target,
            ),
        ]

    def _result(
        self,
        env: PandaUTableEnv,
        *,
        success: bool,
        failure_reason: FailureReason | None,
        stage: str,
        lift_height: float | None,
        exception_message: str | None,
        key_errors: dict[str, float],
    ) -> EpisodeResult:
        episode = env.current_episode
        final_tcp: tuple[float, float, float] | None = None
        final_object: tuple[float, float, float] | None = None
        target: tuple[float, float, float] | None = None
        xy_error: float | None = None
        height_error: float | None = None
        if episode is not None:
            final_tcp = tuple(float(value) for value in env.data.site_xpos[env.tcp_site_id])
            final_object = tuple(
                float(value) for value in env.data.xpos[env.object_body_id]
            )
            target = tuple(
                float(value) for value in env.data.site_xpos[env.place_target_site_id]
            )
            xy_error, height_error = env.placement_errors()
        return EpisodeResult(
            seed=env.current_seed,
            pick_mode=env.config.pick.mode,
            place_mode=env.config.place.mode,
            physics_mode=env.config.physics.mode,
            pick_region=None if episode is None else episode.pick_region,
            place_region=None if episode is None else episode.place_region,
            sampled_pick_position=None if episode is None else episode.pick_position,
            sampled_place_position=None if episode is None else episode.place_position,
            sampled_mass=None if episode is None else episode.mass,
            sampled_friction=None if episode is None else episode.friction,
            success=success,
            failure_reason=None if failure_reason is None else failure_reason.value,
            final_stage=stage,
            simulation_time=float(env.data.time),
            lift_height=lift_height,
            final_xy_error=xy_error,
            final_height_error=height_error,
            collision_count=env.collision_count,
            exception_message=exception_message,
            final_tcp_position=final_tcp,
            final_object_position=final_object,
            target_position=target,
            key_errors=dict(key_errors) or None,
        )

    def run_episode(
        self,
        env: PandaUTableEnv,
        *,
        seed: int | None = None,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None = None,
    ) -> EpisodeResult:
        stage = "reset"
        lift_height: float | None = None
        key_errors: dict[str, float] = {}
        try:
            env.reset(seed=seed)
        except InvalidResetError as exc:
            return self._result(
                env,
                success=False,
                failure_reason=FailureReason.INVALID_RESET,
                stage=stage,
                lift_height=lift_height,
                exception_message=str(exc),
                key_errors=key_errors,
            )
        except Exception:
            return self._result(
                env,
                success=False,
                failure_reason=FailureReason.UNEXPECTED_EXCEPTION,
                stage=stage,
                lift_height=lift_height,
                exception_message=traceback.format_exc(),
                key_errors=key_errors,
            )

        initial_object = env.data.xpos[env.object_body_id].copy()
        target_position = env.data.site_xpos[env.place_target_site_id].copy()
        target_rotation = env.data.site_xmat[env.tcp_site_id].reshape(3, 3).copy()
        actions = self._actions(env, initial_object, target_position)
        action_index = 0
        action_start_time = float(env.data.time)
        motion_plan: MotionPlan | None = None
        lift_was_confirmed = False

        try:
            while action_index < len(actions):
                action = actions[action_index]
                stage = action.stage
                if action.kind == "motion":
                    if motion_plan is None:
                        motion_plan = self._make_motion_plan(
                            env, action, target_rotation, key_errors
                        )
                    self._apply_motion(env, motion_plan)
                elif action.kind == "gripper":
                    if action.gripper_control is None:
                        raise ValueError(f"Gripper action {action.stage} has no control")
                    env.data.ctrl[env.gripper_actuator_id] = action.gripper_control
                else:
                    raise ValueError(f"Unknown action kind: {action.kind}")

                env.step(env.data.ctrl.copy())
                if step_callback is not None and step_callback(env) is False:
                    raise ControllerFailure(
                        FailureReason.TIMEOUT, "Viewer was closed before episode completion"
                    )

                current_lift = float(
                    env.data.xpos[env.object_body_id, 2] - initial_object[2]
                )
                lift_height = (
                    current_lift if lift_height is None else max(lift_height, current_lift)
                )
                reason = env.failure_reason(
                    stage=stage,
                    initial_object_height=float(initial_object[2]),
                    lift_was_confirmed=lift_was_confirmed,
                )
                if reason is not None:
                    raise ControllerFailure(
                        FailureReason(reason), f"Environment failure during {stage}: {reason}"
                    )

                if action.kind == "motion" and motion_plan is not None:
                    elapsed = float(env.data.time - motion_plan.start_time)
                    if elapsed >= action.duration + self.config.motion_hold_time:
                        waypoint_error = float(
                            np.linalg.norm(
                                motion_plan.target_position
                                - env.data.site_xpos[env.tcp_site_id]
                            )
                        )
                        key_errors["waypoint_error"] = waypoint_error
                        if waypoint_error > self.config.waypoint_tolerance:
                            raise ControllerFailure(
                                FailureReason.WAYPOINT_ERROR,
                                f"Waypoint {stage} error {waypoint_error:.6f} m exceeds "
                                f"{self.config.waypoint_tolerance:.6f} m",
                                errors={"waypoint_error": waypoint_error},
                            )
                        if stage == "lift_object":
                            lift_gain = float(
                                env.data.xpos[env.object_body_id, 2] - initial_object[2]
                            )
                            key_errors["lift_height"] = lift_gain
                            if lift_gain < self.config.minimum_lift_height:
                                raise ControllerFailure(
                                    FailureReason.LIFT_FAILURE,
                                    f"Object lift {lift_gain:.6f} m is below "
                                    f"{self.config.minimum_lift_height:.6f} m",
                                    errors={"lift_height": lift_gain},
                                )
                            lift_was_confirmed = True
                        motion_plan = None
                        action_index += 1
                        action_start_time = float(env.data.time)
                elif action.kind == "gripper":
                    if env.data.time - action_start_time >= action.duration:
                        action_index += 1
                        action_start_time = float(env.data.time)

            stage = "final_validation"
            xy_error, height_error = env.placement_errors()
            key_errors["final_xy_error"] = xy_error
            key_errors["final_height_error"] = height_error
            if xy_error > self.config.place_xy_tolerance:
                raise ControllerFailure(
                    FailureReason.PLACE_XY_ERROR,
                    f"Final XY error {xy_error:.6f} m exceeds "
                    f"{self.config.place_xy_tolerance:.6f} m",
                    errors={"final_xy_error": xy_error},
                )
            if height_error > self.config.place_height_tolerance:
                raise ControllerFailure(
                    FailureReason.PLACE_HEIGHT_ERROR,
                    f"Final height error {height_error:.6f} m exceeds "
                    f"{self.config.place_height_tolerance:.6f} m",
                    errors={"final_height_error": height_error},
                )
            return self._result(
                env,
                success=True,
                failure_reason=None,
                stage="completed",
                lift_height=lift_height,
                exception_message=None,
                key_errors=key_errors,
            )
        except ControllerFailure as exc:
            key_errors.update(exc.errors)
            return self._result(
                env,
                success=False,
                failure_reason=exc.reason,
                stage=stage,
                lift_height=lift_height,
                exception_message=str(exc),
                key_errors=key_errors,
            )
        except Exception:
            return self._result(
                env,
                success=False,
                failure_reason=FailureReason.UNEXPECTED_EXCEPTION,
                stage=stage,
                lift_height=lift_height,
                exception_message=traceback.format_exc(),
                key_errors=key_errors,
            )
