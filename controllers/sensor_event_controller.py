from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import traceback
from typing import Any, Callable

import mujoco
import numpy as np

from environments.config import B1Config, ControllerConfig
from environments.panda_u_table_env import InvalidResetError, PandaUTableEnv
from evaluation.episode_result import EpisodeResult, FailureReason
from evaluation.perception_evaluator import EpisodeOutcome, build_episode_result
from perception.state_provider import (
    TaskStateProvider,
    task_state_from_perception_frame,
)
from perception.types import TaskStateEstimate
from sensors import ContactFeedback, ContactSensor, GripperFeedback, GripperFeedbackSensor

from .fixed_dls_controller import (
    IKNotConvergedError,
    rotation_error_world,
    smoothstep,
    solve_pose_ik,
)
from .grasp_state_machine import GraspState, GraspStateMachine, GraspUpdate


class B1Stage(str, Enum):
    SCENE_PERCEPTION = "scene_perception"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    PREGRASP_REACQUISITION = "pregrasp_reacquisition"
    DESCEND_TO_GRASP = "descend_to_grasp"
    CLOSE_GRIPPER = "close_gripper"
    GRASP_CANDIDATE_CHECK = "grasp_candidate_check"
    TRIAL_LIFT = "trial_lift"
    GRASP_CONFIRMATION = "grasp_confirmation"
    TRANSFER = "transfer"
    DESCEND_TO_PLACE = "descend_to_place"
    RELEASE = "release"
    WITHDRAW = "withdraw"
    FINAL_VISUAL_VERIFICATION = "final_visual_verification"
    COMPLETED = "completed"


class B1ControllerFailure(RuntimeError):
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


@dataclass
class EventMotionPlan:
    start_time: float
    reference_duration: float
    start_control: np.ndarray
    target_control: np.ndarray
    target_position: np.ndarray


@dataclass
class B1Runtime:
    stage: B1Stage = B1Stage.SCENE_PERCEPTION
    stage_start_time: float = 0.0
    target_rotation: np.ndarray | None = None
    initial_object_position: np.ndarray | None = None
    locked_target_position: np.ndarray | None = None
    corrected_object_position: np.ndarray | None = None
    grasp_monitor: GraspStateMachine | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    key_errors: dict[str, float] = field(default_factory=dict)
    stage_durations: dict[str, float] = field(default_factory=dict)
    diagnostic_observer: Callable[["B1DiagnosticSnapshot"], None] | None = None


@dataclass(frozen=True)
class B1DiagnosticSnapshot:
    """Immutable, controller-observable telemetry for an optional passive recorder.

    Object truth and diagnostic camera data are intentionally absent.  A runner-side
    recorder may align those privileged values by ``simulation_time`` without ever
    returning them to the controller.
    """

    event: str
    simulation_time: float
    stage: str
    next_stage: str | None
    failure_reason: str | None
    grasp_state: str | None
    gripper_aperture: float | None
    gripper_aperture_velocity: float | None
    left_finger_position: float | None
    right_finger_position: float | None
    commanded_state: str | None
    left_contact: bool | None
    right_contact: bool | None
    bilateral_contact: bool | None
    bilateral_contact_duration: float | None
    candidate_aperture: float | None
    aperture_drop: float | None
    commanded_closing_predicate: bool | None
    minimum_aperture_predicate: bool | None
    contact_predicate: bool | None
    lift_predicate: bool | None
    aperture_retention_predicate: bool | None
    collision_free_predicate: bool | None
    combined_predicate: bool | None
    candidate_hold_steps: int
    confirmation_hold_steps: int
    contact_loss_hold_steps: int
    contact_loss_event_count: int
    trial_lift_completed: bool
    robot_table_collision: bool
    tcp_position: tuple[float, float, float]
    finger_positions: tuple[float, float]


def _finite_position(position: tuple[float, float, float] | None) -> bool:
    return bool(
        position is not None
        and np.asarray(position, dtype=float).shape == (3,)
        and np.all(np.isfinite(position))
    )


def _component_confidence(estimate: TaskStateEstimate, component: str) -> float:
    confidence = getattr(estimate, f"{component}_confidence")
    return float(estimate.confidence if confidence is None else confidence)


def _component_valid(
    estimate: TaskStateEstimate,
    component: str,
    minimum_confidence: float,
) -> bool:
    position = getattr(estimate, f"{component}_position")
    explicit_valid = getattr(estimate, f"{component}_valid")
    return bool(
        (explicit_valid is not False)
        and _finite_position(position)
        and _component_confidence(estimate, component) >= minimum_confidence
    )


def _robust_positions(
    estimates: list[TaskStateEstimate],
    component: str,
) -> tuple[np.ndarray, float, float]:
    positions = np.asarray(
        [getattr(estimate, f"{component}_position") for estimate in estimates],
        dtype=float,
    )
    estimate = np.median(positions, axis=0)
    distances = np.linalg.norm(positions - estimate, axis=1)
    spread = float(np.max(distances)) if distances.size else float("inf")
    confidence = float(
        np.median(
            [_component_confidence(sample, component) for sample in estimates]
        )
    )
    return estimate, spread, confidence


class SensorEventPickPlaceController:
    """B1 controller driven by external state, encoders, and touch proxies."""

    controller_type = "sensor_event_b1"
    external_state_sources = frozenset({"perception", "oracle"})

    def __init__(self, controller_config: ControllerConfig, b1_config: B1Config) -> None:
        self.controller_config = controller_config
        self.config = b1_config
        self._gripper_sensor: GripperFeedbackSensor | None = None
        self._contact_sensor: ContactSensor | None = None

    def _make_runtime(
        self,
        env: PandaUTableEnv,
        diagnostic_observer: Callable[[B1DiagnosticSnapshot], None] | None = None,
    ) -> B1Runtime:
        observation = env.observation()
        return B1Runtime(
            stage_start_time=float(observation["simulation_time"]),
            target_rotation=np.asarray(observation["tcp_orientation"], dtype=float).copy(),
            grasp_monitor=GraspStateMachine(
                empty_aperture_threshold=self.config.empty_gripper_aperture_threshold,
                minimum_grasp_aperture=self.config.minimum_grasp_aperture,
                candidate_hold_steps=self.config.bilateral_contact_hold_steps,
                confirmation_hold_steps=self.config.grasp_confirmation_hold_steps,
                contact_loss_hold_steps=self.config.contact_loss_hold_steps,
                aperture_drop_threshold=self.config.aperture_drop_threshold,
            ),
            metrics={
                "controller_type": self.controller_type,
                "external_state_provider_source": None,
                "initial_perception_frame_count": 0,
                "initial_valid_frame_count": 0,
                "initial_object_position": None,
                "locked_target_position": None,
                "initial_object_confidence": None,
                "initial_target_confidence": None,
                "initial_object_position_spread": None,
                "initial_target_position_spread": None,
                "initial_position_spread": None,
                "initial_perception_latency_ms": None,
                "initial_perception_timestamp": None,
                "camera_name": None,
                "image_resolution": None,
                "pregrasp_perception_frame_count": 0,
                "pregrasp_valid_frame_count": 0,
                "pregrasp_corrected_object_position": None,
                "pregrasp_correction_magnitude": None,
                "pregrasp_position_spread": None,
                "pregrasp_perception_latency_ms": None,
                "pregrasp_used_initial_fallback": False,
                "pregrasp_correction_exceeded_threshold": False,
                "gripper_aperture_before_close": None,
                "gripper_aperture_after_close": None,
                "bilateral_contact_duration": 0.0,
                "grasp_candidate": False,
                "trial_lift_completed": False,
                "grasp_confirmed": False,
                "contact_loss_event_count": 0,
                "grasp_lost": False,
                "final_visual_frame_count": 0,
                "final_visual_valid_frame_count": 0,
                "final_visual_object_position": None,
                "final_visual_xy_error": None,
                "final_visual_height_error": None,
                "final_visual_latency_ms": None,
                "controller_reported_success": False,
                "stage_durations": {},
            },
            diagnostic_observer=diagnostic_observer,
        )

    @staticmethod
    def _emit_diagnostic(
        env: PandaUTableEnv,
        runtime: B1Runtime,
        *,
        event: str,
        gripper: GripperFeedback | None = None,
        contact: ContactFeedback | None = None,
        update: GraspUpdate | None = None,
        next_stage: B1Stage | None = None,
        failure_reason: FailureReason | None = None,
    ) -> None:
        observer = runtime.diagnostic_observer
        if observer is None:
            return
        try:
            observation = env.observation()
            monitor = runtime.grasp_monitor
            snapshot = B1DiagnosticSnapshot(
                event=event,
                simulation_time=float(observation["simulation_time"]),
                stage=runtime.stage.value,
                next_stage=None if next_stage is None else next_stage.value,
                failure_reason=(
                    None if failure_reason is None else failure_reason.value
                ),
                grasp_state=None if monitor is None else monitor.state.value,
                gripper_aperture=None if gripper is None else gripper.aperture,
                gripper_aperture_velocity=(
                    None if gripper is None else gripper.aperture_velocity
                ),
                left_finger_position=(
                    None if gripper is None else gripper.left_finger_position
                ),
                right_finger_position=(
                    None if gripper is None else gripper.right_finger_position
                ),
                commanded_state=(
                    None if gripper is None else gripper.commanded_state
                ),
                left_contact=(
                    None if contact is None else contact.left_finger_object_contact
                ),
                right_contact=(
                    None if contact is None else contact.right_finger_object_contact
                ),
                bilateral_contact=(
                    None if contact is None else contact.bilateral_contact
                ),
                bilateral_contact_duration=(
                    None if contact is None else contact.contact_duration
                ),
                candidate_aperture=(
                    None if monitor is None else monitor.candidate_aperture
                ),
                aperture_drop=None if update is None else update.aperture_drop,
                commanded_closing_predicate=(
                    None
                    if update is None
                    else update.commanded_closing_predicate
                ),
                minimum_aperture_predicate=(
                    None if update is None else update.minimum_aperture_predicate
                ),
                contact_predicate=(
                    None if update is None else update.contact_predicate
                ),
                lift_predicate=None if update is None else update.lift_predicate,
                aperture_retention_predicate=(
                    None if update is None else update.aperture_retention_predicate
                ),
                collision_free_predicate=(
                    None if update is None else update.collision_free_predicate
                ),
                combined_predicate=(
                    None if update is None else update.combined_predicate
                ),
                candidate_hold_steps=(
                    0 if monitor is None else monitor.candidate_steps
                ),
                confirmation_hold_steps=(
                    0 if monitor is None else monitor.confirmation_steps
                ),
                contact_loss_hold_steps=(
                    0 if monitor is None else monitor.contact_loss_steps
                ),
                contact_loss_event_count=(
                    0 if monitor is None else monitor.contact_loss_event_count
                ),
                trial_lift_completed=bool(
                    runtime.metrics.get("trial_lift_completed", False)
                ),
                robot_table_collision=env.robot_table_collision(),
                tcp_position=tuple(
                    float(value) for value in observation["tcp_position"]
                ),
                finger_positions=tuple(
                    float(value) for value in observation["finger_positions"]
                ),
            )
            observer(snapshot)
        except Exception:
            # A diagnostic observer is outside the control contract.  It cannot
            # change an action, a transition, or the episode result.  Recorder
            # implementations retain their own errors and report them after the
            # controller has returned.
            return

    def _transition(
        self, runtime: B1Runtime, env: PandaUTableEnv, next_stage: B1Stage
    ) -> None:
        now = float(env.data.time)
        runtime.stage_durations[runtime.stage.value] = now - runtime.stage_start_time
        self._emit_diagnostic(
            env,
            runtime,
            event="stage_transition",
            next_stage=next_stage,
        )
        runtime.stage = next_stage
        runtime.stage_start_time = now

    def _finish_current_stage(self, runtime: B1Runtime, env: PandaUTableEnv) -> None:
        now = float(env.data.time)
        runtime.stage_durations[runtime.stage.value] = now - runtime.stage_start_time
        runtime.metrics["stage_durations"] = dict(runtime.stage_durations)

    @staticmethod
    def _stage_timeout_reason(stage: B1Stage) -> FailureReason:
        if stage is B1Stage.SCENE_PERCEPTION:
            return FailureReason.INITIAL_PERCEPTION_FAILED
        if stage is B1Stage.PREGRASP_REACQUISITION:
            return FailureReason.PREGRASP_REACQUISITION_FAILED
        if stage in (
            B1Stage.MOVE_TO_PREGRASP,
            B1Stage.DESCEND_TO_GRASP,
            B1Stage.TRANSFER,
            B1Stage.DESCEND_TO_PLACE,
            B1Stage.WITHDRAW,
        ):
            return FailureReason.MOTION_STAGE_TIMEOUT
        if stage is B1Stage.CLOSE_GRIPPER:
            return FailureReason.GRASP_CANDIDATE_FAILED
        if stage is B1Stage.GRASP_CANDIDATE_CHECK:
            return FailureReason.GRASP_CANDIDATE_FAILED
        if stage is B1Stage.TRIAL_LIFT:
            return FailureReason.TRIAL_LIFT_FAILED
        if stage is B1Stage.GRASP_CONFIRMATION:
            return FailureReason.GRASP_NOT_CONFIRMED
        if stage is B1Stage.RELEASE:
            return FailureReason.RELEASE_FAILED
        if stage is B1Stage.FINAL_VISUAL_VERIFICATION:
            return FailureReason.FINAL_OBJECT_NOT_FOUND
        return FailureReason.TIMEOUT

    def _step(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> None:
        _, _, terminated, truncated, info = env.step(env.data.ctrl.copy())
        if step_callback is not None and step_callback(env) is False:
            raise B1ControllerFailure(
                FailureReason.TIMEOUT, "Viewer was closed before B1 completed"
            )
        if terminated and info.get("failure_reason") == "robot_table_collision":
            raise B1ControllerFailure(
                FailureReason.ROBOT_TABLE_COLLISION,
                f"Robot-table collision during {runtime.stage.value}",
            )
        if truncated:
            reason = self._stage_timeout_reason(runtime.stage)
            raise B1ControllerFailure(
                reason,
                f"Episode timeout during {runtime.stage.value}",
            )

    @staticmethod
    def _estimate(
        provider: TaskStateProvider,
        minimum_confidence: float,
    ) -> TaskStateEstimate:
        try:
            estimate = getattr(provider, "estimate", None)
            if callable(estimate):
                return estimate()
            observe = getattr(provider, "observe", None)
            if callable(observe):
                return task_state_from_perception_frame(
                    observe(),
                    source=provider.source,
                    minimum_confidence=minimum_confidence,
                )
            raise TypeError("provider exposes neither estimate() nor observe()")
        except Exception as exc:
            raise RuntimeError(
                f"External-state provider failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _sample_sensors(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        commanded_state: str,
    ) -> tuple[GripperFeedback, ContactFeedback]:
        if self._gripper_sensor is None or self._contact_sensor is None:
            raise RuntimeError("B1 sensors have not been initialized")
        gripper = self._gripper_sensor.read(commanded_state=commanded_state)
        contact = self._contact_sensor.read()
        runtime.metrics["bilateral_contact_duration"] = max(
            float(runtime.metrics["bilateral_contact_duration"]),
            float(contact.contact_duration),
        )
        return gripper, contact

    def _monitor_grasp(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        *,
        failure_reason: FailureReason,
    ) -> None:
        gripper, contact = self._sample_sensors(env, runtime, "closing")
        if runtime.grasp_monitor is None:
            raise RuntimeError("Grasp state machine was not initialized")
        update = runtime.grasp_monitor.update_transport(gripper, contact)
        runtime.metrics["contact_loss_event_count"] = (
            runtime.grasp_monitor.contact_loss_event_count
        )
        self._emit_diagnostic(
            env,
            runtime,
            event="transport_sample",
            gripper=gripper,
            contact=contact,
            update=update,
        )
        if update.state is GraspState.GRASP_LOST:
            runtime.metrics["grasp_lost"] = True
            candidate_aperture = runtime.grasp_monitor.candidate_aperture
            further_closure = (
                0.0
                if candidate_aperture is None
                else candidate_aperture - gripper.aperture
            )
            raise B1ControllerFailure(
                failure_reason,
                "Sustained bilateral-contact loss coincided with further gripper closure",
                errors={"aperture_drop": float(further_closure)},
            )

    def _motion_plan(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        target_position: np.ndarray,
        reference_duration: float,
    ) -> EventMotionPlan:
        if runtime.target_rotation is None:
            raise RuntimeError("B1 target orientation was not initialized")
        robot_observation = env.observation()
        ik_initial_qpos = env.model.qpos0.copy()
        ik_initial_qpos[env.arm_qpos_addresses] = robot_observation[
            "arm_joint_positions"
        ]
        ik_initial_qpos[env.finger_qpos_addresses] = robot_observation[
            "finger_positions"
        ]
        try:
            target, position_error, orientation_error, _ = solve_pose_ik(
                model=env.model,
                initial_qpos=ik_initial_qpos,
                tcp_site_id=env.tcp_site_id,
                arm_qpos_addresses=env.arm_qpos_addresses,
                arm_dof_addresses=env.arm_dof_addresses,
                arm_joint_ranges=env.arm_joint_ranges,
                target_position=np.asarray(target_position, dtype=float),
                target_rotation=runtime.target_rotation,
                config=self.controller_config,
            )
        except IKNotConvergedError as exc:
            raise B1ControllerFailure(FailureReason.IK_NOT_CONVERGED, str(exc)) from exc
        runtime.key_errors["ik_position_error"] = position_error
        runtime.key_errors["ik_orientation_error"] = orientation_error
        target = np.clip(target, env.arm_ctrl_ranges[:, 0], env.arm_ctrl_ranges[:, 1])
        return EventMotionPlan(
            start_time=float(env.data.time),
            reference_duration=reference_duration,
            start_control=env.data.ctrl[env.arm_actuator_ids].copy(),
            target_control=target,
            target_position=np.asarray(target_position, dtype=float).copy(),
        )

    def _move_until_arrived(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        target_position: np.ndarray,
        reference_duration: float,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
        *,
        monitor_grasp: bool = False,
        grasp_failure_reason: FailureReason = FailureReason.GRASP_LOST_DURING_TRANSFER,
        timeout: float | None = None,
        timeout_failure_reason: FailureReason | None = None,
    ) -> None:
        plan = self._motion_plan(env, runtime, target_position, reference_duration)
        deadline = plan.start_time + (
            self.config.motion_timeout if timeout is None else timeout
        )
        hold_steps = 0
        position_error = float("inf")
        orientation_error = float("inf")
        joint_speed = float("inf")
        while float(env.data.time) <= deadline + 1e-12:
            elapsed = float(env.data.time - plan.start_time)
            interpolation = smoothstep(elapsed / plan.reference_duration)
            env.data.ctrl[env.arm_actuator_ids] = plan.start_control + interpolation * (
                plan.target_control - plan.start_control
            )
            self._step(env, runtime, step_callback)
            if monitor_grasp:
                self._monitor_grasp(
                    env, runtime, failure_reason=grasp_failure_reason
                )
            observation = env.observation()
            current_position = np.asarray(observation["tcp_position"], dtype=float)
            current_rotation = np.asarray(observation["tcp_orientation"], dtype=float)
            joint_velocity = np.asarray(
                observation["arm_joint_velocities"], dtype=float
            )
            position_error = float(np.linalg.norm(plan.target_position - current_position))
            orientation_error = float(
                np.linalg.norm(rotation_error_world(current_rotation, runtime.target_rotation))
            )
            joint_speed = float(np.max(np.abs(joint_velocity)))

            strict = (
                position_error <= self.config.arrival_position_tolerance
                and orientation_error <= self.config.arrival_orientation_tolerance
                and joint_speed <= self.config.settled_joint_velocity_threshold
            )
            hysteresis = (
                hold_steps > 0
                and position_error <= 1.25 * self.config.arrival_position_tolerance
                and orientation_error
                <= 1.25 * self.config.arrival_orientation_tolerance
                and joint_speed
                <= 1.25 * self.config.settled_joint_velocity_threshold
            )
            hold_steps = hold_steps + 1 if strict or hysteresis else 0
            if hold_steps >= self.config.arrival_hold_steps:
                prefix = runtime.stage.value
                runtime.key_errors["waypoint_error"] = position_error
                runtime.key_errors["arrival_orientation_error"] = orientation_error
                runtime.key_errors["settled_joint_speed"] = joint_speed
                runtime.key_errors[f"{prefix}_position_error"] = position_error
                runtime.key_errors[f"{prefix}_orientation_error"] = orientation_error
                runtime.key_errors[f"{prefix}_joint_speed"] = joint_speed
                return

        errors = {
            "waypoint_error": position_error,
            "arrival_orientation_error": orientation_error,
            "settled_joint_speed": joint_speed,
        }
        runtime.key_errors.update(errors)
        prefix = runtime.stage.value
        runtime.key_errors[f"{prefix}_position_error"] = position_error
        runtime.key_errors[f"{prefix}_orientation_error"] = orientation_error
        runtime.key_errors[f"{prefix}_joint_speed"] = joint_speed
        pose_outside = (
            position_error > 1.25 * self.config.arrival_position_tolerance
            or orientation_error > 1.25 * self.config.arrival_orientation_tolerance
        )
        reason = timeout_failure_reason or (
            FailureReason.MOTION_STAGE_TIMEOUT
            if pose_outside
            else FailureReason.MOTION_NOT_SETTLED
        )
        raise B1ControllerFailure(
            reason,
            f"{runtime.stage.value} did not satisfy arrival and settling hold conditions",
            errors=errors,
        )

    def _collect_scene_perception(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> None:
        object_estimates: list[TaskStateEstimate] = []
        target_estimates: list[TaskStateEstimate] = []
        latencies: list[float] = []
        for frame_index in range(self.config.initial_perception_frames):
            try:
                sample = self._estimate(
                    provider, env.config.perception.minimum_confidence
                )
            except RuntimeError as exc:
                raise B1ControllerFailure(
                    FailureReason.INITIAL_PERCEPTION_FAILED, str(exc)
                ) from exc
            runtime.metrics["initial_perception_frame_count"] += 1
            runtime.metrics["initial_perception_timestamp"] = sample.timestamp
            if runtime.metrics["camera_name"] is None:
                runtime.metrics["camera_name"] = sample.camera_name
                runtime.metrics["image_resolution"] = sample.image_resolution
            latencies.append(float(sample.latency_ms))
            object_valid = _component_valid(
                sample,
                "object",
                env.config.perception.minimum_confidence,
            )
            target_valid = _component_valid(
                sample,
                "target",
                env.config.perception.minimum_confidence,
            )
            if object_valid and target_valid:
                object_estimates.append(sample)
                target_estimates.append(sample)
            if frame_index + 1 < self.config.initial_perception_frames:
                self._step(env, runtime, step_callback)

        valid_count = len(object_estimates)
        runtime.metrics["initial_valid_frame_count"] = valid_count
        runtime.metrics["initial_perception_latency_ms"] = (
            float(np.sum(latencies)) if latencies else None
        )
        if valid_count < self.config.minimum_valid_perception_frames:
            raise B1ControllerFailure(
                FailureReason.INITIAL_PERCEPTION_FAILED,
                f"Only {valid_count}/{self.config.initial_perception_frames} initial "
                "external-state samples contained valid object and target positions",
            )
        object_position, object_spread, object_confidence = _robust_positions(
            object_estimates, "object"
        )
        target_position, target_spread, target_confidence = _robust_positions(
            target_estimates, "target"
        )
        spread = max(object_spread, target_spread)
        runtime.metrics["initial_object_position_spread"] = object_spread
        runtime.metrics["initial_target_position_spread"] = target_spread
        runtime.metrics["initial_position_spread"] = spread
        if spread > self.config.maximum_position_spread:
            raise B1ControllerFailure(
                FailureReason.INITIAL_PERCEPTION_FAILED,
                f"Initial external-state position spread {spread:.6f} m exceeds "
                f"{self.config.maximum_position_spread:.6f} m",
                errors={"initial_position_spread": spread},
            )
        runtime.initial_object_position = object_position
        runtime.locked_target_position = target_position
        runtime.metrics.update(
            {
                "initial_object_position": tuple(float(v) for v in object_position),
                "locked_target_position": tuple(float(v) for v in target_position),
                "initial_object_confidence": object_confidence,
                "initial_target_confidence": target_confidence,
            }
        )

    def _collect_pregrasp_object(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> None:
        estimates: list[TaskStateEstimate] = []
        latencies: list[float] = []
        for frame_index in range(self.config.pregrasp_perception_frames):
            try:
                sample = self._estimate(
                    provider, env.config.perception.minimum_confidence
                )
            except RuntimeError as exc:
                raise B1ControllerFailure(
                    FailureReason.PREGRASP_REACQUISITION_FAILED, str(exc)
                ) from exc
            runtime.metrics["pregrasp_perception_frame_count"] += 1
            latencies.append(float(sample.latency_ms))
            if _component_valid(
                sample,
                "object",
                env.config.perception.minimum_confidence,
            ):
                estimates.append(sample)
            if frame_index + 1 < self.config.pregrasp_perception_frames:
                self._step(env, runtime, step_callback)

        runtime.metrics["pregrasp_valid_frame_count"] = len(estimates)
        runtime.metrics["pregrasp_perception_latency_ms"] = (
            float(np.sum(latencies)) if latencies else None
        )
        if len(estimates) < self.config.minimum_valid_pregrasp_frames:
            if (
                self.config.allow_initial_object_fallback
                and runtime.initial_object_position is not None
            ):
                runtime.corrected_object_position = runtime.initial_object_position.copy()
                runtime.metrics["pregrasp_used_initial_fallback"] = True
                runtime.metrics["pregrasp_corrected_object_position"] = tuple(
                    float(v) for v in runtime.corrected_object_position
                )
                runtime.metrics["pregrasp_correction_magnitude"] = 0.0
                return
            raise B1ControllerFailure(
                FailureReason.PREGRASP_REACQUISITION_FAILED,
                f"Only {len(estimates)}/{self.config.pregrasp_perception_frames} "
                "pregrasp samples contained a valid object position",
            )
        corrected, spread, _ = _robust_positions(estimates, "object")
        runtime.metrics["pregrasp_position_spread"] = spread
        if spread > self.config.maximum_position_spread:
            raise B1ControllerFailure(
                FailureReason.PREGRASP_POSITION_UNSTABLE,
                f"Pregrasp object spread {spread:.6f} m exceeds "
                f"{self.config.maximum_position_spread:.6f} m",
                errors={"pregrasp_position_spread": spread},
            )
        if runtime.initial_object_position is None:
            raise RuntimeError("Initial object memory is missing")
        correction = float(np.linalg.norm(corrected - runtime.initial_object_position))
        runtime.corrected_object_position = corrected
        runtime.metrics.update(
            {
                "pregrasp_corrected_object_position": tuple(float(v) for v in corrected),
                "pregrasp_correction_magnitude": correction,
                "pregrasp_correction_exceeded_threshold": bool(
                    correction > self.config.maximum_pregrasp_correction
                ),
            }
        )

    def _handle_scene_perception(
        self, env: PandaUTableEnv, runtime: B1Runtime, provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        self._collect_scene_perception(env, runtime, provider, step_callback)
        return B1Stage.MOVE_TO_PREGRASP

    def _handle_move_to_pregrasp(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.initial_object_position is None:
            raise RuntimeError("Initial object memory is missing")
        target = (
            runtime.initial_object_position
            + np.asarray(self.config.pregrasp_observation_offset, dtype=float)
            + np.array([0.0, 0.0, self.controller_config.waypoint_height])
        )
        self._move_until_arrived(
            env, runtime, target, self.controller_config.approach_duration, step_callback
        )
        return B1Stage.PREGRASP_REACQUISITION

    def _handle_pregrasp_reacquisition(
        self, env: PandaUTableEnv, runtime: B1Runtime, provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        self._collect_pregrasp_object(env, runtime, provider, step_callback)
        return B1Stage.DESCEND_TO_GRASP

    def _handle_descend_to_grasp(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.corrected_object_position is None:
            raise RuntimeError("Corrected object memory is missing")
        target = runtime.corrected_object_position + np.array(
            [0.0, 0.0, self.controller_config.grasp_z_offset]
        )
        self._move_until_arrived(
            env, runtime, target, self.controller_config.descent_duration, step_callback
        )
        return B1Stage.CLOSE_GRIPPER

    def _handle_close_gripper(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        gripper, _ = self._sample_sensors(env, runtime, "open")
        runtime.metrics["gripper_aperture_before_close"] = gripper.aperture
        if runtime.grasp_monitor is None:
            raise RuntimeError("Grasp state machine was not initialized")
        runtime.grasp_monitor.begin_closing()
        env.data.ctrl[env.gripper_actuator_id] = self.controller_config.gripper_close_control
        deadline = float(env.data.time) + self.config.close_timeout
        saw_left = False
        saw_right = False
        while float(env.data.time) <= deadline + 1e-12:
            self._step(env, runtime, step_callback)
            gripper, contact = self._sample_sensors(env, runtime, "closing")
            self._emit_diagnostic(
                env,
                runtime,
                event="close_sample",
                gripper=gripper,
                contact=contact,
            )
            saw_left = saw_left or contact.left_finger_object_contact
            saw_right = saw_right or contact.right_finger_object_contact
            if contact.bilateral_contact:
                self._emit_diagnostic(
                    env,
                    runtime,
                    event="close_gripper_complete",
                    gripper=gripper,
                    contact=contact,
                )
                return B1Stage.GRASP_CANDIDATE_CHECK
            if gripper.aperture <= self.config.empty_gripper_aperture_threshold:
                runtime.metrics["gripper_aperture_after_close"] = gripper.aperture
                raise B1ControllerFailure(
                    FailureReason.EMPTY_GRIPPER_CLOSURE,
                    "Gripper reached the calibrated empty-closure aperture",
                )
        reason = (
            FailureReason.BILATERAL_CONTACT_MISSING
            if saw_left or saw_right
            else FailureReason.GRASP_CANDIDATE_FAILED
        )
        raise B1ControllerFailure(reason, "Gripper close timed out without bilateral contact")

    def _handle_grasp_candidate_check(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        deadline = float(env.data.time) + self.config.close_timeout
        held = 0
        saw_bilateral = False
        last_gripper: GripperFeedback | None = None
        while float(env.data.time) <= deadline + 1e-12:
            env.data.ctrl[env.gripper_actuator_id] = self.controller_config.gripper_close_control
            self._step(env, runtime, step_callback)
            gripper, contact = self._sample_sensors(env, runtime, "closing")
            last_gripper = gripper
            saw_bilateral = saw_bilateral or contact.bilateral_contact
            if runtime.grasp_monitor is None:
                raise RuntimeError("Grasp state machine was not initialized")
            update = runtime.grasp_monitor.update_candidate(
                gripper,
                contact,
                robot_table_collision=env.robot_table_collision(),
            )
            held = runtime.grasp_monitor.candidate_steps
            self._emit_diagnostic(
                env,
                runtime,
                event="candidate_sample",
                gripper=gripper,
                contact=contact,
                update=update,
            )
            if update.state is GraspState.GRASP_CANDIDATE:
                runtime.metrics["gripper_aperture_after_close"] = gripper.aperture
                runtime.metrics["grasp_candidate"] = True
                return B1Stage.TRIAL_LIFT
            if update.empty_closure:
                runtime.metrics["gripper_aperture_after_close"] = gripper.aperture
                raise B1ControllerFailure(
                    FailureReason.EMPTY_GRIPPER_CLOSURE,
                    "Candidate check reached empty-gripper closure",
                )
        if last_gripper is not None:
            runtime.metrics["gripper_aperture_after_close"] = last_gripper.aperture
        reason = (
            FailureReason.GRASP_CANDIDATE_FAILED
            if saw_bilateral
            else FailureReason.BILATERAL_CONTACT_MISSING
        )
        raise B1ControllerFailure(reason, "Grasp-candidate conditions did not hold")

    def _handle_trial_lift(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        observation = env.observation()
        target = np.asarray(observation["tcp_position"], dtype=float) + np.array(
            [0.0, 0.0, self.config.trial_lift_distance]
        )
        self._move_until_arrived(
            env,
            runtime,
            target,
            min(self.controller_config.lift_duration, 0.6 * self.config.trial_lift_timeout),
            step_callback,
            monitor_grasp=True,
            grasp_failure_reason=FailureReason.TRIAL_LIFT_FAILED,
            timeout=self.config.trial_lift_timeout,
            timeout_failure_reason=FailureReason.TRIAL_LIFT_FAILED,
        )
        runtime.metrics["trial_lift_completed"] = True
        return B1Stage.GRASP_CONFIRMATION

    def _handle_grasp_confirmation(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        deadline = float(env.data.time) + self.config.trial_lift_timeout
        held = 0
        while float(env.data.time) <= deadline + 1e-12:
            self._step(env, runtime, step_callback)
            gripper, contact = self._sample_sensors(env, runtime, "closing")
            if runtime.grasp_monitor is None:
                raise RuntimeError("Grasp state machine was not initialized")
            update = runtime.grasp_monitor.update_confirmation(
                gripper,
                contact,
                trial_lift_completed=bool(runtime.metrics["trial_lift_completed"]),
                robot_table_collision=env.robot_table_collision(),
            )
            held = runtime.grasp_monitor.confirmation_steps
            self._emit_diagnostic(
                env,
                runtime,
                event="confirmation_sample",
                gripper=gripper,
                contact=contact,
                update=update,
            )
            if update.state is GraspState.GRASP_CONFIRMED:
                runtime.metrics["grasp_confirmed"] = True
                return B1Stage.TRANSFER
        raise B1ControllerFailure(
            FailureReason.GRASP_NOT_CONFIRMED,
            "Trial-lift feedback did not remain stable for confirmation hold steps",
        )

    def _handle_transfer(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.corrected_object_position is None or runtime.locked_target_position is None:
            raise RuntimeError("B1 task memory is incomplete")
        transport_z = float(
            runtime.corrected_object_position[2] + self.controller_config.lift_height
        )
        target = np.array(
            [runtime.locked_target_position[0], runtime.locked_target_position[1], transport_z]
        )
        self._move_until_arrived(
            env,
            runtime,
            target,
            self.controller_config.transfer_duration,
            step_callback,
            monitor_grasp=True,
        )
        return B1Stage.DESCEND_TO_PLACE

    def _handle_descend_to_place(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.locked_target_position is None:
            raise RuntimeError("Locked target memory is missing")
        place_tcp_z = float(
            runtime.locked_target_position[2]
            - env.config.workspace.target_site_offset
            + env.config.workspace.object_half_size
            + self.controller_config.grasp_z_offset
        )
        target = np.array(
            [runtime.locked_target_position[0], runtime.locked_target_position[1], place_tcp_z]
        )
        self._move_until_arrived(
            env,
            runtime,
            target,
            self.controller_config.descent_duration,
            step_callback,
            monitor_grasp=True,
        )
        return B1Stage.RELEASE

    def _handle_release(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        env.data.ctrl[env.gripper_actuator_id] = self.controller_config.gripper_open_control
        deadline = float(env.data.time) + self.config.release_timeout
        held = 0
        while float(env.data.time) <= deadline + 1e-12:
            self._step(env, runtime, step_callback)
            gripper, _ = self._sample_sensors(env, runtime, "opening")
            self._emit_diagnostic(
                env,
                runtime,
                event="release_sample",
                gripper=gripper,
            )
            held = held + 1 if gripper.aperture >= self.config.release_aperture_threshold else 0
            if held >= self.config.arrival_hold_steps:
                if runtime.grasp_monitor is None:
                    raise RuntimeError("Grasp state machine was not initialized")
                runtime.grasp_monitor.mark_released()
                return B1Stage.WITHDRAW
        raise B1ControllerFailure(
            FailureReason.RELEASE_FAILED,
            "Gripper did not reach and hold the configured release aperture",
        )

    def _handle_withdraw(
        self, env: PandaUTableEnv, runtime: B1Runtime, _provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.locked_target_position is None:
            raise RuntimeError("Locked target memory is missing")
        target = (
            runtime.locked_target_position
            + np.asarray(self.config.final_observation_offset, dtype=float)
            + np.array([0.0, 0.0, self.controller_config.waypoint_height])
        )
        self._move_until_arrived(
            env, runtime, target, self.controller_config.withdraw_duration, step_callback
        )
        return B1Stage.FINAL_VISUAL_VERIFICATION

    def _handle_final_visual_verification(
        self, env: PandaUTableEnv, runtime: B1Runtime, provider: TaskStateProvider,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None,
    ) -> B1Stage:
        if runtime.locked_target_position is None:
            raise RuntimeError("Locked target memory is missing")
        estimates: list[TaskStateEstimate] = []
        latencies: list[float] = []
        for frame_index in range(self.config.final_verification_frames):
            try:
                sample = self._estimate(
                    provider, env.config.perception.minimum_confidence
                )
            except RuntimeError as exc:
                raise B1ControllerFailure(
                    FailureReason.FINAL_OBJECT_NOT_FOUND, str(exc)
                ) from exc
            runtime.metrics["final_visual_frame_count"] += 1
            latencies.append(float(sample.latency_ms))
            if _component_valid(
                sample,
                "object",
                env.config.perception.minimum_confidence,
            ):
                estimates.append(sample)
            if frame_index + 1 < self.config.final_verification_frames:
                self._step(env, runtime, step_callback)
        runtime.metrics["final_visual_valid_frame_count"] = len(estimates)
        runtime.metrics["final_visual_latency_ms"] = (
            float(np.sum(latencies)) if latencies else None
        )
        if len(estimates) < self.config.final_minimum_valid_frames:
            raise B1ControllerFailure(
                FailureReason.FINAL_OBJECT_NOT_FOUND,
                f"Only {len(estimates)}/{self.config.final_verification_frames} final "
                "samples contained the released object",
            )
        final_object, spread, _ = _robust_positions(estimates, "object")
        runtime.key_errors["final_visual_position_spread"] = spread
        xy_error = float(
            np.linalg.norm(final_object[:2] - runtime.locked_target_position[:2])
        )
        expected_center_z = float(
            runtime.locked_target_position[2]
            - env.config.workspace.target_site_offset
            + env.config.workspace.object_half_size
        )
        height_error = abs(float(final_object[2] - expected_center_z))
        runtime.metrics.update(
            {
                "final_visual_object_position": tuple(float(v) for v in final_object),
                "final_visual_xy_error": xy_error,
                "final_visual_height_error": height_error,
            }
        )
        runtime.key_errors["final_visual_xy_error"] = xy_error
        runtime.key_errors["final_visual_height_error"] = height_error
        if xy_error > self.config.final_place_xy_tolerance:
            raise B1ControllerFailure(
                FailureReason.FINAL_VISUAL_PLACE_XY_ERROR,
                f"Final visual XY error {xy_error:.6f} m exceeds "
                f"{self.config.final_place_xy_tolerance:.6f} m",
                errors={"final_visual_xy_error": xy_error},
            )
        if height_error > self.config.final_place_height_tolerance:
            raise B1ControllerFailure(
                FailureReason.FINAL_VISUAL_PLACE_HEIGHT_ERROR,
                f"Final visual height error {height_error:.6f} m exceeds "
                f"{self.config.final_place_height_tolerance:.6f} m",
                errors={"final_visual_height_error": height_error},
            )
        runtime.metrics["controller_reported_success"] = True
        return B1Stage.COMPLETED

    def _build_result(
        self,
        env: PandaUTableEnv,
        runtime: B1Runtime,
        *,
        success: bool,
        failure_reason: FailureReason | None,
        exception_message: str | None,
    ) -> EpisodeResult:
        runtime.metrics["controller_reported_success"] = success
        runtime.metrics["final_failure_reason"] = (
            None if failure_reason is None else failure_reason.value
        )
        runtime.metrics["stage_durations"] = dict(runtime.stage_durations)
        if runtime.stage is B1Stage.COMPLETED:
            runtime.metrics["stage_durations"].setdefault(B1Stage.COMPLETED.value, 0.0)
        self._emit_diagnostic(
            env,
            runtime,
            event="episode_end",
            failure_reason=failure_reason,
        )
        return build_episode_result(
            env,
            EpisodeOutcome(
                success=success,
                failure_reason=failure_reason,
                stage=runtime.stage.value,
                lift_height=(
                    self.config.trial_lift_distance
                    if runtime.metrics.get("trial_lift_completed", False)
                    else None
                ),
                exception_message=exception_message,
                key_errors=runtime.key_errors,
                controller_type=self.controller_type,
                b1_metrics=runtime.metrics,
            ),
            task_state=None,
            perception_metrics=None,
        )

    def run_episode(
        self,
        env: PandaUTableEnv,
        *,
        seed: int | None = None,
        state_provider: TaskStateProvider | None = None,
        step_callback: Callable[[PandaUTableEnv], bool | None] | None = None,
        diagnostic_observer: Callable[[B1DiagnosticSnapshot], None] | None = None,
    ) -> EpisodeResult:
        runtime = B1Runtime(diagnostic_observer=diagnostic_observer)
        try:
            env.reset(seed=seed)
            runtime = self._make_runtime(
                env,
                diagnostic_observer=diagnostic_observer,
            )
            self._emit_diagnostic(env, runtime, event="episode_reset")
        except InvalidResetError as exc:
            runtime.metrics = {"controller_type": self.controller_type}
            return self._build_result(
                env,
                runtime,
                success=False,
                failure_reason=FailureReason.INVALID_RESET,
                exception_message=str(exc),
            )
        except Exception:
            runtime.metrics = {"controller_type": self.controller_type}
            return self._build_result(
                env,
                runtime,
                success=False,
                failure_reason=FailureReason.UNEXPECTED_EXCEPTION,
                exception_message=traceback.format_exc(),
            )

        try:
            if env.config.observation.source != "perception":
                raise ValueError("sensor_event_b1 requires observation source 'perception'")
            if (
                state_provider is None
                or state_provider.source not in self.external_state_sources
            ):
                raise B1ControllerFailure(
                    FailureReason.INITIAL_PERCEPTION_FAILED,
                    "sensor_event_b1 requires an explicit perception or oracle "
                    "external-state provider",
                )
            if not (
                callable(getattr(state_provider, "estimate", None))
                or callable(getattr(state_provider, "observe", None))
            ):
                raise B1ControllerFailure(
                    FailureReason.INITIAL_PERCEPTION_FAILED,
                    "B1 provider does not expose external-state samples",
                )
            runtime.metrics["external_state_provider_source"] = state_provider.source
            self._gripper_sensor = GripperFeedbackSensor(env.model, env.data)
            self._contact_sensor = ContactSensor(
                env.model,
                env.data,
                object_geom_name="pick_object_geom",
                present_debounce_steps=self.config.contact_debounce_steps,
                absent_debounce_steps=self.config.contact_debounce_steps,
            )
            handlers = {
                B1Stage.SCENE_PERCEPTION: self._handle_scene_perception,
                B1Stage.MOVE_TO_PREGRASP: self._handle_move_to_pregrasp,
                B1Stage.PREGRASP_REACQUISITION: self._handle_pregrasp_reacquisition,
                B1Stage.DESCEND_TO_GRASP: self._handle_descend_to_grasp,
                B1Stage.CLOSE_GRIPPER: self._handle_close_gripper,
                B1Stage.GRASP_CANDIDATE_CHECK: self._handle_grasp_candidate_check,
                B1Stage.TRIAL_LIFT: self._handle_trial_lift,
                B1Stage.GRASP_CONFIRMATION: self._handle_grasp_confirmation,
                B1Stage.TRANSFER: self._handle_transfer,
                B1Stage.DESCEND_TO_PLACE: self._handle_descend_to_place,
                B1Stage.RELEASE: self._handle_release,
                B1Stage.WITHDRAW: self._handle_withdraw,
                B1Stage.FINAL_VISUAL_VERIFICATION: self._handle_final_visual_verification,
            }
            while runtime.stage is not B1Stage.COMPLETED:
                handler = handlers[runtime.stage]
                next_stage = handler(env, runtime, state_provider, step_callback)
                self._transition(runtime, env, next_stage)
            runtime.metrics["stage_durations"] = dict(runtime.stage_durations)
            return self._build_result(
                env,
                runtime,
                success=True,
                failure_reason=None,
                exception_message=None,
            )
        except B1ControllerFailure as exc:
            runtime.key_errors.update(exc.errors)
            self._finish_current_stage(runtime, env)
            return self._build_result(
                env,
                runtime,
                success=False,
                failure_reason=exc.reason,
                exception_message=str(exc),
            )
        except Exception:
            self._finish_current_stage(runtime, env)
            return self._build_result(
                env,
                runtime,
                success=False,
                failure_reason=FailureReason.UNEXPECTED_EXCEPTION,
                exception_message=traceback.format_exc(),
            )
