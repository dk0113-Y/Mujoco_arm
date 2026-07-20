from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from environments.panda_u_table_env import PandaUTableEnv
from perception.types import PerceptionMetrics, TaskStateEstimate

from .episode_result import EpisodeResult, FailureReason


_B1_PERCEPTION_FAILURE_REASONS = frozenset(
    {
        FailureReason.INITIAL_PERCEPTION_FAILED,
        FailureReason.PREGRASP_REACQUISITION_FAILED,
        FailureReason.PREGRASP_POSITION_UNSTABLE,
        FailureReason.FINAL_OBJECT_NOT_FOUND,
        FailureReason.FINAL_VISUAL_PLACE_XY_ERROR,
        FailureReason.FINAL_VISUAL_PLACE_HEIGHT_ERROR,
    }
)


def evaluate_task_state(
    env: PandaUTableEnv,
    estimate: TaskStateEstimate,
) -> PerceptionMetrics:
    """Compare an already-produced estimate with same-timestamp privileged labels."""
    object_xy_error: float | None = None
    object_z_error: float | None = None
    object_3d_error: float | None = None
    target_xy_error: float | None = None
    target_z_error: float | None = None
    target_3d_error: float | None = None
    if estimate.object_position is not None:
        object_delta = np.asarray(estimate.object_position) - env.data.xpos[
            env.object_body_id
        ]
        object_xy_error = float(np.linalg.norm(object_delta[:2]))
        object_z_error = abs(float(object_delta[2]))
        object_3d_error = float(np.linalg.norm(object_delta))
    if estimate.target_position is not None:
        target_delta = np.asarray(estimate.target_position) - env.data.site_xpos[
            env.place_target_site_id
        ]
        target_xy_error = float(np.linalg.norm(target_delta[:2]))
        target_z_error = abs(float(target_delta[2]))
        target_3d_error = float(np.linalg.norm(target_delta))
    return PerceptionMetrics(
        object_xy_error=object_xy_error,
        object_z_error=object_z_error,
        object_3d_error=object_3d_error,
        target_xy_error=target_xy_error,
        target_z_error=target_z_error,
        target_3d_error=target_3d_error,
    )


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    failure_reason: FailureReason | None
    stage: str
    lift_height: float | None
    exception_message: str | None
    key_errors: dict[str, float]
    controller_type: str = "fixed_dls_b0"
    b1_metrics: dict[str, Any] | None = None


def build_episode_result(
    env: PandaUTableEnv,
    outcome: EpisodeOutcome,
    task_state: TaskStateEstimate | None,
    perception_metrics: PerceptionMetrics | None,
) -> EpisodeResult:
    """Privileged result recorder; its outputs never feed controller planning."""
    episode = env.current_episode
    final_tcp: tuple[float, float, float] | None = None
    final_object: tuple[float, float, float] | None = None
    target: tuple[float, float, float] | None = None
    xy_error: float | None = None
    height_error: float | None = None
    if episode is not None:
        final_tcp = tuple(float(value) for value in env.data.site_xpos[env.tcp_site_id])
        final_object = tuple(float(value) for value in env.data.xpos[env.object_body_id])
        target = tuple(
            float(value) for value in env.data.site_xpos[env.place_target_site_id]
        )
        xy_error, height_error = env.placement_errors()

    b1_metrics = dict(outcome.b1_metrics or {})
    is_b1 = outcome.controller_type == "sensor_event_b1"
    perception_stage_failed = bool(
        outcome.failure_reason is not None
        and (
            outcome.failure_reason.value.startswith("perception_")
            or (is_b1 and outcome.failure_reason in _B1_PERCEPTION_FAILURE_REASONS)
        )
    )
    controller_reported_success: bool | None = None
    privileged_ground_truth_success: bool | None = None
    false_positive: bool | None = None
    false_negative: bool | None = None
    if is_b1:
        controller_reported_success = bool(outcome.success)
        if xy_error is not None and height_error is not None:
            privileged_ground_truth_success = bool(
                xy_error <= env.config.b1.final_place_xy_tolerance
                and height_error <= env.config.b1.final_place_height_tolerance
            )
            false_positive = bool(
                controller_reported_success and not privileged_ground_truth_success
            )
            false_negative = bool(
                not controller_reported_success and privileged_ground_truth_success
            )

    source = (
        str(b1_metrics.get("external_state_provider_source"))
        if is_b1 and b1_metrics.get("external_state_provider_source") is not None
        else (env.config.observation.source if task_state is None else task_state.source)
    )
    perception_failure_reason: str | None = None
    if source == "perception":
        perception_failure_reason = (
            None if task_state is None else task_state.failure_reason
        )
        if perception_stage_failed and outcome.failure_reason is not None:
            perception_failure_reason = outcome.failure_reason.value
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
        success=outcome.success,
        failure_reason=(
            None if outcome.failure_reason is None else outcome.failure_reason.value
        ),
        final_stage=outcome.stage,
        simulation_time=float(env.data.time),
        lift_height=outcome.lift_height,
        final_xy_error=xy_error,
        final_height_error=height_error,
        collision_count=env.collision_count,
        exception_message=outcome.exception_message,
        final_tcp_position=final_tcp,
        final_object_position=final_object,
        target_position=target,
        key_errors=dict(outcome.key_errors) or None,
        observation_source=source,
        perception_success=(
            None
            if source != "perception"
            else (
                bool(
                    b1_metrics.get("initial_object_position") is not None
                    and b1_metrics.get("locked_target_position") is not None
                    and not perception_stage_failed
                )
                if is_b1
                else bool(
                    task_state
                    and task_state.valid
                    and not (
                        outcome.failure_reason
                        and outcome.failure_reason.value.startswith("perception_")
                    )
                )
            )
        ),
        estimated_object_position=(
            b1_metrics.get("initial_object_position")
            if is_b1
            else (None if task_state is None else task_state.object_position)
        ),
        estimated_target_position=(
            b1_metrics.get("locked_target_position")
            if is_b1
            else (None if task_state is None else task_state.target_position)
        ),
        object_position_error=(
            None if perception_metrics is None else perception_metrics.object_3d_error
        ),
        target_position_error=(
            None if perception_metrics is None else perception_metrics.target_3d_error
        ),
        perception_failure_reason=perception_failure_reason,
        perception_latency_ms=(
            b1_metrics.get("initial_perception_latency_ms")
            if is_b1
            else (
                None
                if task_state is None or source != "perception"
                else task_state.latency_ms
            )
        ),
        camera_name=(
            b1_metrics.get("camera_name")
            if is_b1
            else (None if task_state is None else task_state.camera_name)
        ),
        image_resolution=(
            b1_metrics.get("image_resolution")
            if is_b1
            else (None if task_state is None else task_state.image_resolution)
        ),
        controller_type=outcome.controller_type,
        initial_perception_frame_count=b1_metrics.get(
            "initial_perception_frame_count"
        ),
        initial_valid_frame_count=b1_metrics.get("initial_valid_frame_count"),
        initial_object_position=b1_metrics.get("initial_object_position"),
        locked_target_position=b1_metrics.get("locked_target_position"),
        initial_object_confidence=b1_metrics.get("initial_object_confidence"),
        initial_target_confidence=b1_metrics.get("initial_target_confidence"),
        initial_position_spread=b1_metrics.get("initial_position_spread"),
        initial_object_position_spread=b1_metrics.get(
            "initial_object_position_spread"
        ),
        initial_target_position_spread=b1_metrics.get(
            "initial_target_position_spread"
        ),
        initial_perception_latency_ms=b1_metrics.get(
            "initial_perception_latency_ms"
        ),
        initial_perception_timestamp=b1_metrics.get(
            "initial_perception_timestamp"
        ),
        pregrasp_perception_frame_count=b1_metrics.get(
            "pregrasp_perception_frame_count"
        ),
        pregrasp_valid_frame_count=b1_metrics.get("pregrasp_valid_frame_count"),
        pregrasp_corrected_object_position=b1_metrics.get(
            "pregrasp_corrected_object_position"
        ),
        pregrasp_correction_magnitude=b1_metrics.get(
            "pregrasp_correction_magnitude"
        ),
        pregrasp_position_spread=b1_metrics.get("pregrasp_position_spread"),
        pregrasp_perception_latency_ms=b1_metrics.get(
            "pregrasp_perception_latency_ms"
        ),
        pregrasp_used_initial_fallback=b1_metrics.get(
            "pregrasp_used_initial_fallback"
        ),
        pregrasp_correction_exceeded_threshold=b1_metrics.get(
            "pregrasp_correction_exceeded_threshold"
        ),
        gripper_aperture_before_close=b1_metrics.get(
            "gripper_aperture_before_close"
        ),
        gripper_aperture_after_close=b1_metrics.get(
            "gripper_aperture_after_close"
        ),
        bilateral_contact_duration=b1_metrics.get("bilateral_contact_duration"),
        grasp_candidate=b1_metrics.get("grasp_candidate"),
        trial_lift_completed=b1_metrics.get("trial_lift_completed"),
        grasp_confirmed=b1_metrics.get("grasp_confirmed"),
        contact_loss_event_count=b1_metrics.get("contact_loss_event_count"),
        grasp_lost=b1_metrics.get("grasp_lost"),
        final_visual_frame_count=b1_metrics.get("final_visual_frame_count"),
        final_visual_valid_frame_count=b1_metrics.get(
            "final_visual_valid_frame_count"
        ),
        final_visual_object_position=b1_metrics.get("final_visual_object_position"),
        final_visual_xy_error=b1_metrics.get("final_visual_xy_error"),
        final_visual_height_error=b1_metrics.get("final_visual_height_error"),
        final_visual_latency_ms=b1_metrics.get("final_visual_latency_ms"),
        controller_reported_success=controller_reported_success,
        privileged_ground_truth_success=privileged_ground_truth_success,
        false_positive=false_positive,
        false_negative=false_negative,
        stage_durations=(
            None
            if b1_metrics.get("stage_durations") is None
            else dict(b1_metrics["stage_durations"])
        ),
        final_failure_reason=(
            b1_metrics.get("final_failure_reason")
            if "final_failure_reason" in b1_metrics
            else (
                None
                if outcome.failure_reason is None
                else outcome.failure_reason.value
            )
        ),
    )
