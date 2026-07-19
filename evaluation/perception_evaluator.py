from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from environments.panda_u_table_env import PandaUTableEnv
from perception.types import PerceptionMetrics, TaskStateEstimate

from .episode_result import EpisodeResult, FailureReason


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

    source = env.config.observation.source if task_state is None else task_state.source
    perception_failure_reason: str | None = None
    if source == "perception":
        perception_failure_reason = (
            None if task_state is None else task_state.failure_reason
        )
        if (
            outcome.failure_reason is not None
            and outcome.failure_reason.value.startswith("perception_")
        ):
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
            else bool(
                task_state
                and task_state.valid
                and not (
                    outcome.failure_reason
                    and outcome.failure_reason.value.startswith("perception_")
                )
            )
        ),
        estimated_object_position=(
            None if task_state is None else task_state.object_position
        ),
        estimated_target_position=(
            None if task_state is None else task_state.target_position
        ),
        object_position_error=(
            None if perception_metrics is None else perception_metrics.object_3d_error
        ),
        target_position_error=(
            None if perception_metrics is None else perception_metrics.target_3d_error
        ),
        perception_failure_reason=perception_failure_reason,
        perception_latency_ms=(
            None if task_state is None or source != "perception" else task_state.latency_ms
        ),
        camera_name=None if task_state is None else task_state.camera_name,
        image_resolution=None if task_state is None else task_state.image_resolution,
    )
