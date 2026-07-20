from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any


class FailureReason(str, Enum):
    INVALID_RESET = "invalid_reset"
    IK_NOT_CONVERGED = "ik_not_converged"
    WAYPOINT_ERROR = "waypoint_error"
    ROBOT_TABLE_COLLISION = "robot_table_collision"
    LIFT_FAILURE = "lift_failure"
    DROPPED_OBJECT = "dropped_object"
    PLACE_XY_ERROR = "place_xy_error"
    PLACE_HEIGHT_ERROR = "place_height_error"
    TIMEOUT = "timeout"
    PERCEPTION_OBJECT_NOT_FOUND = "perception_object_not_found"
    PERCEPTION_TARGET_NOT_FOUND = "perception_target_not_found"
    PERCEPTION_INVALID_DEPTH = "perception_invalid_depth"
    PERCEPTION_PROJECTION_ERROR = "perception_projection_error"
    PERCEPTION_LOW_CONFIDENCE = "perception_low_confidence"
    INITIAL_PERCEPTION_FAILED = "initial_perception_failed"
    PREGRASP_REACQUISITION_FAILED = "pregrasp_reacquisition_failed"
    PREGRASP_POSITION_UNSTABLE = "pregrasp_position_unstable"
    MOTION_STAGE_TIMEOUT = "motion_stage_timeout"
    MOTION_NOT_SETTLED = "motion_not_settled"
    GRASP_CANDIDATE_FAILED = "grasp_candidate_failed"
    BILATERAL_CONTACT_MISSING = "bilateral_contact_missing"
    EMPTY_GRIPPER_CLOSURE = "empty_gripper_closure"
    TRIAL_LIFT_FAILED = "trial_lift_failed"
    GRASP_NOT_CONFIRMED = "grasp_not_confirmed"
    GRASP_LOST_DURING_TRANSFER = "grasp_lost_during_transfer"
    RELEASE_FAILED = "release_failed"
    FINAL_OBJECT_NOT_FOUND = "final_object_not_found"
    FINAL_VISUAL_PLACE_XY_ERROR = "final_visual_place_xy_error"
    FINAL_VISUAL_PLACE_HEIGHT_ERROR = "final_visual_place_height_error"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


@dataclass(frozen=True)
class EpisodeResult:
    seed: int | None
    pick_mode: str
    place_mode: str
    physics_mode: str
    pick_region: str | None
    place_region: str | None
    sampled_pick_position: tuple[float, float, float] | None
    sampled_place_position: tuple[float, float, float] | None
    sampled_mass: float | None
    sampled_friction: tuple[float, float, float] | None
    success: bool
    failure_reason: str | None
    final_stage: str
    simulation_time: float
    lift_height: float | None
    final_xy_error: float | None
    final_height_error: float | None
    collision_count: int
    exception_message: str | None
    final_tcp_position: tuple[float, float, float] | None = None
    final_object_position: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    key_errors: dict[str, float] | None = None
    observation_source: str = "privileged"
    perception_success: bool | None = None
    estimated_object_position: tuple[float, float, float] | None = None
    estimated_target_position: tuple[float, float, float] | None = None
    object_position_error: float | None = None
    target_position_error: float | None = None
    perception_failure_reason: str | None = None
    perception_latency_ms: float | None = None
    camera_name: str | None = None
    image_resolution: tuple[int, int] | None = None
    controller_type: str = "fixed_dls_b0"
    initial_perception_frame_count: int | None = None
    initial_valid_frame_count: int | None = None
    initial_object_position: tuple[float, float, float] | None = None
    locked_target_position: tuple[float, float, float] | None = None
    initial_object_confidence: float | None = None
    initial_target_confidence: float | None = None
    initial_position_spread: float | None = None
    initial_object_position_spread: float | None = None
    initial_target_position_spread: float | None = None
    initial_perception_latency_ms: float | None = None
    initial_perception_timestamp: float | None = None
    pregrasp_perception_frame_count: int | None = None
    pregrasp_valid_frame_count: int | None = None
    pregrasp_corrected_object_position: tuple[float, float, float] | None = None
    pregrasp_correction_magnitude: float | None = None
    pregrasp_position_spread: float | None = None
    pregrasp_perception_latency_ms: float | None = None
    pregrasp_used_initial_fallback: bool | None = None
    pregrasp_correction_exceeded_threshold: bool | None = None
    gripper_aperture_before_close: float | None = None
    gripper_aperture_after_close: float | None = None
    bilateral_contact_duration: float | None = None
    grasp_candidate: bool | None = None
    trial_lift_completed: bool | None = None
    grasp_confirmed: bool | None = None
    contact_loss_event_count: int | None = None
    grasp_lost: bool | None = None
    final_visual_frame_count: int | None = None
    final_visual_valid_frame_count: int | None = None
    final_visual_object_position: tuple[float, float, float] | None = None
    final_visual_xy_error: float | None = None
    final_visual_height_error: float | None = None
    final_visual_latency_ms: float | None = None
    controller_reported_success: bool | None = None
    privileged_ground_truth_success: bool | None = None
    false_positive: bool | None = None
    false_negative: bool | None = None
    stage_durations: dict[str, float] | None = None
    final_failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible plain dictionary."""
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a scalar-only row suitable for ``csv.DictWriter``."""
        data = self.to_dict()
        key_errors = data.pop("key_errors") or {}
        stage_durations = data.pop("stage_durations") or {}
        for name, value in key_errors.items():
            data[f"key_error.{name}"] = value
        for name, value in stage_durations.items():
            data[f"stage_duration.{name}"] = value
        for name, value in tuple(data.items()):
            if isinstance(value, (dict, list, tuple)):
                data[name] = json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
        return data
