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
