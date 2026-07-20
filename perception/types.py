from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    position_world: np.ndarray
    rotation_world_from_camera: np.ndarray
    camera_to_world: np.ndarray
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    simulation_time: float
    camera_name: str
    width: int
    height: int
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    depth_semantics: str = "metric axial depth along camera -Z"


@dataclass(frozen=True)
class DetectionResult:
    detection_id: str
    success: bool
    mask: np.ndarray
    pixel_count: int
    center_pixel: tuple[float, float] | None
    position: tuple[float, float, float] | None
    confidence: float
    failure_reason: str | None


@dataclass(frozen=True)
class TaskPerceptionFrame:
    """Independent object/target detections from one truth-isolated RGB-D frame."""

    object_detection: DetectionResult
    target_detection: DetectionResult
    timestamp: float
    latency_ms: float
    camera_name: str
    image_resolution: tuple[int, int]


@dataclass(frozen=True)
class TaskStateEstimate:
    object_id: str
    target_id: str
    object_position: tuple[float, float, float] | None
    target_position: tuple[float, float, float] | None
    timestamp: float
    source: str
    valid: bool
    confidence: float
    failure_reason: str | None
    object_pixel_count: int
    target_pixel_count: int
    latency_ms: float
    camera_name: str | None
    image_resolution: tuple[int, int] | None
    object_valid: bool | None = None
    target_valid: bool | None = None
    object_confidence: float | None = None
    target_confidence: float | None = None
    object_failure_reason: str | None = None
    target_failure_reason: str | None = None


@dataclass(frozen=True)
class PerceptionMetrics:
    object_xy_error: float | None
    object_z_error: float | None
    object_3d_error: float | None
    target_xy_error: float | None
    target_z_error: float | None
    target_3d_error: float | None
