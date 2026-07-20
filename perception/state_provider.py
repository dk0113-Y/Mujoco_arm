from __future__ import annotations

from dataclasses import replace
import time
from typing import Protocol

import mujoco
import numpy as np

from environments.panda_u_table_env import PandaUTableEnv

from .color_depth_detector import ColorDepthDetector
from .overhead_rgbd import OverheadRGBDCamera
from .types import (
    DetectionResult,
    RGBDFrame,
    TaskPerceptionFrame,
    TaskStateEstimate,
)


class TaskStateProvider(Protocol):
    source: str

    def estimate(self) -> TaskStateEstimate:
        ...

    def close(self) -> None:
        ...


class PerceptionFrameProvider(TaskStateProvider, Protocol):
    def observe(self) -> TaskPerceptionFrame:
        ...


def task_state_from_perception_frame(
    frame: TaskPerceptionFrame,
    *,
    source: str = "perception",
    minimum_confidence: float = 0.0,
) -> TaskStateEstimate:
    """Adapt component RGB-D detections to the shared external-state contract."""
    object_detection = frame.object_detection
    target_detection = frame.target_detection
    object_valid = bool(
        object_detection.success
        and object_detection.position is not None
        and object_detection.confidence >= minimum_confidence
        and np.all(np.isfinite(object_detection.position))
    )
    target_valid = bool(
        target_detection.success
        and target_detection.position is not None
        and target_detection.confidence >= minimum_confidence
        and np.all(np.isfinite(target_detection.position))
    )
    failure_reason: str | None = None
    if not object_valid:
        failure_reason = object_detection.failure_reason
    elif not target_valid:
        failure_reason = target_detection.failure_reason
    if failure_reason is None and not (object_valid and target_valid):
        failure_reason = "perception_low_confidence"
    return TaskStateEstimate(
        object_id=object_detection.detection_id,
        target_id=target_detection.detection_id,
        object_position=object_detection.position,
        target_position=target_detection.position,
        timestamp=frame.timestamp,
        source=source,
        valid=bool(object_valid and target_valid),
        confidence=float(
            min(object_detection.confidence, target_detection.confidence)
        ),
        failure_reason=failure_reason,
        object_pixel_count=object_detection.pixel_count,
        target_pixel_count=target_detection.pixel_count,
        latency_ms=frame.latency_ms,
        camera_name=frame.camera_name,
        image_resolution=frame.image_resolution,
        object_valid=object_valid,
        target_valid=target_valid,
        object_confidence=float(object_detection.confidence),
        target_confidence=float(target_detection.confidence),
        object_failure_reason=object_detection.failure_reason,
        target_failure_reason=target_detection.failure_reason,
    )


class PrivilegedStateProvider:
    source = "privileged"

    def __init__(self, env: PandaUTableEnv) -> None:
        self.env = env

    def estimate(self) -> TaskStateEstimate:
        start = time.perf_counter()
        object_position = tuple(
            float(value) for value in self.env.data.xpos[self.env.object_body_id]
        )
        target_position = tuple(
            float(value)
            for value in self.env.data.site_xpos[self.env.place_target_site_id]
        )
        return TaskStateEstimate(
            object_id="pick_object_0",
            target_id="place_target_0",
            object_position=object_position,
            target_position=target_position,
            timestamp=float(self.env.data.time),
            source=self.source,
            valid=True,
            confidence=1.0,
            failure_reason=None,
            object_pixel_count=0,
            target_pixel_count=0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            camera_name=None,
            image_resolution=None,
            object_valid=True,
            target_valid=True,
            object_confidence=1.0,
            target_confidence=1.0,
        )

    def close(self) -> None:
        return None


class RGBDPerceptionProvider:
    """Task-state provider whose only external-state input is an RGB-D frame."""

    source = "perception"

    def __init__(
        self,
        camera: OverheadRGBDCamera,
        data: mujoco.MjData,
        detector: ColorDepthDetector,
    ) -> None:
        self.camera = camera
        self.data = data
        self.detector = detector
        self.last_frame: RGBDFrame | None = None
        self.last_object_detection: DetectionResult | None = None
        self.last_target_detection: DetectionResult | None = None

    def observe(self) -> TaskPerceptionFrame:
        """Return both component detections without imposing combined validity."""
        start = time.perf_counter()
        frame = self.camera.capture(self.data)
        object_detection = self.detector.detect_object(frame)
        target_detection = self.detector.detect_target(frame)
        self.last_frame = frame
        self.last_object_detection = object_detection
        self.last_target_detection = target_detection
        return TaskPerceptionFrame(
            object_detection=object_detection,
            target_detection=target_detection,
            timestamp=frame.simulation_time,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            camera_name=frame.camera_name,
            image_resolution=(frame.width, frame.height),
        )

    def estimate(self) -> TaskStateEstimate:
        start = time.perf_counter()
        observation = self.observe()
        estimate = task_state_from_perception_frame(
            observation,
            source=self.source,
            minimum_confidence=self.detector.config.minimum_confidence,
        )
        return replace(
            estimate,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def close(self) -> None:
        self.camera.close()
