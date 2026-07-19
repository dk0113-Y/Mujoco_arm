from __future__ import annotations

from typing import Protocol
import time

import mujoco
import numpy as np

from environments.panda_u_table_env import PandaUTableEnv

from .color_depth_detector import ColorDepthDetector
from .overhead_rgbd import OverheadRGBDCamera
from .types import DetectionResult, RGBDFrame, TaskStateEstimate


class TaskStateProvider(Protocol):
    source: str

    def estimate(self) -> TaskStateEstimate:
        ...

    def close(self) -> None:
        ...


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

    def estimate(self) -> TaskStateEstimate:
        start = time.perf_counter()
        frame = self.camera.capture(self.data)
        object_detection = self.detector.detect_object(frame)
        target_detection = self.detector.detect_target(frame)
        self.last_frame = frame
        self.last_object_detection = object_detection
        self.last_target_detection = target_detection

        failure_reason: str | None = None
        if not object_detection.success:
            failure_reason = object_detection.failure_reason
        elif not target_detection.success:
            failure_reason = target_detection.failure_reason
        confidence = min(object_detection.confidence, target_detection.confidence)
        if failure_reason is None and confidence < self.detector.config.minimum_confidence:
            failure_reason = "perception_low_confidence"
        valid = bool(
            failure_reason is None
            and object_detection.position is not None
            and target_detection.position is not None
            and np.all(np.isfinite(object_detection.position))
            and np.all(np.isfinite(target_detection.position))
        )
        return TaskStateEstimate(
            object_id="pick_object_0",
            target_id="place_target_0",
            object_position=object_detection.position,
            target_position=target_detection.position,
            timestamp=frame.simulation_time,
            source=self.source,
            valid=valid,
            confidence=float(confidence),
            failure_reason=failure_reason,
            object_pixel_count=object_detection.pixel_count,
            target_pixel_count=target_detection.pixel_count,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            camera_name=frame.camera_name,
            image_resolution=(frame.width, frame.height),
        )

    def close(self) -> None:
        self.camera.close()
