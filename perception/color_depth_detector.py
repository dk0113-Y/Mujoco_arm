from __future__ import annotations

from collections import deque

import numpy as np

from environments.config import PerceptionConfig

from .camera_geometry import ProjectionError, pixels_depth_to_world
from .types import DetectionResult, RGBDFrame


class ColorDepthDetector:
    """Detect the red cube and green target using RGB and metric depth only."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.config = config

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best: list[tuple[int, int]] = []
        for start_v, start_u in np.argwhere(mask):
            start = (int(start_v), int(start_u))
            if visited[start]:
                continue
            visited[start] = True
            queue: deque[tuple[int, int]] = deque([start])
            component: list[tuple[int, int]] = []
            while queue:
                v, u = queue.popleft()
                component.append((v, u))
                for next_v, next_u in (
                    (v - 1, u),
                    (v + 1, u),
                    (v, u - 1),
                    (v, u + 1),
                ):
                    if (
                        0 <= next_v < height
                        and 0 <= next_u < width
                        and mask[next_v, next_u]
                        and not visited[next_v, next_u]
                    ):
                        visited[next_v, next_u] = True
                        queue.append((next_v, next_u))
            if len(component) > len(best):
                best = component
        result = np.zeros_like(mask, dtype=bool)
        if best:
            coordinates = np.asarray(best, dtype=int)
            result[coordinates[:, 0], coordinates[:, 1]] = True
        return result

    def _color_mask(self, rgb: np.ndarray, kind: str) -> np.ndarray:
        colors = rgb.astype(np.float32)
        red, green, blue = colors[..., 0], colors[..., 1], colors[..., 2]
        if kind == "object":
            minimum = np.asarray(self.config.object_min_rgb, dtype=float)
            ratio = self.config.object_dominance_ratio
            return (
                (red >= minimum[0])
                & (green >= minimum[1])
                & (blue >= minimum[2])
                & (red >= ratio * green)
                & (red >= ratio * blue)
            )
        minimum = np.asarray(self.config.target_min_rgb, dtype=float)
        ratio = self.config.target_dominance_ratio
        return (
            (red >= minimum[0])
            & (green >= minimum[1])
            & (blue >= minimum[2])
            & (green >= ratio * red)
            & (green >= ratio * blue)
        )

    def _failure(
        self,
        detection_id: str,
        mask: np.ndarray,
        reason: str,
    ) -> DetectionResult:
        return DetectionResult(
            detection_id=detection_id,
            success=False,
            mask=mask,
            pixel_count=int(np.count_nonzero(mask)),
            center_pixel=None,
            position=None,
            confidence=0.0,
            failure_reason=reason,
        )

    def _detect(self, frame: RGBDFrame, kind: str) -> DetectionResult:
        detection_id = "pick_object_0" if kind == "object" else "place_target_0"
        not_found_reason = (
            "perception_object_not_found"
            if kind == "object"
            else "perception_target_not_found"
        )
        minimum_pixels = (
            self.config.minimum_object_pixels
            if kind == "object"
            else self.config.minimum_target_pixels
        )
        z_range = (
            self.config.object_world_z_range
            if kind == "object"
            else self.config.target_world_z_range
        )
        surface_correction = (
            self.config.object_surface_to_center
            if kind == "object"
            else self.config.target_surface_to_center
        )

        color_mask = self._color_mask(frame.rgb, kind)
        if not np.any(color_mask):
            return self._failure(detection_id, color_mask, not_found_reason)
        valid_depth = (
            np.isfinite(frame.depth)
            & (frame.depth >= self.config.minimum_depth)
            & (frame.depth <= self.config.maximum_depth)
        )
        color_and_depth = color_mask & valid_depth
        if not np.any(color_and_depth):
            return self._failure(
                detection_id, color_and_depth, "perception_invalid_depth"
            )

        v_all, u_all = np.nonzero(color_and_depth)
        try:
            world_all = pixels_depth_to_world(
                u_all,
                v_all,
                frame.depth[v_all, u_all],
                frame.intrinsics,
                frame.extrinsics,
            )
        except ProjectionError:
            return self._failure(
                detection_id, color_and_depth, "perception_projection_error"
            )
        height_valid = (world_all[:, 2] >= z_range[0]) & (
            world_all[:, 2] <= z_range[1]
        )
        geometry_mask = np.zeros_like(color_mask, dtype=bool)
        geometry_mask[v_all[height_valid], u_all[height_valid]] = True
        component_mask = self._largest_component(geometry_mask)
        pixel_count = int(np.count_nonzero(component_mask))
        if pixel_count < minimum_pixels:
            return self._failure(detection_id, component_mask, not_found_reason)

        v_pixels, u_pixels = np.nonzero(component_mask)
        try:
            world_points = pixels_depth_to_world(
                u_pixels,
                v_pixels,
                frame.depth[v_pixels, u_pixels],
                frame.intrinsics,
                frame.extrinsics,
            )
        except ProjectionError:
            return self._failure(
                detection_id, component_mask, "perception_projection_error"
            )
        robust_surface = np.median(world_points, axis=0)
        robust_surface[2] -= surface_correction
        center_pixel = (
            float(np.median(u_pixels)),
            float(np.median(v_pixels)),
        )
        depth_median = float(np.median(frame.depth[v_pixels, u_pixels]))
        depth_mad = float(
            np.median(np.abs(frame.depth[v_pixels, u_pixels] - depth_median))
        )
        count_score = min(1.0, pixel_count / float(2 * minimum_pixels))
        consistency_score = max(0.0, 1.0 - depth_mad / 0.02)
        confidence = float(count_score * consistency_score)
        if confidence < self.config.minimum_confidence:
            return DetectionResult(
                detection_id=detection_id,
                success=False,
                mask=component_mask,
                pixel_count=pixel_count,
                center_pixel=center_pixel,
                position=tuple(float(value) for value in robust_surface),
                confidence=confidence,
                failure_reason="perception_low_confidence",
            )
        return DetectionResult(
            detection_id=detection_id,
            success=True,
            mask=component_mask,
            pixel_count=pixel_count,
            center_pixel=center_pixel,
            position=tuple(float(value) for value in robust_surface),
            confidence=confidence,
            failure_reason=None,
        )

    def detect_object(self, frame: RGBDFrame) -> DetectionResult:
        return self._detect(frame, "object")

    def detect_target(self, frame: RGBDFrame) -> DetectionResult:
        return self._detect(frame, "target")
