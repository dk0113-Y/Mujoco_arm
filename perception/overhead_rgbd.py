from __future__ import annotations

import mujoco
import numpy as np

from environments.config import CameraConfig

from .camera_geometry import extrinsics_from_mujoco, intrinsics_from_fovy
from .types import RGBDFrame


class OverheadRGBDCamera:
    """Own one MuJoCo offscreen Renderer for the fixed overhead camera.

    MuJoCo 3.10's ``Renderer.render`` converts the OpenGL buffer to metric
    axial depth: positive distance along camera local -Z, not Euclidean range.
    """

    def __init__(self, model: mujoco.MjModel, config: CameraConfig) -> None:
        self.model = model
        self.config = config
        self.camera_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, config.name)
        )
        if self.camera_id < 0:
            raise RuntimeError(f"MuJoCo model is missing camera: {config.name}")
        if config.width > model.vis.global_.offwidth or (
            config.height > model.vis.global_.offheight
        ):
            raise ValueError(
                "Camera resolution exceeds MuJoCo offscreen framebuffer: "
                f"camera={config.width}x{config.height}, "
                f"buffer={model.vis.global_.offwidth}x{model.vis.global_.offheight}"
            )
        self._renderer = mujoco.Renderer(
            model, height=config.height, width=config.width
        )
        self._closed = False

    def capture(self, data: mujoco.MjData) -> RGBDFrame:
        if self._closed:
            raise RuntimeError("Cannot capture with a closed OverheadRGBDCamera")
        self._renderer.update_scene(data, camera=self.camera_id)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        try:
            depth = self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()

        expected_rgb_shape = (self.config.height, self.config.width, 3)
        expected_depth_shape = (self.config.height, self.config.width)
        if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
            raise RuntimeError(
                f"Unexpected RGB output: shape={rgb.shape}, dtype={rgb.dtype}"
            )
        if depth.shape != expected_depth_shape or not np.issubdtype(
            depth.dtype, np.floating
        ):
            raise RuntimeError(
                f"Unexpected depth output: shape={depth.shape}, dtype={depth.dtype}"
            )
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid_depth):
            raise RuntimeError("Depth image contains no finite positive samples")

        intrinsics = intrinsics_from_fovy(
            width=self.config.width,
            height=self.config.height,
            fovy_degrees=float(self.model.cam_fovy[self.camera_id]),
        )
        extrinsics = extrinsics_from_mujoco(self.model, data, self.camera_id)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            simulation_time=float(data.time),
            camera_name=self.config.name,
            width=self.config.width,
            height=self.config.height,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    def close(self) -> None:
        if not self._closed:
            self._renderer.close()
            self._closed = True

    def __enter__(self) -> "OverheadRGBDCamera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
