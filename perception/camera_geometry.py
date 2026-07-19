from __future__ import annotations

import math

import mujoco
import numpy as np

from .types import CameraExtrinsics, CameraIntrinsics


class ProjectionError(ValueError):
    pass


def intrinsics_from_fovy(
    *, width: int, height: int, fovy_degrees: float
) -> CameraIntrinsics:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 0.0 < fovy_degrees < 180.0:
        raise ValueError("Vertical field of view must be between 0 and 180 degrees")
    focal = 0.5 * height / math.tan(math.radians(fovy_degrees) / 2.0)
    return CameraIntrinsics(
        fx=float(focal),
        fy=float(focal),
        cx=0.5 * (width - 1),
        cy=0.5 * (height - 1),
        width=width,
        height=height,
    )


def extrinsics_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
) -> CameraExtrinsics:
    if camera_id < 0 or camera_id >= model.ncam:
        raise ValueError(f"Invalid MuJoCo camera ID: {camera_id}")
    position = np.asarray(data.cam_xpos[camera_id], dtype=float).copy()
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy()
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ProjectionError("Camera extrinsics contain NaN or Inf")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ProjectionError("MuJoCo camera rotation is not orthonormal")
    camera_to_world = np.eye(4, dtype=float)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = position
    world_to_camera = np.eye(4, dtype=float)
    world_to_camera[:3, :3] = rotation.T
    world_to_camera[:3, 3] = -rotation.T @ position
    return CameraExtrinsics(
        position_world=position,
        rotation_world_from_camera=rotation,
        camera_to_world=camera_to_world,
        world_to_camera=world_to_camera,
    )


def pixel_depth_to_camera(
    pixel: tuple[float, float],
    depth: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    u, v = (float(pixel[0]), float(pixel[1]))
    depth = float(depth)
    if not math.isfinite(u) or not math.isfinite(v):
        raise ProjectionError("Pixel coordinates must be finite")
    if not (0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height):
        raise ProjectionError(f"Pixel is outside the image: {(u, v)}")
    if not math.isfinite(depth) or depth <= 0.0:
        raise ProjectionError(f"Depth must be finite and positive, got {depth}")
    # Image v grows downward while MuJoCo camera +Y points image-up.
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = -(v - intrinsics.cy) * depth / intrinsics.fy
    # MuJoCo cameras observe along local -Z; Renderer depth is axial -Z depth.
    return np.array([x, y, -depth], dtype=float)


def camera_to_world(
    camera_point: np.ndarray,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    point = np.asarray(camera_point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ProjectionError("Camera point must be a finite 3-vector")
    return extrinsics.rotation_world_from_camera @ point + extrinsics.position_world


def world_to_camera(
    world_point: np.ndarray,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    point = np.asarray(world_point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ProjectionError("World point must be a finite 3-vector")
    return extrinsics.rotation_world_from_camera.T @ (
        point - extrinsics.position_world
    )


def camera_to_pixel(
    camera_point: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    require_inside: bool = True,
) -> tuple[float, float, float]:
    point = np.asarray(camera_point, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ProjectionError("Camera point must be a finite 3-vector")
    depth = -float(point[2])
    if depth <= 0.0:
        raise ProjectionError("Point is on or behind the camera plane")
    u = intrinsics.cx + intrinsics.fx * float(point[0]) / depth
    v = intrinsics.cy - intrinsics.fy * float(point[1]) / depth
    if require_inside and not (
        0.0 <= u < intrinsics.width and 0.0 <= v < intrinsics.height
    ):
        raise ProjectionError(f"Projected pixel is outside the image: {(u, v)}")
    return u, v, depth


def world_to_pixel(
    world_point: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    *,
    require_inside: bool = True,
) -> tuple[float, float, float]:
    return camera_to_pixel(
        world_to_camera(world_point, extrinsics),
        intrinsics,
        require_inside=require_inside,
    )


def pixel_depth_to_world(
    pixel: tuple[float, float],
    depth: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    return camera_to_world(
        pixel_depth_to_camera(pixel, depth, intrinsics), extrinsics
    )


def pixels_depth_to_world(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    u_array = np.asarray(u, dtype=float)
    v_array = np.asarray(v, dtype=float)
    depth_array = np.asarray(depth, dtype=float)
    if not (u_array.shape == v_array.shape == depth_array.shape):
        raise ProjectionError("Pixel and depth arrays must have matching shapes")
    if np.any(~np.isfinite(depth_array)) or np.any(depth_array <= 0.0):
        raise ProjectionError("Depth array contains invalid values")
    if np.any(u_array < 0.0) or np.any(u_array >= intrinsics.width):
        raise ProjectionError("Pixel u array contains out-of-image coordinates")
    if np.any(v_array < 0.0) or np.any(v_array >= intrinsics.height):
        raise ProjectionError("Pixel v array contains out-of-image coordinates")
    camera_points = np.column_stack(
        (
            (u_array - intrinsics.cx) * depth_array / intrinsics.fx,
            -(v_array - intrinsics.cy) * depth_array / intrinsics.fy,
            -depth_array,
        )
    )
    return (
        camera_points @ extrinsics.rotation_world_from_camera.T
        + extrinsics.position_world
    )
