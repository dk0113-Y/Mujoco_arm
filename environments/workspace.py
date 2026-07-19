from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


TABLE_GEOM_NAMES = {
    "front": "u_table_front_geom",
    "left": "u_table_left_geom",
    "right": "u_table_right_geom",
}
TABLE_BODY_NAMES = {
    "front": "u_table_front",
    "left": "u_table_left",
    "right": "u_table_right",
}


@dataclass(frozen=True)
class TableRegion:
    name: str
    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]
    top_z: float

    def bounds(self, edge_margin: float) -> tuple[float, float, float, float]:
        min_x = self.center_xy[0] - self.half_extents_xy[0] + edge_margin
        max_x = self.center_xy[0] + self.half_extents_xy[0] - edge_margin
        min_y = self.center_xy[1] - self.half_extents_xy[1] + edge_margin
        max_y = self.center_xy[1] + self.half_extents_xy[1] - edge_margin
        if min_x > max_x or min_y > max_y:
            raise ValueError(
                f"edge_margin={edge_margin} leaves no sample area on {self.name} table"
            )
        return min_x, max_x, min_y, max_y

    def contains_xy(self, xy: np.ndarray, edge_margin: float) -> bool:
        min_x, max_x, min_y, max_y = self.bounds(edge_margin)
        return bool(min_x <= xy[0] <= max_x and min_y <= xy[1] <= max_y)


@dataclass(frozen=True)
class Workspace:
    regions: dict[str, TableRegion]
    base_clearance_radius: float
    object_half_size: float
    spawn_clearance: float
    target_site_offset: float

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        base_clearance_radius: float,
        object_half_size: float,
        spawn_clearance: float,
        target_site_offset: float,
    ) -> "Workspace":
        mujoco.mj_forward(model, data)
        regions: dict[str, TableRegion] = {}
        top_heights: list[float] = []
        for region_name, geom_name in TABLE_GEOM_NAMES.items():
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                raise RuntimeError(f"Missing U-table geom: {geom_name}")
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            if not np.allclose(rotation, np.eye(3), atol=1e-9):
                raise RuntimeError(f"Sampling requires axis-aligned table geom: {geom_name}")
            center = data.geom_xpos[geom_id]
            half_size = model.geom_size[geom_id]
            top_z = float(center[2] + half_size[2])
            top_heights.append(top_z)
            regions[region_name] = TableRegion(
                name=region_name,
                center_xy=(float(center[0]), float(center[1])),
                half_extents_xy=(float(half_size[0]), float(half_size[1])),
                top_z=top_z,
            )
        if not np.allclose(top_heights, top_heights[0], atol=1e-9):
            raise RuntimeError(f"U-table top heights do not match: {top_heights}")
        return cls(
            regions=regions,
            base_clearance_radius=base_clearance_radius,
            object_half_size=object_half_size,
            spawn_clearance=spawn_clearance,
            target_site_offset=target_site_offset,
        )

    @property
    def table_top_z(self) -> float:
        return next(iter(self.regions.values())).top_z

    def region(self, name: str) -> TableRegion:
        try:
            return self.regions[name]
        except KeyError as exc:
            raise ValueError(f"Unknown table region: {name}") from exc

    def is_clear_of_base(self, xy: np.ndarray) -> bool:
        required = self.base_clearance_radius + self.object_half_size
        return bool(np.linalg.norm(np.asarray(xy, dtype=float)) >= required)

    def locate_region(
        self,
        xy: np.ndarray,
        allowed_regions: tuple[str, ...],
        edge_margin: float,
    ) -> str | None:
        for name in allowed_regions:
            if self.region(name).contains_xy(xy, edge_margin):
                return name
        return None

    def object_spawn_z(self, region_name: str) -> float:
        return (
            self.region(region_name).top_z
            + self.object_half_size
            + self.spawn_clearance
        )

    def target_z(self, region_name: str) -> float:
        return self.region(region_name).top_z + self.target_site_offset
