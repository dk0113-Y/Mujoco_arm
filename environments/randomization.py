from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvConfig, PositionConfig
from .workspace import Workspace


@dataclass(frozen=True)
class EpisodeParameters:
    seed: int | None
    pick_region: str
    place_region: str
    pick_position: tuple[float, float, float]
    place_position: tuple[float, float, float]
    mass: float
    friction: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "pick_region": self.pick_region,
            "place_region": self.place_region,
            "pick_position": list(self.pick_position),
            "place_position": list(self.place_position),
            "mass": self.mass,
            "friction": list(self.friction),
        }


def _sample_position(
    rng: np.random.Generator,
    workspace: Workspace,
    config: PositionConfig,
    *,
    is_pick: bool,
    max_attempts: int,
) -> tuple[np.ndarray, str]:
    if config.mode == "fixed":
        position = np.asarray(config.fixed_position, dtype=float)
        region = workspace.locate_region(
            position[:2], config.allowed_regions, config.edge_margin
        )
        if region is None:
            raise ValueError(
                f"Fixed {'pick' if is_pick else 'place'} position is outside its "
                f"allowed table regions: {position.tolist()}"
            )
        expected_z = (
            workspace.object_spawn_z(region) if is_pick else workspace.target_z(region)
        )
        if not np.isclose(position[2], expected_z, atol=1e-6):
            raise ValueError(
                f"Fixed {'pick' if is_pick else 'place'} z={position[2]} does not "
                f"match the {region} table-derived z={expected_z}"
            )
        if not workspace.is_clear_of_base(position[:2]):
            raise ValueError("Fixed position overlaps the Panda base clearance")
        return position, region

    for _ in range(max_attempts):
        region_name = str(rng.choice(config.allowed_regions))
        region = workspace.region(region_name)
        min_x, max_x, min_y, max_y = region.bounds(config.edge_margin)
        xy = np.array(
            [rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)], dtype=float
        )
        if not workspace.is_clear_of_base(xy):
            continue
        z = (
            workspace.object_spawn_z(region_name)
            if is_pick
            else workspace.target_z(region_name)
        )
        return np.array([xy[0], xy[1], z], dtype=float), region_name
    raise RuntimeError(
        f"Unable to sample a valid {'pick' if is_pick else 'place'} position "
        f"within {max_attempts} attempts"
    )


def sample_episode_parameters(
    rng: np.random.Generator,
    config: EnvConfig,
    workspace: Workspace,
    *,
    seed: int | None,
) -> EpisodeParameters:
    attempts = config.workspace.max_sampling_attempts
    pick_position, pick_region = _sample_position(
        rng, workspace, config.pick, is_pick=True, max_attempts=attempts
    )

    last_distance = float("nan")
    for _ in range(attempts):
        place_position, place_region = _sample_position(
            rng, workspace, config.place, is_pick=False, max_attempts=attempts
        )
        last_distance = float(np.linalg.norm(place_position[:2] - pick_position[:2]))
        if last_distance >= config.place.minimum_xy_distance:
            break
        if config.place.mode == "fixed":
            raise ValueError(
                "Fixed pick/place positions violate place.minimum_xy_distance: "
                f"{last_distance:.6f} < {config.place.minimum_xy_distance:.6f}"
            )
    else:
        raise RuntimeError(
            "Unable to sample a place position satisfying minimum XY distance "
            f"within {attempts} attempts (last distance={last_distance:.6f})"
        )

    if config.physics.mode == "fixed":
        mass = config.physics.fixed_mass
        friction = config.physics.fixed_friction
    else:
        mass = float(rng.uniform(*config.physics.mass_range))
        friction_array = rng.uniform(
            np.asarray(config.physics.friction_min),
            np.asarray(config.physics.friction_max),
        )
        friction = tuple(float(value) for value in friction_array)

    return EpisodeParameters(
        seed=seed,
        pick_region=pick_region,
        place_region=place_region,
        pick_position=tuple(float(value) for value in pick_position),
        place_position=tuple(float(value) for value in place_position),
        mass=float(mass),
        friction=tuple(float(value) for value in friction),
    )
