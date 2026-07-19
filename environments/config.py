from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping


REGION_NAMES = frozenset({"front", "left", "right"})
MODES = frozenset({"fixed", "random"})
OBSERVATION_SOURCES = frozenset({"privileged", "perception"})


@dataclass(frozen=True)
class WorkspaceConfig:
    object_half_size: float
    base_clearance_radius: float
    spawn_clearance: float
    target_site_offset: float
    max_sampling_attempts: int


@dataclass(frozen=True)
class PositionConfig:
    mode: str
    fixed_position: tuple[float, float, float]
    allowed_regions: tuple[str, ...]
    edge_margin: float


@dataclass(frozen=True)
class PlaceConfig(PositionConfig):
    minimum_xy_distance: float


@dataclass(frozen=True)
class PhysicsConfig:
    mode: str
    fixed_mass: float
    mass_range: tuple[float, float]
    fixed_friction: tuple[float, float, float]
    friction_min: tuple[float, float, float]
    friction_max: tuple[float, float, float]


@dataclass(frozen=True)
class SimulationConfig:
    settle_time: float
    episode_timeout: float
    frame_skip: int
    viewer: bool


@dataclass(frozen=True)
class ObservationConfig:
    source: str


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int
    position: tuple[float, float, float]
    x_axis_world: tuple[float, float, float]
    y_axis_world: tuple[float, float, float]
    fovy: float


@dataclass(frozen=True)
class PerceptionConfig:
    minimum_object_pixels: int
    minimum_target_pixels: int
    minimum_confidence: float
    minimum_depth: float
    maximum_depth: float
    object_min_rgb: tuple[float, float, float]
    object_dominance_ratio: float
    target_min_rgb: tuple[float, float, float]
    target_dominance_ratio: float
    object_world_z_range: tuple[float, float]
    target_world_z_range: tuple[float, float]
    object_surface_to_center: float
    target_surface_to_center: float


@dataclass(frozen=True)
class ControllerConfig:
    ik_max_iterations: int
    ik_damping: float
    ik_step_gain: float
    ik_max_joint_step: float
    ik_position_tolerance: float
    orientation_tolerance: float
    orientation_weight: float
    waypoint_tolerance: float
    waypoint_height: float
    grasp_z_offset: float
    lift_height: float
    minimum_lift_height: float
    place_xy_tolerance: float
    place_height_tolerance: float
    approach_duration: float
    descent_duration: float
    gripper_duration: float
    lift_duration: float
    transfer_duration: float
    withdraw_duration: float
    motion_hold_time: float
    gripper_open_control: float
    gripper_close_control: float


@dataclass(frozen=True)
class EnvConfig:
    seed: int
    workspace: WorkspaceConfig
    pick: PositionConfig
    place: PlaceConfig
    physics: PhysicsConfig
    simulation: SimulationConfig
    observation: ObservationConfig
    camera: CameraConfig
    perception: PerceptionConfig
    controller: ControllerConfig

    def with_modes(
        self,
        *,
        pick_mode: str | None = None,
        place_mode: str | None = None,
        physics_mode: str | None = None,
        seed: int | None = None,
        viewer: bool | None = None,
        observation_source: str | None = None,
    ) -> "EnvConfig":
        updated = replace(
            self,
            seed=self.seed if seed is None else seed,
            pick=replace(
                self.pick,
                mode=self.pick.mode if pick_mode is None else pick_mode,
            ),
            place=replace(
                self.place,
                mode=self.place.mode if place_mode is None else place_mode,
            ),
            physics=replace(
                self.physics,
                mode=self.physics.mode if physics_mode is None else physics_mode,
            ),
            simulation=replace(
                self.simulation,
                viewer=self.simulation.viewer if viewer is None else viewer,
            ),
            observation=replace(
                self.observation,
                source=(
                    self.observation.source
                    if observation_source is None
                    else observation_source
                ),
            ),
        )
        validate_config(updated)
        return updated


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid [{name}] section")
    return value


def _tuple_of_floats(
    value: Any,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field_name} must contain exactly {length} numbers")
    try:
        converted = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numbers") from exc
    return converted


def _position(section: Mapping[str, Any], name: str) -> PositionConfig:
    regions = section.get("allowed_regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"{name}.allowed_regions must be a non-empty list")
    return PositionConfig(
        mode=str(section.get("mode", "")),
        fixed_position=_tuple_of_floats(
            section.get("fixed_position"), 3, f"{name}.fixed_position"
        ),
        allowed_regions=tuple(str(region) for region in regions),
        edge_margin=float(section.get("edge_margin", -1.0)),
    )


def load_config(path: str | Path) -> EnvConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    environment = _section(raw, "environment")
    workspace = _section(raw, "workspace")
    pick = _position(_section(raw, "pick"), "pick")
    place_raw = _section(raw, "place")
    place_base = _position(place_raw, "place")
    physics = _section(raw, "physics")
    simulation = _section(raw, "simulation")
    observation = _section(raw, "observation")
    camera = _section(raw, "camera")
    perception = _section(raw, "perception")
    controller = _section(raw, "controller")

    config = EnvConfig(
        seed=int(environment.get("seed", 0)),
        workspace=WorkspaceConfig(
            object_half_size=float(workspace.get("object_half_size", -1.0)),
            base_clearance_radius=float(
                workspace.get("base_clearance_radius", -1.0)
            ),
            spawn_clearance=float(workspace.get("spawn_clearance", -1.0)),
            target_site_offset=float(workspace.get("target_site_offset", -1.0)),
            max_sampling_attempts=int(workspace.get("max_sampling_attempts", 0)),
        ),
        pick=pick,
        place=PlaceConfig(
            **place_base.__dict__,
            minimum_xy_distance=float(place_raw.get("minimum_xy_distance", -1.0)),
        ),
        physics=PhysicsConfig(
            mode=str(physics.get("mode", "")),
            fixed_mass=float(physics.get("fixed_mass", -1.0)),
            mass_range=_tuple_of_floats(
                physics.get("mass_range"), 2, "physics.mass_range"
            ),
            fixed_friction=_tuple_of_floats(
                physics.get("fixed_friction"), 3, "physics.fixed_friction"
            ),
            friction_min=_tuple_of_floats(
                physics.get("friction_min"), 3, "physics.friction_min"
            ),
            friction_max=_tuple_of_floats(
                physics.get("friction_max"), 3, "physics.friction_max"
            ),
        ),
        simulation=SimulationConfig(
            settle_time=float(simulation.get("settle_time", -1.0)),
            episode_timeout=float(simulation.get("episode_timeout", -1.0)),
            frame_skip=int(simulation.get("frame_skip", 0)),
            viewer=bool(simulation.get("viewer", False)),
        ),
        observation=ObservationConfig(source=str(observation.get("source", ""))),
        camera=CameraConfig(
            name=str(camera.get("name", "")),
            width=int(camera.get("width", 0)),
            height=int(camera.get("height", 0)),
            position=_tuple_of_floats(camera.get("position"), 3, "camera.position"),
            x_axis_world=_tuple_of_floats(
                camera.get("x_axis_world"), 3, "camera.x_axis_world"
            ),
            y_axis_world=_tuple_of_floats(
                camera.get("y_axis_world"), 3, "camera.y_axis_world"
            ),
            fovy=float(camera.get("fovy", -1.0)),
        ),
        perception=PerceptionConfig(
            minimum_object_pixels=int(
                perception.get("minimum_object_pixels", 0)
            ),
            minimum_target_pixels=int(
                perception.get("minimum_target_pixels", 0)
            ),
            minimum_confidence=float(perception.get("minimum_confidence", -1.0)),
            minimum_depth=float(perception.get("minimum_depth", -1.0)),
            maximum_depth=float(perception.get("maximum_depth", -1.0)),
            object_min_rgb=_tuple_of_floats(
                perception.get("object_min_rgb"), 3, "perception.object_min_rgb"
            ),
            object_dominance_ratio=float(
                perception.get("object_dominance_ratio", -1.0)
            ),
            target_min_rgb=_tuple_of_floats(
                perception.get("target_min_rgb"), 3, "perception.target_min_rgb"
            ),
            target_dominance_ratio=float(
                perception.get("target_dominance_ratio", -1.0)
            ),
            object_world_z_range=_tuple_of_floats(
                perception.get("object_world_z_range"),
                2,
                "perception.object_world_z_range",
            ),
            target_world_z_range=_tuple_of_floats(
                perception.get("target_world_z_range"),
                2,
                "perception.target_world_z_range",
            ),
            object_surface_to_center=float(
                perception.get("object_surface_to_center", -1.0)
            ),
            target_surface_to_center=float(
                perception.get("target_surface_to_center", -1.0)
            ),
        ),
        controller=ControllerConfig(
            **{
                "ik_max_iterations": int(controller.get("ik_max_iterations", 0)),
                "ik_damping": float(controller.get("ik_damping", -1.0)),
                "ik_step_gain": float(controller.get("ik_step_gain", -1.0)),
                "ik_max_joint_step": float(
                    controller.get("ik_max_joint_step", -1.0)
                ),
                "ik_position_tolerance": float(
                    controller.get("ik_position_tolerance", -1.0)
                ),
                "orientation_tolerance": float(
                    controller.get("orientation_tolerance", -1.0)
                ),
                "orientation_weight": float(
                    controller.get("orientation_weight", -1.0)
                ),
                "waypoint_tolerance": float(
                    controller.get("waypoint_tolerance", -1.0)
                ),
                "waypoint_height": float(controller.get("waypoint_height", -1.0)),
                "grasp_z_offset": float(controller.get("grasp_z_offset", -1.0)),
                "lift_height": float(controller.get("lift_height", -1.0)),
                "minimum_lift_height": float(
                    controller.get("minimum_lift_height", -1.0)
                ),
                "place_xy_tolerance": float(
                    controller.get("place_xy_tolerance", -1.0)
                ),
                "place_height_tolerance": float(
                    controller.get("place_height_tolerance", -1.0)
                ),
                "approach_duration": float(
                    controller.get("approach_duration", -1.0)
                ),
                "descent_duration": float(controller.get("descent_duration", -1.0)),
                "gripper_duration": float(controller.get("gripper_duration", -1.0)),
                "lift_duration": float(controller.get("lift_duration", -1.0)),
                "transfer_duration": float(
                    controller.get("transfer_duration", -1.0)
                ),
                "withdraw_duration": float(
                    controller.get("withdraw_duration", -1.0)
                ),
                "motion_hold_time": float(
                    controller.get("motion_hold_time", -1.0)
                ),
                "gripper_open_control": float(
                    controller.get("gripper_open_control", -1.0)
                ),
                "gripper_close_control": float(
                    controller.get("gripper_close_control", -1.0)
                ),
            }
        ),
    )
    validate_config(config)
    return config


def validate_config(config: EnvConfig) -> None:
    for name, position in (("pick", config.pick), ("place", config.place)):
        if position.mode not in MODES:
            raise ValueError(f"{name}.mode must be 'fixed' or 'random', got {position.mode!r}")
        unknown = set(position.allowed_regions) - REGION_NAMES
        if unknown:
            raise ValueError(f"{name}.allowed_regions contains unknown regions: {sorted(unknown)}")
        if position.edge_margin < config.workspace.object_half_size:
            raise ValueError(
                f"{name}.edge_margin must be at least object_half_size "
                f"({config.workspace.object_half_size})"
            )

    workspace = config.workspace
    if workspace.object_half_size <= 0.0:
        raise ValueError("workspace.object_half_size must be positive")
    if workspace.base_clearance_radius <= 0.0:
        raise ValueError("workspace.base_clearance_radius must be positive")
    if workspace.spawn_clearance < 0.0 or workspace.target_site_offset < 0.0:
        raise ValueError("workspace clearances must be non-negative")
    if workspace.max_sampling_attempts <= 0:
        raise ValueError("workspace.max_sampling_attempts must be positive")
    if config.place.minimum_xy_distance <= 2.0 * workspace.object_half_size:
        raise ValueError(
            "place.minimum_xy_distance must exceed the object's full XY size"
        )

    physics = config.physics
    if physics.mode not in MODES:
        raise ValueError(
            f"physics.mode must be 'fixed' or 'random', got {physics.mode!r}"
        )
    if physics.fixed_mass <= 0.0 or physics.mass_range[0] <= 0.0:
        raise ValueError("Object mass values must be positive")
    if physics.mass_range[0] > physics.mass_range[1]:
        raise ValueError("physics.mass_range lower bound exceeds upper bound")
    if any(value <= 0.0 for value in physics.fixed_friction):
        raise ValueError("physics.fixed_friction values must be positive")
    if any(value <= 0.0 for value in physics.friction_min):
        raise ValueError("physics.friction_min values must be positive")
    if any(low > high for low, high in zip(physics.friction_min, physics.friction_max)):
        raise ValueError("physics friction range lower bound exceeds upper bound")

    simulation = config.simulation
    if simulation.settle_time < 0.0 or simulation.episode_timeout <= 0.0:
        raise ValueError("Simulation times must be non-negative, with a positive timeout")
    if simulation.frame_skip <= 0:
        raise ValueError("simulation.frame_skip must be positive")

    if config.observation.source not in OBSERVATION_SOURCES:
        raise ValueError(
            "observation.source must be 'privileged' or 'perception', got "
            f"{config.observation.source!r}"
        )

    camera = config.camera
    if camera.name != "overhead_rgbd":
        raise ValueError("camera.name must be 'overhead_rgbd'")
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError("camera width and height must be positive")
    if not 0.0 < camera.fovy < 180.0:
        raise ValueError("camera.fovy must be between 0 and 180 degrees")
    vectors = (camera.position, camera.x_axis_world, camera.y_axis_world)
    if not all(math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("camera position and axes must contain finite values")
    x_norm = math.sqrt(sum(value * value for value in camera.x_axis_world))
    y_norm = math.sqrt(sum(value * value for value in camera.y_axis_world))
    dot = sum(
        x_value * y_value
        for x_value, y_value in zip(camera.x_axis_world, camera.y_axis_world)
    )
    if not math.isclose(x_norm, 1.0, abs_tol=1e-6):
        raise ValueError("camera.x_axis_world must be a unit vector")
    if not math.isclose(y_norm, 1.0, abs_tol=1e-6):
        raise ValueError("camera.y_axis_world must be a unit vector")
    if not math.isclose(dot, 0.0, abs_tol=1e-6):
        raise ValueError("camera x/y axes must be orthogonal")
    z_axis = (
        camera.x_axis_world[1] * camera.y_axis_world[2]
        - camera.x_axis_world[2] * camera.y_axis_world[1],
        camera.x_axis_world[2] * camera.y_axis_world[0]
        - camera.x_axis_world[0] * camera.y_axis_world[2],
        camera.x_axis_world[0] * camera.y_axis_world[1]
        - camera.x_axis_world[1] * camera.y_axis_world[0],
    )
    if z_axis[2] < 0.9:
        raise ValueError("overhead camera local -Z axis must point mostly downward")

    perception = config.perception
    if perception.minimum_object_pixels <= 0 or perception.minimum_target_pixels <= 0:
        raise ValueError("perception minimum pixel counts must be positive")
    if not 0.0 <= perception.minimum_confidence <= 1.0:
        raise ValueError("perception.minimum_confidence must be in [0, 1]")
    if perception.minimum_depth <= 0.0 or (
        perception.minimum_depth >= perception.maximum_depth
    ):
        raise ValueError("perception depth range is invalid")
    for field_name, values in (
        ("object_min_rgb", perception.object_min_rgb),
        ("target_min_rgb", perception.target_min_rgb),
    ):
        if any(value < 0.0 or value > 255.0 for value in values):
            raise ValueError(f"perception.{field_name} values must be in [0, 255]")
    if (
        perception.object_dominance_ratio <= 1.0
        or perception.target_dominance_ratio <= 1.0
    ):
        raise ValueError("perception color dominance ratios must exceed 1")
    for field_name, value_range in (
        ("object_world_z_range", perception.object_world_z_range),
        ("target_world_z_range", perception.target_world_z_range),
    ):
        if value_range[0] >= value_range[1]:
            raise ValueError(f"perception.{field_name} lower bound exceeds upper bound")
    if (
        perception.object_surface_to_center < 0.0
        or perception.target_surface_to_center < 0.0
    ):
        raise ValueError("perception surface-to-center corrections must be non-negative")

    controller = config.controller
    positive_values = {
        field: getattr(controller, field)
        for field in (
            "ik_max_iterations",
            "ik_damping",
            "ik_step_gain",
            "ik_max_joint_step",
            "ik_position_tolerance",
            "orientation_tolerance",
            "orientation_weight",
            "waypoint_tolerance",
            "waypoint_height",
            "lift_height",
            "minimum_lift_height",
            "place_xy_tolerance",
            "place_height_tolerance",
            "approach_duration",
            "descent_duration",
            "gripper_duration",
            "lift_duration",
            "transfer_duration",
            "withdraw_duration",
        )
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Controller values must be positive: {', '.join(invalid)}")
    if controller.grasp_z_offset < 0.0 or controller.motion_hold_time < 0.0:
        raise ValueError("Controller offsets/hold time must be non-negative")
    if controller.gripper_open_control <= controller.gripper_close_control:
        raise ValueError("Open gripper control must exceed close gripper control")
