"""Strict configuration for the isolated CI-Baseline v1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping

from .config import DYNAMICS_MODES, PANDA_JOINT_LIMITS, PROJECT_ROOT


JOINT_COUNT = 7


@dataclass(frozen=True)
class CartesianModelConfig:
    path: Path
    tcp_site: str


@dataclass(frozen=True)
class CartesianSimulationConfig:
    control_frequency_hz: float
    substeps: int
    maximum_duration: float
    output_sampling_frequency_hz: float

    @property
    def control_period(self) -> float:
        return 1.0 / self.control_frequency_hz


@dataclass(frozen=True)
class CartesianControllerConfig:
    translational_stiffness: tuple[float, float, float]
    rotational_stiffness: tuple[float, float, float]
    translational_damping: tuple[float, float, float]
    rotational_damping: tuple[float, float, float]
    dynamics_compensation_mode: str
    torque_limits: tuple[float, ...]
    torque_rate_limits: tuple[float, ...]


@dataclass(frozen=True)
class KinematicsConfig:
    jacobian_rank_tolerance: float
    minimum_singular_value: float
    maximum_condition_number: float
    quaternion_convention: str
    orientation_error_convention: str


@dataclass(frozen=True)
class CartesianSafetyConfig:
    joint_limit_margin: float
    joint_velocity_limits: tuple[float, ...]
    maximum_tcp_position_error: float
    maximum_tcp_orientation_error: float
    sustained_violation_duration: float
    simulation_instability_acceleration: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]


@dataclass(frozen=True)
class CartesianTrajectoryConfig:
    initial_joint_pose: tuple[float, ...]
    hold_joint_poses: tuple[tuple[float, ...], ...]
    hold_duration: float
    translation_amplitudes: tuple[float, float, float]
    translation_frequency_hz: float
    translation_ramp_duration: float
    translation_duration: float
    orientation_amplitudes: tuple[float, float, float]
    orientation_frequency_hz: float
    orientation_ramp_duration: float
    orientation_duration: float
    line_displacement: tuple[float, float, float]
    line_duration: float
    circle_radius: float
    circle_plane_axes: tuple[int, int]
    circle_duration: float
    seed: int


@dataclass(frozen=True)
class CartesianBenchmarkConfig:
    source_path: Path
    source_text: str
    model: CartesianModelConfig
    simulation: CartesianSimulationConfig
    controller: CartesianControllerConfig
    kinematics: KinematicsConfig
    safety: CartesianSafetyConfig
    trajectory: CartesianTrajectoryConfig

    @property
    def seed(self) -> int:
        """Compatibility value used by the isolated torque environment."""

        return self.trajectory.seed


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid [{name}] section")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numbers")
    return tuple(_number(item, name) for item in value)


def _joint_pose_is_legal(pose: tuple[float, ...], margin: float) -> bool:
    return all(
        lower + margin < value < upper - margin
        for value, (lower, upper) in zip(pose, PANDA_JOINT_LIMITS)
    )


def validate_cartesian_config(config: CartesianBenchmarkConfig) -> None:
    expected_root = (PROJECT_ROOT / "models" / "panda_torque").resolve()
    if not config.model.path.is_file():
        raise FileNotFoundError(
            f"Torque-control model does not exist: {config.model.path}"
        )
    try:
        config.model.path.resolve().relative_to(expected_root)
    except ValueError as exc:
        raise ValueError("model.path must remain inside models/panda_torque") from exc
    if not config.model.tcp_site.strip():
        raise ValueError("model.tcp_site must be non-empty")

    simulation = config.simulation
    if simulation.control_frequency_hz <= 0.0:
        raise ValueError("simulation.control_frequency_hz must be positive")
    if simulation.substeps <= 0:
        raise ValueError("simulation.substeps must be positive")
    if simulation.maximum_duration <= 0.0:
        raise ValueError("simulation.maximum_duration must be positive")
    if not math.isclose(
        simulation.output_sampling_frequency_hz,
        simulation.control_frequency_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "CI-Baseline v1 records every control period, so output sampling "
            "frequency must equal control frequency"
        )

    controller = config.controller
    gains = (
        *controller.translational_stiffness,
        *controller.rotational_stiffness,
    )
    damping = (
        *controller.translational_damping,
        *controller.rotational_damping,
    )
    if any(value <= 0.0 for value in gains):
        raise ValueError("all Cartesian stiffness values must be positive")
    if any(value < 0.0 for value in damping):
        raise ValueError("all Cartesian damping values must be non-negative")
    if controller.dynamics_compensation_mode not in DYNAMICS_MODES:
        raise ValueError(
            "controller.dynamics_compensation_mode must be one of "
            f"{sorted(DYNAMICS_MODES)}"
        )
    if any(value <= 0.0 for value in controller.torque_limits):
        raise ValueError("controller.torque_limits values must be positive")
    if any(value <= 0.0 for value in controller.torque_rate_limits):
        raise ValueError("controller.torque_rate_limits values must be positive")
    physical_limits = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    if any(
        configured > physical + 1e-12
        for configured, physical in zip(
            controller.torque_limits, physical_limits
        )
    ):
        raise ValueError(
            "controller.torque_limits cannot exceed the MJCF actuator limits"
        )

    kinematics = config.kinematics
    if kinematics.jacobian_rank_tolerance <= 0.0:
        raise ValueError("kinematics.jacobian_rank_tolerance must be positive")
    if kinematics.minimum_singular_value <= 0.0:
        raise ValueError("kinematics.minimum_singular_value must be positive")
    if kinematics.maximum_condition_number <= 1.0:
        raise ValueError(
            "kinematics.maximum_condition_number must be greater than one"
        )
    if kinematics.quaternion_convention != "wxyz":
        raise ValueError("kinematics.quaternion_convention must be 'wxyz'")
    if (
        kinematics.orientation_error_convention
        != "world_log_target_times_current_transpose"
    ):
        raise ValueError(
            "Unsupported kinematics.orientation_error_convention"
        )

    safety = config.safety
    if safety.joint_limit_margin <= 0.0:
        raise ValueError("safety.joint_limit_margin must be positive")
    if any(value <= 0.0 for value in safety.joint_velocity_limits):
        raise ValueError("safety.joint_velocity_limits values must be positive")
    if safety.maximum_tcp_position_error <= 0.0:
        raise ValueError("safety.maximum_tcp_position_error must be positive")
    if safety.maximum_tcp_orientation_error <= 0.0:
        raise ValueError("safety.maximum_tcp_orientation_error must be positive")
    if safety.sustained_violation_duration <= 0.0:
        raise ValueError("safety.sustained_violation_duration must be positive")
    if safety.simulation_instability_acceleration <= 0.0:
        raise ValueError(
            "safety.simulation_instability_acceleration must be positive"
        )
    if any(
        lower >= upper
        for lower, upper in zip(safety.workspace_min, safety.workspace_max)
    ):
        raise ValueError("safety workspace bounds must be strictly ordered")

    trajectory = config.trajectory
    poses = (trajectory.initial_joint_pose, *trajectory.hold_joint_poses)
    if len(trajectory.hold_joint_poses) < 3:
        raise ValueError(
            "trajectory.hold_joint_poses must contain at least three poses"
        )
    if not all(
        _joint_pose_is_legal(pose, safety.joint_limit_margin) for pose in poses
    ):
        raise ValueError("configured joint pose violates a soft joint limit")
    durations = (
        trajectory.hold_duration,
        trajectory.translation_duration,
        trajectory.orientation_duration,
        trajectory.line_duration,
        trajectory.circle_duration,
    )
    if any(value <= 0.0 for value in durations):
        raise ValueError("all Cartesian trajectory durations must be positive")
    if max(durations) > simulation.maximum_duration:
        raise ValueError("trajectory duration exceeds simulation.maximum_duration")
    if not (
        0.0
        < trajectory.translation_ramp_duration * 2.0
        <= trajectory.translation_duration
    ):
        raise ValueError("translation ramp must be positive and at most duration/2")
    if not (
        0.0
        < trajectory.orientation_ramp_duration * 2.0
        <= trajectory.orientation_duration
    ):
        raise ValueError("orientation ramp must be positive and at most duration/2")
    if trajectory.translation_frequency_hz <= 0.0:
        raise ValueError("translation frequency must be positive")
    if trajectory.orientation_frequency_hz <= 0.0:
        raise ValueError("orientation frequency must be positive")
    if any(value <= 0.0 for value in trajectory.translation_amplitudes):
        raise ValueError("translation amplitudes must be positive")
    if any(value <= 0.0 for value in trajectory.orientation_amplitudes):
        raise ValueError("orientation amplitudes must be positive")
    if trajectory.circle_radius <= 0.0:
        raise ValueError("circle radius must be positive")
    if (
        len(set(trajectory.circle_plane_axes)) != 2
        or any(axis not in (0, 1, 2) for axis in trajectory.circle_plane_axes)
    ):
        raise ValueError("circle_plane_axes must contain two distinct xyz indices")
    if trajectory.seed < 0:
        raise ValueError("trajectory.seed must be non-negative")


def load_cartesian_config(path: str | Path) -> CartesianBenchmarkConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Cartesian configuration does not exist: {source_path}"
        )
    source_text = source_path.read_text(encoding="utf-8")
    raw = tomllib.loads(source_text)
    model = _section(raw, "model")
    simulation = _section(raw, "simulation")
    controller = _section(raw, "controller")
    kinematics = _section(raw, "kinematics")
    safety = _section(raw, "safety")
    trajectory = _section(raw, "trajectory")

    model_path_text = model.get("path")
    if not isinstance(model_path_text, str) or not model_path_text.strip():
        raise ValueError("model.path must be a non-empty string")
    model_path = Path(model_path_text)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    tcp_site = model.get("tcp_site")
    if not isinstance(tcp_site, str):
        raise ValueError("model.tcp_site must be a string")

    hold_raw = trajectory.get("hold_joint_poses")
    if not isinstance(hold_raw, list):
        raise ValueError("trajectory.hold_joint_poses must be an array")
    hold_poses = tuple(
        _vector(value, JOINT_COUNT, f"trajectory.hold_joint_poses[{index}]")
        for index, value in enumerate(hold_raw)
    )
    circle_axes_raw = trajectory.get("circle_plane_axes")
    if not isinstance(circle_axes_raw, list) or len(circle_axes_raw) != 2:
        raise ValueError("trajectory.circle_plane_axes must contain two integers")
    circle_axes = tuple(
        _integer(value, "trajectory.circle_plane_axes")
        for value in circle_axes_raw
    )

    config = CartesianBenchmarkConfig(
        source_path=source_path,
        source_text=source_text,
        model=CartesianModelConfig(
            path=model_path.resolve(),
            tcp_site=tcp_site,
        ),
        simulation=CartesianSimulationConfig(
            control_frequency_hz=_number(
                simulation.get("control_frequency_hz"),
                "simulation.control_frequency_hz",
            ),
            substeps=_integer(
                simulation.get("substeps"), "simulation.substeps"
            ),
            maximum_duration=_number(
                simulation.get("maximum_duration"),
                "simulation.maximum_duration",
            ),
            output_sampling_frequency_hz=_number(
                simulation.get("output_sampling_frequency_hz"),
                "simulation.output_sampling_frequency_hz",
            ),
        ),
        controller=CartesianControllerConfig(
            translational_stiffness=_vector(
                controller.get("translational_stiffness"),
                3,
                "controller.translational_stiffness",
            ),
            rotational_stiffness=_vector(
                controller.get("rotational_stiffness"),
                3,
                "controller.rotational_stiffness",
            ),
            translational_damping=_vector(
                controller.get("translational_damping"),
                3,
                "controller.translational_damping",
            ),
            rotational_damping=_vector(
                controller.get("rotational_damping"),
                3,
                "controller.rotational_damping",
            ),
            dynamics_compensation_mode=str(
                controller.get("dynamics_compensation_mode", "")
            ),
            torque_limits=_vector(
                controller.get("torque_limits"),
                JOINT_COUNT,
                "controller.torque_limits",
            ),
            torque_rate_limits=_vector(
                controller.get("torque_rate_limits"),
                JOINT_COUNT,
                "controller.torque_rate_limits",
            ),
        ),
        kinematics=KinematicsConfig(
            jacobian_rank_tolerance=_number(
                kinematics.get("jacobian_rank_tolerance"),
                "kinematics.jacobian_rank_tolerance",
            ),
            minimum_singular_value=_number(
                kinematics.get("minimum_singular_value"),
                "kinematics.minimum_singular_value",
            ),
            maximum_condition_number=_number(
                kinematics.get("maximum_condition_number"),
                "kinematics.maximum_condition_number",
            ),
            quaternion_convention=str(
                kinematics.get("quaternion_convention", "")
            ),
            orientation_error_convention=str(
                kinematics.get("orientation_error_convention", "")
            ),
        ),
        safety=CartesianSafetyConfig(
            joint_limit_margin=_number(
                safety.get("joint_limit_margin"), "safety.joint_limit_margin"
            ),
            joint_velocity_limits=_vector(
                safety.get("joint_velocity_limits"),
                JOINT_COUNT,
                "safety.joint_velocity_limits",
            ),
            maximum_tcp_position_error=_number(
                safety.get("maximum_tcp_position_error"),
                "safety.maximum_tcp_position_error",
            ),
            maximum_tcp_orientation_error=_number(
                safety.get("maximum_tcp_orientation_error"),
                "safety.maximum_tcp_orientation_error",
            ),
            sustained_violation_duration=_number(
                safety.get("sustained_violation_duration"),
                "safety.sustained_violation_duration",
            ),
            simulation_instability_acceleration=_number(
                safety.get("simulation_instability_acceleration"),
                "safety.simulation_instability_acceleration",
            ),
            workspace_min=_vector(
                safety.get("workspace_min"), 3, "safety.workspace_min"
            ),
            workspace_max=_vector(
                safety.get("workspace_max"), 3, "safety.workspace_max"
            ),
        ),
        trajectory=CartesianTrajectoryConfig(
            initial_joint_pose=_vector(
                trajectory.get("initial_joint_pose"),
                JOINT_COUNT,
                "trajectory.initial_joint_pose",
            ),
            hold_joint_poses=hold_poses,
            hold_duration=_number(
                trajectory.get("hold_duration"), "trajectory.hold_duration"
            ),
            translation_amplitudes=_vector(
                trajectory.get("translation_amplitudes"),
                3,
                "trajectory.translation_amplitudes",
            ),
            translation_frequency_hz=_number(
                trajectory.get("translation_frequency_hz"),
                "trajectory.translation_frequency_hz",
            ),
            translation_ramp_duration=_number(
                trajectory.get("translation_ramp_duration"),
                "trajectory.translation_ramp_duration",
            ),
            translation_duration=_number(
                trajectory.get("translation_duration"),
                "trajectory.translation_duration",
            ),
            orientation_amplitudes=_vector(
                trajectory.get("orientation_amplitudes"),
                3,
                "trajectory.orientation_amplitudes",
            ),
            orientation_frequency_hz=_number(
                trajectory.get("orientation_frequency_hz"),
                "trajectory.orientation_frequency_hz",
            ),
            orientation_ramp_duration=_number(
                trajectory.get("orientation_ramp_duration"),
                "trajectory.orientation_ramp_duration",
            ),
            orientation_duration=_number(
                trajectory.get("orientation_duration"),
                "trajectory.orientation_duration",
            ),
            line_displacement=_vector(
                trajectory.get("line_displacement"),
                3,
                "trajectory.line_displacement",
            ),
            line_duration=_number(
                trajectory.get("line_duration"), "trajectory.line_duration"
            ),
            circle_radius=_number(
                trajectory.get("circle_radius"), "trajectory.circle_radius"
            ),
            circle_plane_axes=(int(circle_axes[0]), int(circle_axes[1])),
            circle_duration=_number(
                trajectory.get("circle_duration"), "trajectory.circle_duration"
            ),
            seed=_integer(trajectory.get("seed"), "trajectory.seed"),
        ),
    )
    validate_cartesian_config(config)
    return config
