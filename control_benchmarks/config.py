from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_COUNT = 7
PANDA_JOINT_LIMITS = (
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
)
DYNAMICS_MODES = frozenset({"none", "gravity", "gravity_coriolis"})


@dataclass(frozen=True)
class ModelConfig:
    path: Path


@dataclass(frozen=True)
class SimulationConfig:
    control_frequency_hz: float
    substeps: int
    maximum_duration: float
    output_sampling_frequency_hz: float

    @property
    def control_period(self) -> float:
        return 1.0 / self.control_frequency_hz


@dataclass(frozen=True)
class ControllerConfig:
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    dynamics_compensation_mode: str
    torque_limits: tuple[float, ...]
    torque_rate_limits: tuple[float, ...]


@dataclass(frozen=True)
class SafetyConfig:
    joint_limit_margin: float
    joint_velocity_limits: tuple[float, ...]
    maximum_tracking_error: tuple[float, ...]
    sustained_violation_duration: float
    simulation_instability_acceleration: float


@dataclass(frozen=True)
class TrajectoryConfig:
    initial_pose: tuple[float, ...]
    hold_poses: tuple[tuple[float, ...], ...]
    zero_torque_duration: float
    compensation_hold_duration: float
    impedance_hold_duration: float
    minimum_jerk_goal: tuple[float, ...]
    minimum_jerk_duration: float
    single_joint_amplitudes: tuple[float, ...]
    single_joint_frequency_hz: float
    single_joint_duration: float
    sine_ramp_duration: float
    multi_joint_amplitudes: tuple[float, ...]
    multi_joint_frequencies_hz: tuple[float, ...]
    multi_joint_phases: tuple[float, ...]
    multi_joint_duration: float


@dataclass(frozen=True)
class ControlBenchmarkConfig:
    source_path: Path
    source_text: str
    model: ModelConfig
    simulation: SimulationConfig
    controller: ControllerConfig
    safety: SafetyConfig
    trajectory: TrajectoryConfig
    seed: int


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


def _vector(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != JOINT_COUNT:
        raise ValueError(f"{name} must contain exactly {JOINT_COUNT} numbers")
    return tuple(_number(item, name) for item in value)


def _pose_is_legal(pose: tuple[float, ...], margin: float) -> bool:
    return all(
        lower + margin < value < upper - margin
        for value, (lower, upper) in zip(pose, PANDA_JOINT_LIMITS)
    )


def validate_control_config(config: ControlBenchmarkConfig) -> None:
    if not config.model.path.is_file():
        raise FileNotFoundError(
            f"Torque-control model does not exist: {config.model.path}"
        )
    expected_root = (PROJECT_ROOT / "models" / "panda_torque").resolve()
    try:
        config.model.path.resolve().relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(
            "model.path must remain inside models/panda_torque"
        ) from exc

    simulation = config.simulation
    if simulation.control_frequency_hz <= 0.0:
        raise ValueError("simulation.control_frequency_hz must be positive")
    if simulation.substeps <= 0:
        raise ValueError("simulation.substeps must be positive")
    if simulation.maximum_duration <= 0.0:
        raise ValueError("simulation.maximum_duration must be positive")
    if not (
        0.0
        < simulation.output_sampling_frequency_hz
        <= simulation.control_frequency_hz
    ):
        raise ValueError(
            "simulation.output_sampling_frequency_hz must be positive and no "
            "greater than the control frequency"
        )
    if not math.isclose(
        simulation.output_sampling_frequency_hz,
        simulation.control_frequency_hz,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "JI-Baseline v1 records every control period, so output sampling "
            "frequency must equal control frequency"
        )

    controller = config.controller
    if controller.dynamics_compensation_mode not in DYNAMICS_MODES:
        raise ValueError(
            "controller.dynamics_compensation_mode must be one of "
            f"{sorted(DYNAMICS_MODES)}"
        )
    if any(value <= 0.0 for value in controller.stiffness):
        raise ValueError("controller.stiffness values must be positive")
    if any(value < 0.0 for value in controller.damping):
        raise ValueError("controller.damping values must be non-negative")
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

    safety = config.safety
    if safety.joint_limit_margin <= 0.0:
        raise ValueError("safety.joint_limit_margin must be positive")
    if any(value <= 0.0 for value in safety.joint_velocity_limits):
        raise ValueError("safety.joint_velocity_limits values must be positive")
    if any(value <= 0.0 for value in safety.maximum_tracking_error):
        raise ValueError("safety.maximum_tracking_error values must be positive")
    if safety.sustained_violation_duration <= 0.0:
        raise ValueError("safety.sustained_violation_duration must be positive")
    if safety.simulation_instability_acceleration <= 0.0:
        raise ValueError(
            "safety.simulation_instability_acceleration must be positive"
        )

    trajectory = config.trajectory
    poses = (trajectory.initial_pose, *trajectory.hold_poses)
    if len(trajectory.hold_poses) < 3:
        raise ValueError("trajectory.hold_poses must contain at least three poses")
    if not all(_pose_is_legal(pose, safety.joint_limit_margin) for pose in poses):
        raise ValueError("configured initial/hold pose violates a soft joint limit")
    if not _pose_is_legal(
        trajectory.minimum_jerk_goal, safety.joint_limit_margin
    ):
        raise ValueError("trajectory.minimum_jerk_goal violates a soft joint limit")
    durations = (
        trajectory.zero_torque_duration,
        trajectory.compensation_hold_duration,
        trajectory.impedance_hold_duration,
        trajectory.minimum_jerk_duration,
        trajectory.single_joint_duration,
        trajectory.sine_ramp_duration,
        trajectory.multi_joint_duration,
    )
    if any(duration <= 0.0 for duration in durations):
        raise ValueError("all trajectory durations must be positive")
    if max(durations[:-1] + (trajectory.multi_joint_duration,)) > (
        simulation.maximum_duration
    ):
        raise ValueError("trajectory duration exceeds simulation.maximum_duration")
    if trajectory.sine_ramp_duration * 2.0 > trajectory.single_joint_duration:
        raise ValueError(
            "trajectory.sine_ramp_duration must not exceed half the sine duration"
        )
    if trajectory.single_joint_frequency_hz <= 0.0:
        raise ValueError("trajectory.single_joint_frequency_hz must be positive")
    if any(value < 0.0 for value in trajectory.single_joint_amplitudes):
        raise ValueError("trajectory.single_joint_amplitudes must be non-negative")
    if any(value < 0.0 for value in trajectory.multi_joint_amplitudes):
        raise ValueError("trajectory.multi_joint_amplitudes must be non-negative")
    if any(value <= 0.0 for value in trajectory.multi_joint_frequencies_hz):
        raise ValueError(
            "trajectory.multi_joint_frequencies_hz values must be positive"
        )
    for base, amplitude, limits in zip(
        trajectory.initial_pose,
        trajectory.single_joint_amplitudes,
        PANDA_JOINT_LIMITS,
    ):
        if not (
            limits[0] + safety.joint_limit_margin
            < base - amplitude
            <= base + amplitude
            < limits[1] - safety.joint_limit_margin
        ):
            raise ValueError("single-joint trajectory exceeds a soft joint limit")
    for base, amplitude, limits in zip(
        trajectory.initial_pose,
        trajectory.multi_joint_amplitudes,
        PANDA_JOINT_LIMITS,
    ):
        if not (
            limits[0] + safety.joint_limit_margin
            < base - amplitude
            <= base + amplitude
            < limits[1] - safety.joint_limit_margin
        ):
            raise ValueError("multi-joint trajectory exceeds a soft joint limit")
    if isinstance(config.seed, bool) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer")


def load_control_config(path: str | Path) -> ControlBenchmarkConfig:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Control configuration does not exist: {source_path}")
    source_text = source_path.read_text(encoding="utf-8")
    raw = tomllib.loads(source_text)
    model = _section(raw, "model")
    simulation = _section(raw, "simulation")
    controller = _section(raw, "controller")
    safety = _section(raw, "safety")
    trajectory = _section(raw, "trajectory")

    model_value = model.get("path")
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("model.path must be a non-empty string")
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    hold_value = trajectory.get("hold_poses")
    if not isinstance(hold_value, list):
        raise ValueError("trajectory.hold_poses must be an array of seven-axis poses")
    hold_poses = tuple(
        _vector(pose, f"trajectory.hold_poses[{index}]")
        for index, pose in enumerate(hold_value)
    )
    config = ControlBenchmarkConfig(
        source_path=source_path,
        source_text=source_text,
        model=ModelConfig(path=model_path.resolve()),
        simulation=SimulationConfig(
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
        controller=ControllerConfig(
            stiffness=_vector(
                controller.get("stiffness"), "controller.stiffness"
            ),
            damping=_vector(controller.get("damping"), "controller.damping"),
            dynamics_compensation_mode=str(
                controller.get("dynamics_compensation_mode", "")
            ),
            torque_limits=_vector(
                controller.get("torque_limits"), "controller.torque_limits"
            ),
            torque_rate_limits=_vector(
                controller.get("torque_rate_limits"),
                "controller.torque_rate_limits",
            ),
        ),
        safety=SafetyConfig(
            joint_limit_margin=_number(
                safety.get("joint_limit_margin"), "safety.joint_limit_margin"
            ),
            joint_velocity_limits=_vector(
                safety.get("joint_velocity_limits"),
                "safety.joint_velocity_limits",
            ),
            maximum_tracking_error=_vector(
                safety.get("maximum_tracking_error"),
                "safety.maximum_tracking_error",
            ),
            sustained_violation_duration=_number(
                safety.get("sustained_violation_duration"),
                "safety.sustained_violation_duration",
            ),
            simulation_instability_acceleration=_number(
                safety.get("simulation_instability_acceleration"),
                "safety.simulation_instability_acceleration",
            ),
        ),
        trajectory=TrajectoryConfig(
            initial_pose=_vector(
                trajectory.get("initial_pose"), "trajectory.initial_pose"
            ),
            hold_poses=hold_poses,
            zero_torque_duration=_number(
                trajectory.get("zero_torque_duration"),
                "trajectory.zero_torque_duration",
            ),
            compensation_hold_duration=_number(
                trajectory.get("compensation_hold_duration"),
                "trajectory.compensation_hold_duration",
            ),
            impedance_hold_duration=_number(
                trajectory.get("impedance_hold_duration"),
                "trajectory.impedance_hold_duration",
            ),
            minimum_jerk_goal=_vector(
                trajectory.get("minimum_jerk_goal"),
                "trajectory.minimum_jerk_goal",
            ),
            minimum_jerk_duration=_number(
                trajectory.get("minimum_jerk_duration"),
                "trajectory.minimum_jerk_duration",
            ),
            single_joint_amplitudes=_vector(
                trajectory.get("single_joint_amplitudes"),
                "trajectory.single_joint_amplitudes",
            ),
            single_joint_frequency_hz=_number(
                trajectory.get("single_joint_frequency_hz"),
                "trajectory.single_joint_frequency_hz",
            ),
            single_joint_duration=_number(
                trajectory.get("single_joint_duration"),
                "trajectory.single_joint_duration",
            ),
            sine_ramp_duration=_number(
                trajectory.get("sine_ramp_duration"),
                "trajectory.sine_ramp_duration",
            ),
            multi_joint_amplitudes=_vector(
                trajectory.get("multi_joint_amplitudes"),
                "trajectory.multi_joint_amplitudes",
            ),
            multi_joint_frequencies_hz=_vector(
                trajectory.get("multi_joint_frequencies_hz"),
                "trajectory.multi_joint_frequencies_hz",
            ),
            multi_joint_phases=_vector(
                trajectory.get("multi_joint_phases"),
                "trajectory.multi_joint_phases",
            ),
            multi_joint_duration=_number(
                trajectory.get("multi_joint_duration"),
                "trajectory.multi_joint_duration",
            ),
        ),
        seed=_integer(raw.get("seed"), "seed"),
    )
    validate_control_config(config)
    return config
