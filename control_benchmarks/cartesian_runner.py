"""Headless runner for the isolated fixed-gain CI-Baseline v1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import mujoco
import numpy as np

from controllers.cartesian_impedance import CartesianImpedanceController
from environments.panda_torque_env import PandaTorqueEnv

from .cartesian_config import (
    CartesianBenchmarkConfig,
    load_cartesian_config,
)
from .cartesian_metrics import (
    compute_cartesian_episode_metrics,
    summarize_cartesian_metrics,
)
from .cartesian_outputs import (
    CARTESIAN_EPISODE_FIELDS,
    CARTESIAN_TIMESERIES_FIELDS,
)
from .cartesian_trajectories import (
    AxisTranslationTrajectory,
    CartesianHoldTrajectory,
    CircleTrajectory,
    OrientationAxisTrajectory,
    StraightLineTrajectory,
    validate_workspace,
)
from .config import PROJECT_ROOT
from .dynamics import MuJoCoDynamicsProvider
from .kinematics import (
    PandaTcpKinematics,
    PandaTcpKinematicsProvider,
    controllability_termination_reason,
    orientation_error_world,
    rotation_to_quaternion_wxyz,
)
from .outputs import prepare_output_directory, write_csv, write_json


SUPPORTED_CARTESIAN_EXPERIMENTS = (
    "cartesian_hold",
    "translation_axes",
    "orientation_axes",
    "straight_line",
    "circle",
    "all",
)
AXIS_NAMES = ("x", "y", "z")
CARTESIAN_TERMINATION_REASONS = (
    "completed",
    "joint_position_limit",
    "joint_velocity_limit",
    "torque_saturation_sustained",
    "torque_rate_limit_sustained",
    "tcp_position_error_exceeded",
    "tcp_orientation_error_exceeded",
    "jacobian_rank_deficient",
    "jacobian_condition_exceeded",
    "invalid_orientation",
    "unexpected_contact",
    "non_finite_state",
    "simulation_instability",
    "timeout",
)


@dataclass(frozen=True)
class CartesianEpisodeSpec:
    experiment: str
    case_name: str
    initial_pose: np.ndarray
    axis: int | None = None


def _episode_specs(
    config: CartesianBenchmarkConfig, experiment: str
) -> list[CartesianEpisodeSpec]:
    trajectory = config.trajectory
    initial = np.asarray(trajectory.initial_joint_pose, dtype=float)
    selected = (
        set(SUPPORTED_CARTESIAN_EXPERIMENTS[:-1])
        if experiment == "all"
        else {experiment}
    )
    specs: list[CartesianEpisodeSpec] = []
    if "cartesian_hold" in selected:
        for index, pose in enumerate(trajectory.hold_joint_poses, start=1):
            specs.append(
                CartesianEpisodeSpec(
                    experiment="cartesian_hold",
                    case_name=f"pose_{index}",
                    initial_pose=np.asarray(pose, dtype=float),
                )
            )
    if "translation_axes" in selected:
        for axis, name in enumerate(AXIS_NAMES):
            specs.append(
                CartesianEpisodeSpec(
                    experiment="translation_axes",
                    case_name=f"world_{name}",
                    initial_pose=initial.copy(),
                    axis=axis,
                )
            )
    if "orientation_axes" in selected:
        for axis, name in enumerate(AXIS_NAMES):
            specs.append(
                CartesianEpisodeSpec(
                    experiment="orientation_axes",
                    case_name=f"world_{name}",
                    initial_pose=initial.copy(),
                    axis=axis,
                )
            )
    if "straight_line" in selected:
        specs.append(
            CartesianEpisodeSpec(
                experiment="straight_line",
                case_name="minimum_jerk_world_xyz",
                initial_pose=initial.copy(),
            )
        )
    if "circle" in selected:
        first, second = trajectory.circle_plane_axes
        specs.append(
            CartesianEpisodeSpec(
                experiment="circle",
                case_name=f"world_{AXIS_NAMES[first]}{AXIS_NAMES[second]}",
                initial_pose=initial.copy(),
            )
        )
    return specs


def _build_trajectory(
    config: CartesianBenchmarkConfig,
    spec: CartesianEpisodeSpec,
    initial: PandaTcpKinematics,
) -> object:
    trajectory = config.trajectory
    if spec.experiment == "cartesian_hold":
        return CartesianHoldTrajectory(
            initial.position, initial.rotation, trajectory.hold_duration
        )
    if spec.experiment == "translation_axes":
        assert spec.axis is not None
        return AxisTranslationTrajectory(
            initial.position,
            initial.rotation,
            axis=spec.axis,
            amplitude=trajectory.translation_amplitudes[spec.axis],
            frequency_hz=trajectory.translation_frequency_hz,
            duration=trajectory.translation_duration,
            ramp_duration=trajectory.translation_ramp_duration,
        )
    if spec.experiment == "orientation_axes":
        assert spec.axis is not None
        return OrientationAxisTrajectory(
            initial.position,
            initial.rotation,
            axis=spec.axis,
            amplitude=trajectory.orientation_amplitudes[spec.axis],
            frequency_hz=trajectory.orientation_frequency_hz,
            duration=trajectory.orientation_duration,
            ramp_duration=trajectory.orientation_ramp_duration,
        )
    if spec.experiment == "straight_line":
        return StraightLineTrajectory(
            initial.position,
            initial.rotation,
            displacement=np.asarray(
                trajectory.line_displacement, dtype=float
            ),
            duration=trajectory.line_duration,
        )
    if spec.experiment == "circle":
        return CircleTrajectory(
            initial.position,
            initial.rotation,
            radius=trajectory.circle_radius,
            plane_axes=trajectory.circle_plane_axes,
            duration=trajectory.circle_duration,
        )
    raise AssertionError(f"Unhandled Cartesian experiment: {spec.experiment}")


def _make_kinematics_provider(
    config: CartesianBenchmarkConfig, env: PandaTorqueEnv
) -> PandaTcpKinematicsProvider:
    return PandaTcpKinematicsProvider(
        env.model,
        env.arm_dof_addresses,
        site_name=config.model.tcp_site,
        rank_tolerance=config.kinematics.jacobian_rank_tolerance,
    )


def _controllability_reason(
    config: CartesianBenchmarkConfig, state: PandaTcpKinematics
) -> str | None:
    return controllability_termination_reason(
        state,
        minimum_rank=6,
        minimum_singular_value=config.kinematics.minimum_singular_value,
        maximum_condition_number=config.kinematics.maximum_condition_number,
    )


def _preflight(
    config: CartesianBenchmarkConfig,
    specs: list[CartesianEpisodeSpec],
) -> list[dict[str, Any]]:
    """Check every formal reset and task trajectory before output creation."""

    results: list[dict[str, Any]] = []
    lower = np.asarray(config.safety.workspace_min, dtype=float)
    upper = np.asarray(config.safety.workspace_max, dtype=float)
    for spec in specs:
        env = PandaTorqueEnv(config)
        try:
            env.reset(
                qpos=spec.initial_pose,
                qvel=np.zeros(7, dtype=float),
                seed=config.seed,
            )
            provider = _make_kinematics_provider(config, env)
            state = provider.compute(env.data)
            reason = _controllability_reason(config, state)
            if reason is not None:
                raise ValueError(
                    f"{spec.experiment}/{spec.case_name} failed initial "
                    f"controllability precheck: {reason}"
                )
            if env.data.ncon:
                raise ValueError(
                    f"{spec.experiment}/{spec.case_name} has an unexpected "
                    "contact at reset"
                )
            trajectory = _build_trajectory(config, spec, state)
            validate_workspace(
                trajectory,
                workspace_min=lower,
                workspace_max=upper,
            )
            results.append(
                {
                    "experiment": spec.experiment,
                    "case": spec.case_name,
                    "initial_jacobian_rank": state.rank,
                    "initial_minimum_singular_value": (
                        state.minimum_singular_value
                    ),
                    "initial_condition_number": state.condition_number,
                    "workspace_sample_count": 201,
                    "unexpected_contact_at_reset": False,
                }
            )
        finally:
            env.close()
    return results


def _controller(config: CartesianBenchmarkConfig) -> CartesianImpedanceController:
    controller = config.controller
    return CartesianImpedanceController(
        translational_stiffness=np.asarray(
            controller.translational_stiffness, dtype=float
        ),
        rotational_stiffness=np.asarray(
            controller.rotational_stiffness, dtype=float
        ),
        translational_damping=np.asarray(
            controller.translational_damping, dtype=float
        ),
        rotational_damping=np.asarray(
            controller.rotational_damping, dtype=float
        ),
        torque_limits=np.asarray(controller.torque_limits, dtype=float),
        torque_rate_limits=np.asarray(
            controller.torque_rate_limits, dtype=float
        ),
    )


def _run_episode(
    config: CartesianBenchmarkConfig,
    spec: CartesianEpisodeSpec,
    episode_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = PandaTorqueEnv(config)
    controller = _controller(config)
    rows: list[dict[str, Any]] = []
    try:
        observation = env.reset(
            qpos=spec.initial_pose,
            qvel=np.zeros(7, dtype=float),
            seed=config.seed,
        )
        provider = _make_kinematics_provider(config, env)
        state = provider.compute(env.data)
        trajectory = _build_trajectory(config, spec, state)
        controller.reset()
        dynamics = MuJoCoDynamicsProvider(
            env.model,
            env.arm_qpos_addresses,
            env.arm_dof_addresses,
            mode=config.controller.dynamics_compensation_mode,
        )
        planned_steps = int(np.ceil(trajectory.duration / env.control_period))
        sustained_steps = max(
            1,
            int(
                np.ceil(
                    config.safety.sustained_violation_duration
                    / env.control_period
                )
            ),
        )
        position_error_streak = 0
        orientation_error_streak = 0
        saturation_streak = 0
        rate_limit_streak = 0
        runner_reason: str | None = None
        started = time.perf_counter()

        for step_index in range(planned_steps):
            target = trajectory.sample(step_index * env.control_period)
            state_before = provider.compute(env.data)
            pre_reason = _controllability_reason(config, state_before)
            if pre_reason is not None:
                runner_reason = pre_reason
                break
            terms = dynamics.compute(env.data)
            commanded, control = controller.compute(
                position=state_before.position,
                rotation=state_before.rotation,
                linear_velocity=state_before.linear_velocity,
                angular_velocity=state_before.angular_velocity,
                target_position=target.position,
                target_rotation=target.rotation,
                target_linear_velocity=target.linear_velocity,
                target_angular_velocity=target.angular_velocity,
                jacobian=state_before.jacobian,
                dynamics_compensation=terms.compensation,
                dt=env.control_period,
            )
            observation, env_diagnostics = env.step(commanded)
            state_after = provider.compute(env.data)
            position_error = target.position - state_after.position
            orientation_error = orientation_error_world(
                state_after.rotation, target.rotation
            )
            linear_velocity_error = (
                target.linear_velocity - state_after.linear_velocity
            )
            angular_velocity_error = (
                target.angular_velocity - state_after.angular_velocity
            )
            position_error_mask = (
                np.linalg.norm(position_error)
                > config.safety.maximum_tcp_position_error
            )
            orientation_error_mask = (
                np.linalg.norm(orientation_error)
                > config.safety.maximum_tcp_orientation_error
            )
            position_error_streak = (
                position_error_streak + 1 if position_error_mask else 0
            )
            orientation_error_streak = (
                orientation_error_streak + 1 if orientation_error_mask else 0
            )
            combined_saturation = np.logical_or(
                control.saturation_mask, env_diagnostics["saturation_mask"]
            )
            combined_rate_limit = np.logical_or(
                control.rate_limit_mask,
                env_diagnostics["torque_rate_limit_mask"],
            )
            saturation_streak = (
                saturation_streak + 1
                if np.any(combined_saturation)
                else 0
            )
            rate_limit_streak = (
                rate_limit_streak + 1
                if np.any(combined_rate_limit)
                else 0
            )
            unexpected_contact = bool(env.data.ncon)
            finite = bool(
                env_diagnostics["finite_value_status"]
                and all(
                    np.all(np.isfinite(value))
                    for value in (
                        state_after.position,
                        state_after.rotation,
                        state_after.twist,
                        state_after.jacobian,
                        position_error,
                        orientation_error,
                        control.task_wrench,
                        control.final_torque,
                    )
                )
            )

            termination_reason = env_diagnostics["termination_reason"] or ""
            if not termination_reason and not finite:
                runner_reason = "non_finite_state"
            elif not termination_reason and unexpected_contact:
                runner_reason = "unexpected_contact"
            elif not termination_reason:
                runner_reason = _controllability_reason(config, state_after)
            if (
                not termination_reason
                and runner_reason is None
                and position_error_streak >= sustained_steps
            ):
                runner_reason = "tcp_position_error_exceeded"
            elif (
                not termination_reason
                and runner_reason is None
                and orientation_error_streak >= sustained_steps
            ):
                runner_reason = "tcp_orientation_error_exceeded"
            elif (
                not termination_reason
                and runner_reason is None
                and saturation_streak >= sustained_steps
            ):
                runner_reason = "torque_saturation_sustained"
            elif (
                not termination_reason
                and runner_reason is None
                and rate_limit_streak >= sustained_steps
            ):
                runner_reason = "torque_rate_limit_sustained"
            if runner_reason is not None:
                termination_reason = runner_reason

            target_quaternion = rotation_to_quaternion_wxyz(target.rotation)
            row = {
                "episode_id": episode_id,
                "experiment": spec.experiment,
                "case": spec.case_name,
                "control_cycle": observation["control_cycle"],
                "sim_time": observation["simulation_time"],
                "q": observation["joint_positions"].copy(),
                "dq": observation["joint_velocities"].copy(),
                "tcp_position": state_after.position.copy(),
                "tcp_quaternion_wxyz": state_after.quaternion_wxyz.copy(),
                "tcp_linear_velocity": state_after.linear_velocity.copy(),
                "tcp_angular_velocity": state_after.angular_velocity.copy(),
                "target_position": target.position.copy(),
                "target_quaternion_wxyz": target_quaternion.copy(),
                "target_linear_velocity": target.linear_velocity.copy(),
                "target_angular_velocity": target.angular_velocity.copy(),
                "position_error": position_error.copy(),
                "orientation_error": orientation_error.copy(),
                "linear_velocity_error": linear_velocity_error.copy(),
                "angular_velocity_error": angular_velocity_error.copy(),
                "task_wrench": control.task_wrench.copy(),
                "task_torque": control.task_torque.copy(),
                "dynamics_compensation": control.dynamics_compensation.copy(),
                "gravity": terms.gravity.copy(),
                "coriolis_centrifugal": terms.coriolis_centrifugal.copy(),
                "passive_force": terms.passive.copy(),
                "raw_torque": control.raw_torque.copy(),
                "rate_limited_torque": control.rate_limited_torque.copy(),
                "final_torque": env_diagnostics["clipped_torque"].copy(),
                "actuator_force": env_diagnostics["actuator_force"].copy(),
                "jacobian_singular_values": state_after.singular_values.copy(),
                "jacobian_rank": state_after.rank,
                "minimum_jacobian_singular_value": (
                    state_after.minimum_singular_value
                ),
                "jacobian_condition_number": state_after.condition_number,
                "twist_consistency_error": state_after.twist_consistency_error,
                "torque_saturation_mask": combined_saturation.copy(),
                "torque_rate_limit_mask": combined_rate_limit.copy(),
                "joint_limit_mask": env_diagnostics["joint_limit_mask"].copy(),
                "joint_velocity_mask": env_diagnostics[
                    "velocity_limit_mask"
                ].copy(),
                "tcp_position_error_mask": bool(position_error_mask),
                "tcp_orientation_error_mask": bool(orientation_error_mask),
                "unexpected_contact": unexpected_contact,
                "finite_value_status": finite,
                "termination_reason": termination_reason,
            }
            rows.append(row)
            if termination_reason:
                break

        wall_clock_duration = time.perf_counter() - started
        if not rows:
            raise RuntimeError(
                f"Episode failed before its first control step: {runner_reason}"
            )
        termination_reason = (
            runner_reason or env.termination_reason or "completed"
        )
        rows[-1]["termination_reason"] = termination_reason
        metrics = compute_cartesian_episode_metrics(
            episode_id=episode_id,
            experiment=spec.experiment,
            case_name=spec.case_name,
            rows=rows,
            termination_reason=termination_reason,
            wall_clock_duration=wall_clock_duration,
        )
        return rows, metrics
    finally:
        env.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _manifest(
    config: CartesianBenchmarkConfig,
    *,
    experiment: str,
    episode_ids: list[str],
    preflight: list[dict[str, Any]],
) -> dict[str, Any]:
    model_dir = config.model.path.parent
    torque_model_path = model_dir / "panda_torque.xml"
    compiled_model = mujoco.MjModel.from_xml_path(str(config.model.path))
    status = _git_output("status", "--porcelain")
    submodule_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT / "models" / "mujoco_menagerie",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    try:
        config_path = config.source_path.relative_to(PROJECT_ROOT).as_posix()
        config_is_external = False
    except ValueError:
        config_path = f"<external>/{config.source_path.name}"
        config_is_external = True
    return {
        "benchmark": "CI-Baseline v1",
        "scope": "fixed-gain free-space Cartesian impedance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "episode_ids": episode_ids,
        "seed": config.seed,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "mujoco_version": mujoco.__version__,
        "platform": platform.platform(),
        "model_path": config.model.path.relative_to(PROJECT_ROOT).as_posix(),
        "model_sha256": _sha256(config.model.path),
        "panda_torque_sha256": _sha256(torque_model_path),
        "menagerie_commit": submodule_commit,
        "config_path": config_path,
        "config_is_external": config_is_external,
        "config_sha256": hashlib.sha256(
            config.source_text.encode("utf-8")
        ).hexdigest(),
        "model_timestep_seconds": float(compiled_model.opt.timestep),
        "control_frequency_hz": config.simulation.control_frequency_hz,
        "control_period_seconds": config.simulation.control_period,
        "simulation_substeps": config.simulation.substeps,
        "tcp_site": config.model.tcp_site,
        "tcp_parent_body": "hand",
        "tcp_position_in_hand_m": [0.0, 0.0, 0.103],
        "tcp_quaternion_in_hand_wxyz": [1.0, 0.0, 0.0, 0.0],
        "quaternion_convention": "wxyz, normalized, canonical hemisphere",
        "pose_frame": "MuJoCo world",
        "jacobian_frame": "MuJoCo world",
        "jacobian_row_order": ["linear_xyz", "angular_xyz"],
        "jacobian_column_order": [
            f"joint{index}" for index in range(1, 8)
        ],
        "orientation_error": "Log(R_target @ R_current.T), world frame",
        "dynamics_compensation_mode": (
            config.controller.dynamics_compensation_mode
        ),
        "dynamics_compensation_terms": [
            "gravity",
            "coriolis_centrifugal",
        ],
        "passive_compensation_included": False,
        "constraint_compensation_included": False,
        "nullspace_torque_included": False,
        "inverse_kinematics_used": False,
        "contact_feedback_used": False,
        "termination_reasons": list(CARTESIAN_TERMINATION_REASONS),
        "controllability_thresholds": {
            "rank": 6,
            "rank_tolerance": config.kinematics.jacobian_rank_tolerance,
            "minimum_singular_value": (
                config.kinematics.minimum_singular_value
            ),
            "maximum_condition_number": (
                config.kinematics.maximum_condition_number
            ),
        },
        "preflight": preflight,
    }


def run_cartesian_benchmark(
    config: CartesianBenchmarkConfig | str | Path,
    *,
    experiment: str,
    output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if experiment not in SUPPORTED_CARTESIAN_EXPERIMENTS:
        raise ValueError(
            f"Unsupported experiment {experiment!r}; expected one of "
            f"{SUPPORTED_CARTESIAN_EXPERIMENTS}"
        )
    config_value = (
        load_cartesian_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    specs = _episode_specs(config_value, experiment)
    preflight = _preflight(config_value, specs)
    output_path = prepare_output_directory(output, overwrite=overwrite)

    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    for index, spec in enumerate(specs, start=1):
        episode_id = f"{index:02d}_{spec.experiment}_{spec.case_name}"
        rows, metrics = _run_episode(config_value, spec, episode_id)
        episode_ids.append(episode_id)
        all_rows.extend(rows)
        all_metrics.append(metrics)

    summary = summarize_cartesian_metrics(all_metrics)
    summary["benchmark"] = "CI-Baseline v1"
    summary["requested_experiment"] = experiment
    summary["episodes"] = all_metrics
    manifest = _manifest(
        config_value,
        experiment=experiment,
        episode_ids=episode_ids,
        preflight=preflight,
    )
    write_json(output_path / "run_manifest.json", manifest)
    write_csv(
        output_path / "episode_metrics.csv",
        all_metrics,
        CARTESIAN_EPISODE_FIELDS,
    )
    write_csv(
        output_path / "timeseries.csv",
        all_rows,
        CARTESIAN_TIMESERIES_FIELDS,
    )
    write_json(output_path / "summary.json", summary)
    (output_path / "config_snapshot.toml").write_text(
        config_value.source_text, encoding="utf-8"
    )
    return summary
