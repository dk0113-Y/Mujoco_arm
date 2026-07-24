from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import mujoco
import numpy as np

from controllers.joint_impedance import JointImpedanceController
from environments.panda_torque_env import PandaTorqueEnv

from .config import ControlBenchmarkConfig, PROJECT_ROOT, load_control_config
from .dynamics import MuJoCoDynamicsProvider
from .metrics import compute_episode_metrics, summarize_metrics
from .outputs import (
    EPISODE_FIELDS,
    TIMESERIES_FIELDS,
    prepare_output_directory,
    write_csv,
    write_json,
)
from .trajectories import (
    HoldTrajectory,
    MultiJointSmoothTrajectory,
    SingleJointSineTrajectory,
)


SUPPORTED_EXPERIMENTS = (
    "zero_torque",
    "compensation_hold",
    "impedance_hold",
    "single_joint",
    "multi_joint",
    "all",
)


@dataclass(frozen=True)
class EpisodeSpec:
    experiment: str
    case_name: str
    initial_pose: np.ndarray
    trajectory: Any
    mode: str
    settling_applicable: bool


def _episode_specs(config: ControlBenchmarkConfig, experiment: str) -> list[EpisodeSpec]:
    trajectory_config = config.trajectory
    initial = np.asarray(trajectory_config.initial_pose, dtype=float)
    specs: list[EpisodeSpec] = []
    selected = set(SUPPORTED_EXPERIMENTS[:-1]) if experiment == "all" else {experiment}
    if "zero_torque" in selected:
        specs.append(
            EpisodeSpec(
                experiment="zero_torque",
                case_name="gravity_response",
                initial_pose=initial,
                trajectory=HoldTrajectory(
                    initial, trajectory_config.zero_torque_duration
                ),
                mode="zero",
                settling_applicable=False,
            )
        )
    if "compensation_hold" in selected:
        specs.append(
            EpisodeSpec(
                experiment="compensation_hold",
                case_name="verified_bias_only",
                initial_pose=initial,
                trajectory=HoldTrajectory(
                    initial, trajectory_config.compensation_hold_duration
                ),
                mode="compensation",
                settling_applicable=False,
            )
        )
    if "impedance_hold" in selected:
        for index, pose in enumerate(trajectory_config.hold_poses):
            pose_value = np.asarray(pose, dtype=float)
            specs.append(
                EpisodeSpec(
                    experiment="impedance_hold",
                    case_name=f"pose_{index + 1}",
                    initial_pose=pose_value,
                    trajectory=HoldTrajectory(
                        pose_value, trajectory_config.impedance_hold_duration
                    ),
                    mode="impedance",
                    settling_applicable=True,
                )
            )
    if "single_joint" in selected:
        for index in range(7):
            specs.append(
                EpisodeSpec(
                    experiment="single_joint",
                    case_name=f"joint_{index + 1}",
                    initial_pose=initial,
                    trajectory=SingleJointSineTrajectory(
                        initial,
                        joint_index=index,
                        amplitude=trajectory_config.single_joint_amplitudes[index],
                        frequency_hz=trajectory_config.single_joint_frequency_hz,
                        duration=trajectory_config.single_joint_duration,
                        ramp_duration=trajectory_config.sine_ramp_duration,
                    ),
                    mode="impedance",
                    settling_applicable=False,
                )
            )
    if "multi_joint" in selected:
        specs.append(
            EpisodeSpec(
                experiment="multi_joint",
                case_name="seven_joint_smooth",
                initial_pose=initial,
                trajectory=MultiJointSmoothTrajectory(
                    initial,
                    amplitudes=np.asarray(
                        trajectory_config.multi_joint_amplitudes, dtype=float
                    ),
                    frequencies_hz=np.asarray(
                        trajectory_config.multi_joint_frequencies_hz, dtype=float
                    ),
                    phases=np.asarray(
                        trajectory_config.multi_joint_phases, dtype=float
                    ),
                    duration=trajectory_config.multi_joint_duration,
                    ramp_duration=trajectory_config.sine_ramp_duration,
                ),
                mode="impedance",
                settling_applicable=False,
            )
        )
    return specs


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


def _run_episode(
    config: ControlBenchmarkConfig,
    spec: EpisodeSpec,
    episode_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = PandaTorqueEnv(config)
    rows: list[dict[str, Any]] = []
    controller = JointImpedanceController(
        stiffness=np.asarray(config.controller.stiffness, dtype=float),
        damping=np.asarray(config.controller.damping, dtype=float),
        torque_limits=np.asarray(config.controller.torque_limits, dtype=float),
        torque_rate_limits=np.asarray(
            config.controller.torque_rate_limits, dtype=float
        ),
    )
    try:
        observation = env.reset(
            qpos=spec.initial_pose,
            qvel=np.zeros(7, dtype=float),
            seed=config.seed,
        )
        controller.reset()
        dynamics = MuJoCoDynamicsProvider(
            env.model,
            env.arm_qpos_addresses,
            env.arm_dof_addresses,
            mode=config.controller.dynamics_compensation_mode,
        )
        planned_steps = int(np.ceil(spec.trajectory.duration / env.control_period))
        sustained_steps = max(
            1,
            int(
                np.ceil(
                    config.safety.sustained_violation_duration
                    / env.control_period
                )
            ),
        )
        saturation_streak = 0
        rate_limit_streak = 0
        runner_termination_reason: str | None = None
        started = time.perf_counter()
        for step_index in range(planned_steps):
            target = spec.trajectory.sample(step_index * env.control_period)
            q_before = observation["joint_positions"]
            dq_before = observation["joint_velocities"]
            terms = dynamics.compute(env.data)
            zeros = np.zeros(7, dtype=float)
            if spec.mode == "zero":
                commanded = zeros.copy()
                feedback = zeros.copy()
                applied_compensation = zeros.copy()
                raw = zeros.copy()
                controller_rate_limited = commanded.copy()
                controller_rate_mask = np.zeros(7, dtype=bool)
                controller_saturation = np.zeros(7, dtype=bool)
                env.set_tracking_target(None)
            elif spec.mode == "compensation":
                commanded = terms.compensation.copy()
                feedback = zeros.copy()
                applied_compensation = terms.compensation.copy()
                raw = commanded.copy()
                controller_rate_limited = commanded.copy()
                controller_rate_mask = np.zeros(7, dtype=bool)
                controller_saturation = np.zeros(7, dtype=bool)
                env.set_tracking_target(None)
            else:
                env.set_tracking_target(target.q)
                commanded, control_diagnostics = controller.compute(
                    q=q_before,
                    dq=dq_before,
                    q_target=target.q,
                    dq_target=target.dq,
                    dynamics_compensation=terms.compensation,
                    dt=env.control_period,
                )
                feedback = control_diagnostics.feedback_torque
                applied_compensation = control_diagnostics.dynamics_compensation
                raw = control_diagnostics.raw_torque
                controller_rate_limited = control_diagnostics.rate_limited_torque
                controller_rate_mask = control_diagnostics.rate_limit_mask
                controller_saturation = control_diagnostics.saturation_mask

            observation, env_diagnostics = env.step(commanded)
            combined_saturation = np.logical_or(
                controller_saturation, env_diagnostics["saturation_mask"]
            )
            combined_rate_limit = np.logical_or(
                controller_rate_mask,
                env_diagnostics["torque_rate_limit_mask"],
            )
            termination_reason = env_diagnostics["termination_reason"] or ""
            saturation_streak = (
                saturation_streak + 1 if np.any(combined_saturation) else 0
            )
            rate_limit_streak = (
                rate_limit_streak + 1 if np.any(combined_rate_limit) else 0
            )
            if not termination_reason and saturation_streak >= sustained_steps:
                runner_termination_reason = "torque_saturation_sustained"
                termination_reason = runner_termination_reason
            elif not termination_reason and rate_limit_streak >= sustained_steps:
                runner_termination_reason = "torque_rate_limit_sustained"
                termination_reason = runner_termination_reason
            q_after = observation["joint_positions"]
            dq_after = observation["joint_velocities"]
            row = {
                "episode_id": episode_id,
                "experiment": spec.experiment,
                "case": spec.case_name,
                "control_cycle": observation["control_cycle"],
                "sim_time": observation["simulation_time"],
                "q": q_after.copy(),
                "dq": dq_after.copy(),
                "q_target": target.q.copy(),
                "dq_target": target.dq.copy(),
                "position_error": (target.q - q_after).copy(),
                "velocity_error": (target.dq - dq_after).copy(),
                "feedback_torque": feedback.copy(),
                "dynamics_compensation": applied_compensation.copy(),
                "gravity": terms.gravity.copy(),
                "coriolis_centrifugal": terms.coriolis_centrifugal.copy(),
                "passive_force": terms.passive.copy(),
                "raw_torque": raw.copy(),
                "rate_limited_torque": controller_rate_limited.copy(),
                "final_torque": env_diagnostics["clipped_torque"].copy(),
                "actuator_force": env_diagnostics["actuator_force"].copy(),
                "saturation_mask": combined_saturation.copy(),
                "rate_limit_mask": combined_rate_limit.copy(),
                "joint_limit_mask": env_diagnostics["joint_limit_mask"].copy(),
                "velocity_limit_mask": env_diagnostics[
                    "velocity_limit_mask"
                ].copy(),
                "finite_value_status": env_diagnostics["finite_value_status"],
                "termination_reason": termination_reason,
            }
            rows.append(row)
            if termination_reason:
                break

        wall_clock_duration = time.perf_counter() - started
        termination_reason = (
            runner_termination_reason or env.termination_reason or "completed"
        )
        rows[-1]["termination_reason"] = termination_reason
        metrics = compute_episode_metrics(
            episode_id=episode_id,
            experiment=spec.experiment,
            case_name=spec.case_name,
            rows=rows,
            termination_reason=termination_reason,
            wall_clock_duration=wall_clock_duration,
            settling_applicable=spec.settling_applicable,
        )
        return rows, metrics
    finally:
        env.close()


def _manifest(
    config: ControlBenchmarkConfig,
    *,
    experiment: str,
    episode_ids: list[str],
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
        "benchmark": "JI-Baseline v1",
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
        "dynamics_compensation_mode": (
            config.controller.dynamics_compensation_mode
        ),
        "dynamics_compensation_terms": (
            ["gravity", "coriolis_centrifugal"]
            if config.controller.dynamics_compensation_mode
            == "gravity_coriolis"
            else [config.controller.dynamics_compensation_mode]
            if config.controller.dynamics_compensation_mode != "none"
            else []
        ),
        "passive_compensation_included": False,
        "constraint_compensation_included": False,
    }


def run_benchmark(
    config: ControlBenchmarkConfig | str | Path,
    *,
    experiment: str,
    output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"Unsupported experiment {experiment!r}; expected one of "
            f"{SUPPORTED_EXPERIMENTS}"
        )
    config_value = (
        load_control_config(config)
        if isinstance(config, (str, Path))
        else config
    )
    specs = _episode_specs(config_value, experiment)
    validation_env = PandaTorqueEnv(config_value)
    validation_env.close()
    output_path = prepare_output_directory(output, overwrite=overwrite)

    all_rows: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    for index, spec in enumerate(specs, start=1):
        episode_id = (
            f"{index:02d}_{spec.experiment}_{spec.case_name}"
        )
        rows, metrics = _run_episode(config_value, spec, episode_id)
        episode_ids.append(episode_id)
        all_rows.extend(rows)
        all_metrics.append(metrics)

    summary = summarize_metrics(all_metrics)
    summary["benchmark"] = "JI-Baseline v1"
    summary["requested_experiment"] = experiment
    summary["episodes"] = all_metrics
    manifest = _manifest(
        config_value, experiment=experiment, episode_ids=episode_ids
    )
    write_json(output_path / "run_manifest.json", manifest)
    write_csv(
        output_path / "episode_metrics.csv", all_metrics, EPISODE_FIELDS
    )
    write_csv(output_path / "timeseries.csv", all_rows, TIMESERIES_FIELDS)
    write_json(output_path / "summary.json", summary)
    (output_path / "config_snapshot.toml").write_text(
        config_value.source_text, encoding="utf-8"
    )
    return summary
