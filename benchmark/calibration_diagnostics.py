from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from controllers import B1DiagnosticSnapshot
from environments import EnvConfig
from evaluation.protocol import ProtocolConfig, validate_baseline_compatibility
from perception.image_io import save_rgb_png

from .manifest import repository_metadata, runtime_metadata, sha256_file
from .methods import FORMAL_METHOD_IDS, MethodSpec, assert_static_fairness, resolve_methods
from .pairing import DEFAULT_FINGERPRINT_ATOL
from .runner import (
    PROJECT_ROOT,
    BenchmarkRunError,
    _effective_config,
    _episode_row,
    _execute_episode,
    _logger_for,
    _paired_row,
    _prepare_output_dir,
    _validate_execution_pair,
)
from .schemas import PAIRED_RESULT_FIELDS, episode_fieldnames, write_csv, write_json
from .seed_io import load_seeds


DIAGNOSTIC_TELEMETRY_SCHEMA_VERSION = "1.0.0"
REQUIRED_DIAGNOSTIC_SEEDS = (2802, 3915, 2957, 1268)
REQUIRED_METHODS = ("b0_oracle", "b1_vision")
RUN_KIND = "calibration_diagnostic_replay"

PROTECTED_FORMAL_PATHS = (
    "configs/splits/evaluation_protocol_v1/calibration_v1.txt",
    "configs/splits/evaluation_protocol_v1/development_v1.txt",
    "configs/splits/evaluation_protocol_v1/held_out_test_v1.txt",
    "configs/splits/evaluation_protocol_v1/split_manifest.json",
    "configs/protocols/evaluation_protocol_v1.toml",
    "configs/baselines/b1_vision_calibration_template.toml",
)

CONTROLLER_OBSERVABLE_PREFIX = "controller_observable."
PRIVILEGED_DIAGNOSTIC_PREFIX = "privileged_diagnostic."


@dataclass(frozen=True)
class CalibrationDiagnosticRunResult:
    output_dir: Path
    requested_pairs: int
    completed_pairs: int
    invalid_pairs: int
    program_errors: int
    exit_code: int


def _finite_tuple(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Diagnostic state contains NaN or Inf")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _file_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def _round_zero_hashes(round_zero_dir: Path) -> dict[str, str]:
    if not round_zero_dir.is_dir():
        raise FileNotFoundError(f"Round 0 archive is missing: {round_zero_dir}")
    files = sorted(path for path in round_zero_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"Round 0 archive is empty: {round_zero_dir}")
    return {path.name: sha256_file(path) for path in files}


def _source_provenance(project_root: Path) -> dict[str, Any]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "benchmark",
        "configs",
        "controllers",
        "docs",
        "environments",
        "evaluation",
        "perception",
        "scenes",
        "scripts",
        "sensors",
        "tests",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    entries: dict[str, str] = {}
    for raw_path in completed.stdout.splitlines():
        relative = raw_path.strip().replace("\\", "/")
        if not relative:
            continue
        path = project_root / relative
        if path.is_file():
            entries[relative] = sha256_file(path)
    if not entries:
        raise RuntimeError("Traceable source snapshot contains no files")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "."],
        cwd=project_root,
        capture_output=True,
        check=True,
    ).stdout
    return {
        "traceability_kind": "content_addressed_worktree",
        "source_file_count": len(entries),
        "source_files": entries,
        "source_snapshot_sha256": _canonical_sha256(entries),
        "tracked_diff_sha256": _sha256_bytes(diff),
    }


class _DiagnosticRenderer:
    def __init__(self, env: Any) -> None:
        self.env = env
        self._renderer: mujoco.Renderer | None = None
        self._side_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._side_camera)
        self._side_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._side_camera.fixedcamid = -1
        self._side_camera.trackbodyid = -1
        self._side_camera.distance = 0.85
        self._side_camera.azimuth = 135.0
        self._side_camera.elevation = -17.0
        self._warmed_up = False

    def _get_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            config = self.env.config.camera
            self._renderer = mujoco.Renderer(
                self.env.model,
                height=config.height,
                width=config.width,
            )
        return self._renderer

    def capture(self, view: str) -> np.ndarray:
        renderer = self._get_renderer()
        if view == "overhead_rgb":
            renderer.update_scene(self.env.data, camera=self.env.overhead_camera_id)
        elif view == "diagnostic_side":
            object_position = np.asarray(
                self.env.data.xpos[self.env.object_body_id], dtype=float
            )
            tcp_position = np.asarray(
                self.env.data.site_xpos[self.env.tcp_site_id], dtype=float
            )
            self._side_camera.lookat[:] = 0.55 * object_position + 0.45 * tcp_position
            renderer.update_scene(self.env.data, camera=self._side_camera)
        else:
            raise ValueError(f"Unknown diagnostic view: {view}")
        image = renderer.render().copy()
        if not self._warmed_up:
            # The first hidden-WGL render can differ by one LSB in a handful of
            # pixels.  Warm the renderer without stepping simulation, then keep
            # the stable second image.
            renderer.update_scene(
                self.env.data,
                camera=(
                    self.env.overhead_camera_id
                    if view == "overhead_rgb"
                    else self._side_camera
                ),
            )
            image = renderer.render().copy()
            self._warmed_up = True
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected diagnostic RGB image: shape={image.shape}, dtype={image.dtype}"
            )
        return image

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class EpisodeDiagnosticRecorder:
    _TRACE_EVENTS = frozenset(
        {
            "episode_reset",
            "close_sample",
            "close_gripper_complete",
            "candidate_sample",
            "transport_sample",
            "confirmation_sample",
            "stage_transition",
            "release_sample",
            "episode_end",
        }
    )

    def __init__(
        self,
        *,
        env: Any,
        method: MethodSpec,
        seed: int,
        pair_id: str,
        execution_index: int,
        output_dir: Path,
        visualization_enabled: bool,
    ) -> None:
        self.env = env
        self.method = method
        self.seed = int(seed)
        self.pair_id = pair_id
        self.execution_index = int(execution_index)
        self.output_dir = output_dir
        self.visualization_enabled = visualization_enabled
        self.trace_rows: list[dict[str, Any]] = []
        self.frame_records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.result: Any | None = None
        self.fingerprint: Any | None = None
        self.initial_robot_state: tuple[float, ...] | None = None
        self.external_state_metrics: Any | None = None
        self._renderer: _DiagnosticRenderer | None = None
        self._captured_milestones: set[str] = set()
        self._initial_object_position: np.ndarray | None = None
        self._initial_object_tilt: float | None = None
        self._candidate_object_tcp_offset: np.ndarray | None = None
        self._diagnostic_grasp_window_active = False
        self._diagnostic_window_max_object_z: float | None = None
        self._max_relative_slip = 0.0
        self._max_tilt_change = 0.0
        self._max_drop_from_peak = 0.0
        self._bilateral_steps = 0
        self._confirmation_bilateral_steps = 0
        self._confirmation_max_bilateral_steps = 0
        self._confirmation_bilateral_loss_events = 0
        self._last_confirmation_bilateral: bool | None = None
        self._last_sensor_timestamp: float | None = None

    def _truth(self, snapshot: B1DiagnosticSnapshot) -> dict[str, Any]:
        object_position = np.asarray(
            self.env.data.xpos[self.env.object_body_id], dtype=float
        ).copy()
        object_quaternion = np.asarray(
            self.env.data.xquat[self.env.object_body_id], dtype=float
        ).copy()
        object_rotation = np.asarray(
            self.env.data.xmat[self.env.object_body_id], dtype=float
        ).reshape(3, 3)
        object_velocity = np.asarray(
            self.env.data.qvel[
                self.env.object_dof_address : self.env.object_dof_address + 6
            ],
            dtype=float,
        ).copy()
        values = np.concatenate(
            (object_position, object_quaternion, object_rotation.reshape(-1), object_velocity)
        )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("Privileged diagnostic truth contains NaN or Inf")

        tilt = float(
            math.acos(float(np.clip(object_rotation[2, 2], -1.0, 1.0)))
        )
        if self._initial_object_position is None:
            self._initial_object_position = object_position.copy()
            self._initial_object_tilt = tilt
        assert self._initial_object_position is not None
        assert self._initial_object_tilt is not None
        lift_height = float(object_position[2] - self._initial_object_position[2])
        tilt_change = abs(tilt - self._initial_object_tilt)

        tcp_position = np.asarray(snapshot.tcp_position, dtype=float)
        object_tcp_offset = object_position - tcp_position
        if (
            snapshot.event == "candidate_sample"
            and snapshot.grasp_state == "grasp_candidate"
            and self._candidate_object_tcp_offset is None
        ):
            self._candidate_object_tcp_offset = object_tcp_offset.copy()
            self._diagnostic_grasp_window_active = True
            self._diagnostic_window_max_object_z = float(object_position[2])
        relative_slip = 0.0
        drop_from_peak = 0.0
        if (
            self._diagnostic_grasp_window_active
            and self._candidate_object_tcp_offset is not None
        ):
            relative_slip = float(
                np.linalg.norm(object_tcp_offset - self._candidate_object_tcp_offset)
            )
            self._max_relative_slip = max(self._max_relative_slip, relative_slip)
            self._max_tilt_change = max(self._max_tilt_change, tilt_change)
            if self._diagnostic_window_max_object_z is None:
                self._diagnostic_window_max_object_z = float(object_position[2])
            self._diagnostic_window_max_object_z = max(
                self._diagnostic_window_max_object_z,
                float(object_position[2]),
            )
            drop_from_peak = float(
                self._diagnostic_window_max_object_z - object_position[2]
            )
            self._max_drop_from_peak = max(self._max_drop_from_peak, drop_from_peak)

        return {
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_position_x": float(object_position[0]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_position_y": float(object_position[1]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_position_z": float(object_position[2]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_quaternion_w": float(object_quaternion[0]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_quaternion_x": float(object_quaternion[1]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_quaternion_y": float(object_quaternion[2]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_quaternion_z": float(object_quaternion[3]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_tilt_rad": tilt,
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_tilt_change_rad": tilt_change,
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_linear_velocity_x": float(object_velocity[0]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_linear_velocity_y": float(object_velocity[1]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_linear_velocity_z": float(object_velocity[2]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_angular_velocity_x": float(object_velocity[3]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_angular_velocity_y": float(object_velocity[4]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_angular_velocity_z": float(object_velocity[5]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_lift_height": lift_height,
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_drop_from_peak": drop_from_peak,
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_tcp_offset_x": float(object_tcp_offset[0]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_tcp_offset_y": float(object_tcp_offset[1]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_tcp_offset_z": float(object_tcp_offset[2]),
            f"{PRIVILEGED_DIAGNOSTIC_PREFIX}relative_slip_from_candidate": relative_slip,
        }

    def _update_contact_holds(self, snapshot: B1DiagnosticSnapshot) -> tuple[int, int]:
        is_sensor_sample = snapshot.event in {
            "close_gripper_complete",
            "candidate_sample",
            "transport_sample",
            "confirmation_sample",
        }
        if not is_sensor_sample or snapshot.bilateral_contact is None:
            return self._bilateral_steps, self._confirmation_bilateral_steps
        if self._last_sensor_timestamp == snapshot.simulation_time:
            return self._bilateral_steps, self._confirmation_bilateral_steps
        self._last_sensor_timestamp = snapshot.simulation_time
        self._bilateral_steps = self._bilateral_steps + 1 if snapshot.bilateral_contact else 0
        if snapshot.stage == "grasp_confirmation":
            if self._last_confirmation_bilateral and not snapshot.bilateral_contact:
                self._confirmation_bilateral_loss_events += 1
            self._last_confirmation_bilateral = snapshot.bilateral_contact
            self._confirmation_bilateral_steps = (
                self._confirmation_bilateral_steps + 1
                if snapshot.bilateral_contact
                else 0
            )
            self._confirmation_max_bilateral_steps = max(
                self._confirmation_max_bilateral_steps,
                self._confirmation_bilateral_steps,
            )
        return self._bilateral_steps, self._confirmation_bilateral_steps

    def _row(self, snapshot: B1DiagnosticSnapshot) -> dict[str, Any]:
        bilateral_steps, confirmation_bilateral_steps = self._update_contact_holds(
            snapshot
        )
        timestep = float(self.env.model.opt.timestep) * int(
            self.env.config.simulation.frame_skip
        )
        aperture = snapshot.gripper_aperture
        if aperture is None:
            aperture = float(sum(snapshot.finger_positions))
        row: dict[str, Any] = {
            "seed": self.seed,
            "method": self.method.method_id,
            "pair_id": self.pair_id,
            "execution_index": self.execution_index,
            "event": snapshot.event,
            "simulation_time": snapshot.simulation_time,
            "stage": snapshot.stage,
            "next_stage": snapshot.next_stage,
            "failure_reason": snapshot.failure_reason,
            f"{CONTROLLER_OBSERVABLE_PREFIX}grasp_state": snapshot.grasp_state,
            f"{CONTROLLER_OBSERVABLE_PREFIX}gripper_aperture": aperture,
            f"{CONTROLLER_OBSERVABLE_PREFIX}gripper_aperture_velocity": snapshot.gripper_aperture_velocity,
            f"{CONTROLLER_OBSERVABLE_PREFIX}left_finger_position": snapshot.left_finger_position,
            f"{CONTROLLER_OBSERVABLE_PREFIX}right_finger_position": snapshot.right_finger_position,
            f"{CONTROLLER_OBSERVABLE_PREFIX}commanded_state": snapshot.commanded_state,
            f"{CONTROLLER_OBSERVABLE_PREFIX}left_contact": snapshot.left_contact,
            f"{CONTROLLER_OBSERVABLE_PREFIX}right_contact": snapshot.right_contact,
            f"{CONTROLLER_OBSERVABLE_PREFIX}bilateral_contact": snapshot.bilateral_contact,
            f"{CONTROLLER_OBSERVABLE_PREFIX}bilateral_contact_duration": snapshot.bilateral_contact_duration,
            f"{CONTROLLER_OBSERVABLE_PREFIX}bilateral_contact_hold_steps": bilateral_steps,
            f"{CONTROLLER_OBSERVABLE_PREFIX}bilateral_contact_hold_time": bilateral_steps * timestep,
            f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_bilateral_hold_steps": confirmation_bilateral_steps,
            f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_bilateral_hold_time": confirmation_bilateral_steps * timestep,
            f"{CONTROLLER_OBSERVABLE_PREFIX}candidate_aperture": snapshot.candidate_aperture,
            f"{CONTROLLER_OBSERVABLE_PREFIX}current_aperture_drop": snapshot.aperture_drop,
            f"{CONTROLLER_OBSERVABLE_PREFIX}commanded_closing_predicate": snapshot.commanded_closing_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}minimum_aperture_predicate": snapshot.minimum_aperture_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}contact_predicate": snapshot.contact_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}lift_predicate": snapshot.lift_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}aperture_retention_predicate": snapshot.aperture_retention_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}collision_free_predicate": snapshot.collision_free_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_combined_predicate": snapshot.combined_predicate,
            f"{CONTROLLER_OBSERVABLE_PREFIX}candidate_hold_steps": snapshot.candidate_hold_steps,
            f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_hold_steps": snapshot.confirmation_hold_steps,
            f"{CONTROLLER_OBSERVABLE_PREFIX}contact_loss_hold_steps": snapshot.contact_loss_hold_steps,
            f"{CONTROLLER_OBSERVABLE_PREFIX}contact_loss_event_count": snapshot.contact_loss_event_count,
            f"{CONTROLLER_OBSERVABLE_PREFIX}trial_lift_completed": snapshot.trial_lift_completed,
            f"{CONTROLLER_OBSERVABLE_PREFIX}robot_table_collision": snapshot.robot_table_collision,
            f"{CONTROLLER_OBSERVABLE_PREFIX}tcp_position_x": snapshot.tcp_position[0],
            f"{CONTROLLER_OBSERVABLE_PREFIX}tcp_position_y": snapshot.tcp_position[1],
            f"{CONTROLLER_OBSERVABLE_PREFIX}tcp_position_z": snapshot.tcp_position[2],
            f"{CONTROLLER_OBSERVABLE_PREFIX}finger_position_1": snapshot.finger_positions[0],
            f"{CONTROLLER_OBSERVABLE_PREFIX}finger_position_2": snapshot.finger_positions[1],
        }
        row.update(self._truth(snapshot))
        return row

    def _capture_milestones(
        self, snapshot: B1DiagnosticSnapshot, milestones: Sequence[str]
    ) -> None:
        pending = [name for name in milestones if name not in self._captured_milestones]
        if not pending or not self.visualization_enabled:
            return
        if self._renderer is None:
            self._renderer = _DiagnosticRenderer(self.env)
        base = (
            self.output_dir
            / "frames"
            / f"seed_{self.seed}"
            / self.method.method_id
        )
        for view in ("overhead_rgb", "diagnostic_side"):
            image = self._renderer.capture(view)
            for milestone in pending:
                path = base / f"{milestone}__{view}.png"
                save_rgb_png(path, image)
                self.frame_records.append(
                    {
                        "seed": self.seed,
                        "method": self.method.method_id,
                        "milestone": milestone,
                        "view": view,
                        "simulation_time": snapshot.simulation_time,
                        "path": path.relative_to(self.output_dir).as_posix(),
                    }
                )
        self._captured_milestones.update(pending)

    def observe(self, snapshot: B1DiagnosticSnapshot) -> None:
        try:
            if not isinstance(snapshot, B1DiagnosticSnapshot):
                raise TypeError("Controller supplied a non-diagnostic snapshot")
            if snapshot.event not in self._TRACE_EVENTS:
                raise ValueError(f"Unexpected diagnostic event: {snapshot.event}")
            row = self._row(snapshot)
            if (
                snapshot.event == "stage_transition"
                and snapshot.stage == "grasp_confirmation"
                and snapshot.next_stage == "transfer"
            ) or (
                snapshot.event == "episode_end"
                and snapshot.stage == "grasp_confirmation"
            ):
                self._diagnostic_grasp_window_active = False
            keep_trace = not (
                snapshot.event == "transport_sample"
                and snapshot.stage not in {"trial_lift", "grasp_confirmation"}
            )
            if keep_trace:
                self.trace_rows.append(row)

            milestones: list[str] = []
            if snapshot.event == "close_gripper_complete":
                milestones.append("close_gripper_end")
            if (
                snapshot.event == "candidate_sample"
                and snapshot.grasp_state == "grasp_candidate"
            ):
                milestones.append("grasp_candidate")
            if (
                snapshot.event == "stage_transition"
                and snapshot.stage == "trial_lift"
                and snapshot.next_stage == "grasp_confirmation"
            ):
                milestones.append("trial_lift_end")
            if (
                snapshot.event == "confirmation_sample"
                and snapshot.grasp_state == "grasp_confirmed"
            ):
                milestones.append("confirmation_success")
            if snapshot.event == "episode_end":
                if snapshot.stage == "grasp_confirmation":
                    milestones.append("confirmation_failure")
                milestones.append("episode_terminal")
            self._capture_milestones(snapshot, milestones)
        except Exception:
            self.errors.append(traceback.format_exc())

    def finish(
        self,
        *,
        result: Any,
        fingerprint: Any,
        initial_robot_state: tuple[float, ...] | None,
        external_state_metrics: Any,
        provider_call_count: int | None = None,
    ) -> None:
        self.result = result
        self.fingerprint = fingerprint
        self.initial_robot_state = initial_robot_state
        self.external_state_metrics = external_state_metrics
        if self.errors:
            raise RuntimeError(
                "Diagnostic recorder failed without changing controller behavior:\n"
                + "\n".join(self.errors)
            )

    @staticmethod
    def _last_value(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
        for row in reversed(rows):
            value = row.get(field)
            if value is not None:
                return value
        return None

    def summary(self) -> dict[str, Any]:
        if self.result is None or self.fingerprint is None:
            raise RuntimeError("Diagnostic episode did not finish")
        candidate_rows = [
            row
            for row in self.trace_rows
            if row["event"] == "candidate_sample"
            and row.get(f"{CONTROLLER_OBSERVABLE_PREFIX}grasp_state")
            == "grasp_candidate"
        ]
        trial_end_rows = [
            row
            for row in self.trace_rows
            if row["event"] == "stage_transition"
            and row["stage"] == "trial_lift"
            and row["next_stage"] == "grasp_confirmation"
        ]
        confirmation_rows = [
            row for row in self.trace_rows if row["event"] == "confirmation_sample"
        ]
        candidate_aperture = (
            None
            if not candidate_rows
            else candidate_rows[0].get(
                f"{CONTROLLER_OBSERVABLE_PREFIX}candidate_aperture"
            )
        )
        trial_aperture = (
            None
            if not trial_end_rows
            else trial_end_rows[-1].get(
                f"{CONTROLLER_OBSERVABLE_PREFIX}gripper_aperture"
            )
        )
        confirmation_aperture = self._last_value(
            confirmation_rows,
            f"{CONTROLLER_OBSERVABLE_PREFIX}gripper_aperture",
        )
        confirmation_drops = [
            float(value)
            for value in (
                row.get(f"{CONTROLLER_OBSERVABLE_PREFIX}current_aperture_drop")
                for row in confirmation_rows
            )
            if value is not None
        ]
        final_confirmation = confirmation_rows[-1] if confirmation_rows else {}
        predicate_fields = {
            "lift_predicate": f"{CONTROLLER_OBSERVABLE_PREFIX}lift_predicate",
            "contact_predicate": f"{CONTROLLER_OBSERVABLE_PREFIX}contact_predicate",
            "minimum_aperture_predicate": f"{CONTROLLER_OBSERVABLE_PREFIX}minimum_aperture_predicate",
            "aperture_retention_predicate": f"{CONTROLLER_OBSERVABLE_PREFIX}aperture_retention_predicate",
            "collision_free_predicate": f"{CONTROLLER_OBSERVABLE_PREFIX}collision_free_predicate",
        }
        final_false = [
            name
            for name, field in predicate_fields.items()
            if final_confirmation.get(field) is False
        ]
        maximum_hold = max(
            (
                int(
                    row.get(
                        f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_hold_steps"
                    )
                    or 0
                )
                for row in confirmation_rows
            ),
            default=0,
        )
        final_hold = int(
            final_confirmation.get(
                f"{CONTROLLER_OBSERVABLE_PREFIX}confirmation_hold_steps"
            )
            or 0
        )
        diagnostic_window_rows = [
            row
            for row in self.trace_rows
            if row["stage"]
            in {"grasp_candidate_check", "trial_lift", "grasp_confirmation"}
        ]
        lift_values = [
            float(row[f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_lift_height"])
            for row in diagnostic_window_rows
        ]
        object_lift_height = max(lift_values, default=0.0)
        trial_object_lift_height = (
            None
            if not trial_end_rows
            else trial_end_rows[-1].get(
                f"{PRIVILEGED_DIAGNOSTIC_PREFIX}object_lift_height"
            )
        )
        confirmation_duration = float(
            (self.result.stage_durations or {}).get("grasp_confirmation", 0.0)
        )
        timestep = float(self.env.model.opt.timestep) * int(
            self.env.config.simulation.frame_skip
        )
        aperture_drop = (
            None
            if candidate_aperture is None or confirmation_aperture is None
            else float(candidate_aperture) - float(confirmation_aperture)
        )
        slip_detected = bool(
            self._max_relative_slip >= 0.003
            or self._max_drop_from_peak >= 0.003
            or self._max_tilt_change >= math.radians(5.0)
        )
        return {
            "seed": self.seed,
            "method": self.method.method_id,
            "pair_id": self.pair_id,
            "execution_index": self.execution_index,
            "episode_fingerprint": self.fingerprint.digest,
            "final_stage": self.result.final_stage,
            "failure_reason": self.result.failure_reason,
            "controller_reported_success": self.result.controller_reported_success,
            "privileged_ground_truth_success": self.result.privileged_ground_truth_success,
            "collision_count": self.result.collision_count,
            "simulation_time": self.result.simulation_time,
            "candidate_aperture": candidate_aperture,
            "aperture_at_trial_lift_completion": trial_aperture,
            "aperture_at_confirmation_end": confirmation_aperture,
            "aperture_drop_from_candidate": aperture_drop,
            "maximum_aperture_drop_during_confirmation": (
                max(confirmation_drops) if confirmation_drops else None
            ),
            "confirmation_elapsed_time": confirmation_duration,
            "confirmation_held_steps_maximum": maximum_hold,
            "final_confirmation_held_steps": final_hold,
            "final_false_predicates": final_false,
            "final_failure_predicate": (
                final_false[0] if len(final_false) == 1 else final_false
            ),
            "bilateral_contact_ever_during_confirmation": any(
                row.get(f"{CONTROLLER_OBSERVABLE_PREFIX}bilateral_contact") is True
                for row in confirmation_rows
            ),
            "longest_continuous_bilateral_contact_steps": self._confirmation_max_bilateral_steps,
            "longest_continuous_bilateral_contact_time": self._confirmation_max_bilateral_steps
            * timestep,
            "confirmation_bilateral_contact_loss_events": self._confirmation_bilateral_loss_events,
            "contact_loss_event_count": self.result.contact_loss_event_count,
            "object_lift_height": object_lift_height,
            "object_lift_height_at_trial_completion": trial_object_lift_height,
            "maximum_relative_object_tcp_slip": self._max_relative_slip,
            "maximum_object_drop_from_peak": self._max_drop_from_peak,
            "maximum_object_tilt_change_rad": self._max_tilt_change,
            "object_drop_or_slip_diagnostic": slip_detected,
            "trial_lift_completed": self.result.trial_lift_completed,
            "confirmed": self.result.grasp_confirmed,
            "initial_object_position": self.result.initial_object_position,
            "pregrasp_corrected_object_position": self.result.pregrasp_corrected_object_position,
            "trace_row_count": len(self.trace_rows),
            "frame_count": len(self.frame_records),
            "frames": [dict(record) for record in self.frame_records],
        }

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class CalibrationDiagnosticSession:
    def __init__(self, output_dir: Path, *, visualization_enabled: bool) -> None:
        self.output_dir = output_dir
        self.visualization_enabled = visualization_enabled
        self.recorders: list[EpisodeDiagnosticRecorder] = []

    def start_episode(
        self,
        *,
        env: Any,
        method: MethodSpec,
        seed: int,
        pair_id: str,
        execution_index: int,
    ) -> EpisodeDiagnosticRecorder:
        recorder = EpisodeDiagnosticRecorder(
            env=env,
            method=method,
            seed=seed,
            pair_id=pair_id,
            execution_index=execution_index,
            output_dir=self.output_dir,
            visualization_enabled=self.visualization_enabled,
        )
        self.recorders.append(recorder)
        return recorder

    def trace_rows(self) -> list[dict[str, Any]]:
        return [row for recorder in self.recorders for row in recorder.trace_rows]

    def episode_summaries(self) -> list[dict[str, Any]]:
        return [recorder.summary() for recorder in self.recorders]

    def frame_records(self) -> list[dict[str, Any]]:
        return [record for recorder in self.recorders for record in recorder.frame_records]


def validate_diagnostic_request(
    *,
    protocol: ProtocolConfig,
    config_path: Path,
    seeds_path: Path,
    method_ids: Sequence[str],
) -> tuple[EnvConfig, tuple[int, ...], tuple[MethodSpec, ...]]:
    expected_baseline = (
        PROJECT_ROOT
        / str(protocol.raw["calibration"]["baseline_template_path"])
    ).resolve()
    if config_path.resolve() != expected_baseline:
        raise ValueError(
            "Round 0.5 requires the registered B1 calibration template exactly; "
            f"expected {expected_baseline}, got {config_path.resolve()}"
        )
    seeds = tuple(load_seeds(seeds_path))
    if seeds != REQUIRED_DIAGNOSTIC_SEEDS:
        raise ValueError(
            "Round 0.5 diagnostic seeds must be exactly, in order, "
            f"{list(REQUIRED_DIAGNOSTIC_SEEDS)}; got {list(seeds)}"
        )
    calibration_seeds = set(load_seeds(protocol.splits["calibration"].path))
    if not set(seeds).issubset(calibration_seeds):
        raise ValueError("Every diagnostic seed must already belong to Calibration")
    if tuple(method_ids) != REQUIRED_METHODS:
        raise ValueError(
            f"Round 0.5 methods must be exactly {list(REQUIRED_METHODS)} in order"
        )
    methods = resolve_methods(list(method_ids))
    config, overrides = _effective_config(config_path)
    if overrides:
        raise ValueError(f"Baseline required behavior-changing overrides: {overrides}")
    validate_baseline_compatibility(protocol, config)
    assert_static_fairness(methods, config)
    return config, seeds, methods


def _read_round_zero_rows(round_zero_dir: Path) -> dict[tuple[int, str], dict[str, str]]:
    with (round_zero_dir / "episodes.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    return {(int(row["seed"]), row["method_id"]): row for row in rows}


def _bool_cell(value: str) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean CSV cell: {value!r}")


def _round_zero_replay_comparison(
    episode_summaries: Sequence[Mapping[str, Any]], round_zero_dir: Path
) -> dict[str, Any]:
    archived = _read_round_zero_rows(round_zero_dir)
    comparisons: list[dict[str, Any]] = []
    for episode in episode_summaries:
        key = (int(episode["seed"]), str(episode["method"]))
        expected = archived.get(key)
        if expected is None:
            raise ValueError(f"Round 0 archive lacks diagnostic reference {key}")
        checks = {
            "episode_fingerprint": (
                episode["episode_fingerprint"] == expected["episode_fingerprint"]
            ),
            "final_stage": episode["final_stage"] == expected["final_stage"],
            "failure_reason": (episode["failure_reason"] or "")
            == expected["failure_reason"],
            "controller_reported_success": episode["controller_reported_success"]
            == _bool_cell(expected["controller_reported_success"]),
            "privileged_ground_truth_success": episode[
                "privileged_ground_truth_success"
            ]
            == _bool_cell(expected["privileged_ground_truth_success"]),
            "collision_count": int(episode["collision_count"])
            == int(expected["collision_count"]),
            "simulation_time": math.isclose(
                float(episode["simulation_time"]),
                float(expected["simulation_time"]),
                rel_tol=0.0,
                abs_tol=0.0020000001,
            ),
        }
        comparisons.append(
            {
                "seed": key[0],
                "method": key[1],
                "checks": checks,
                "all_behavior_fields_match": all(checks.values()),
                "archived_simulation_time": float(expected["simulation_time"]),
                "diagnostic_simulation_time": float(episode["simulation_time"]),
            }
        )
    return {
        "reference": "Round 0 archived episodes.csv",
        "comparisons": comparisons,
        "all_episodes_match": all(
            comparison["all_behavior_fields_match"] for comparison in comparisons
        ),
    }


def _target_difference_by_pair(
    episode_summaries: Sequence[Mapping[str, Any]], seed: int
) -> float | None:
    by_method = {
        str(row["method"]): row
        for row in episode_summaries
        if int(row["seed"]) == seed
    }
    if set(by_method) != set(REQUIRED_METHODS):
        return None
    left = by_method["b0_oracle"].get("pregrasp_corrected_object_position")
    right = by_method["b1_vision"].get("pregrasp_corrected_object_position")
    if left is None or right is None:
        return None
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _build_summary(
    *,
    manifest: Mapping[str, Any],
    episode_summaries: list[dict[str, Any]],
    replay: Mapping[str, Any],
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
    round_zero_before: Mapping[str, str],
    round_zero_after: Mapping[str, str],
    frame_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    controller_fields = sorted(
        {
            key
            for recorder in episode_summaries
            for key in recorder
            if key.startswith(CONTROLLER_OBSERVABLE_PREFIX)
        }
    )
    # The independent trace schema, rather than the episode summary, owns these lists.
    controller_trace_fields = [
        "grasp_state",
        "gripper_aperture",
        "gripper_aperture_velocity",
        "left_finger_position",
        "right_finger_position",
        "commanded_state",
        "left_contact",
        "right_contact",
        "bilateral_contact",
        "bilateral_contact_duration",
        "bilateral_contact_hold_steps",
        "candidate_aperture",
        "current_aperture_drop",
        "commanded_closing_predicate",
        "minimum_aperture_predicate",
        "contact_predicate",
        "lift_predicate",
        "aperture_retention_predicate",
        "collision_free_predicate",
        "confirmation_combined_predicate",
        "candidate_hold_steps",
        "confirmation_hold_steps",
        "contact_loss_hold_steps",
        "contact_loss_event_count",
        "trial_lift_completed",
        "robot_table_collision",
        "tcp_position",
        "finger_positions",
    ]
    privileged_trace_fields = [
        "object_position",
        "object_quaternion",
        "object_tilt_rad",
        "object_tilt_change_rad",
        "object_linear_velocity",
        "object_angular_velocity",
        "object_lift_height",
        "object_drop_from_peak",
        "object_tcp_offset",
        "relative_slip_from_candidate",
    ]
    return {
        "diagnostic_telemetry_schema_version": DIAGNOSTIC_TELEMETRY_SCHEMA_VERSION,
        "run_kind": RUN_KIND,
        "diagnostic_only": True,
        "excluded_from_formal_calibration_metrics": True,
        "production_metrics_generated": False,
        "formal_split_modified": False,
        "manual_assessment": {
            "assessment_kind": "structured_evidence_review",
            "status": "evidence_generated_pending_final_structured_review",
            "round_1_decision": None,
            "parameter_change_recommended_now": None,
        },
        "run": dict(manifest),
        "completion": {
            "requested_pairs": len(REQUIRED_DIAGNOSTIC_SEEDS),
            "requested_episodes": len(REQUIRED_DIAGNOSTIC_SEEDS)
            * len(REQUIRED_METHODS),
            "completed_pairs": manifest["completed_pairs"],
            "completed_episodes": len(episode_summaries),
            "invalid_pairs": manifest["invalid_pairs"],
            "program_errors": manifest["program_errors"],
        },
        "field_boundaries": {
            "controller_observable_prefix": CONTROLLER_OBSERVABLE_PREFIX,
            "controller_observable_trace_fields": controller_trace_fields,
            "privileged_diagnostic_prefix": PRIVILEGED_DIAGNOSTIC_PREFIX,
            "privileged_diagnostic_trace_fields": privileged_trace_fields,
            "unused_summary_discovery": controller_fields,
            "privileged_data_entered_controller": False,
        },
        "behavior_invariance": replay,
        "protected_hashes": {
            "formal_before": dict(protected_before),
            "formal_after": dict(protected_after),
            "formal_unchanged": protected_before == protected_after,
            "round_0_before": dict(round_zero_before),
            "round_0_after": dict(round_zero_after),
            "round_0_unchanged": round_zero_before == round_zero_after,
        },
        "target_position_pair_differences": {
            str(seed): _target_difference_by_pair(episode_summaries, seed)
            for seed in REQUIRED_DIAGNOSTIC_SEEDS
        },
        "episodes": episode_summaries,
        "visualization": {
            "enabled": bool(manifest["visualization_enabled"]),
            "video_generated": False,
            "video_limitation": (
                "No reliable MP4 encoder is installed; deterministic PNG keyframes "
                "were generated instead."
            ),
            "frame_count": len(frame_records),
            "frames": [dict(record) for record in frame_records],
        },
        "analysis_boundary": (
            "This diagnostic replay is not a Calibration sample, is excluded from "
            "production metrics, and does not select or modify parameters."
        ),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_report(summary: Mapping[str, Any]) -> str:
    run = summary["run"]
    lines = [
        "# B1 Round 0.5 定向诊断报告",
        "",
        "## 范围与身份",
        "",
        f"- run kind：{run['run_kind']}",
        f"- protocol：{run['protocol_id']} / {run['protocol_version']}",
        f"- config SHA-256：{run['config_sha256']}",
        f"- seeds：{_fmt(run['seeds'])}",
        f"- methods：{_fmt(run['methods'])}",
        "- 本运行只引用既有 Calibration seeds，不是新的 Calibration 统计样本。",
        "- 未生成 production metrics，未修改 split manifest，未选择或修改参数。",
        "",
        "## Confirmation 真实 predicates",
        "",
        "连续 15 个 confirmation 更新样本必须同时满足：",
        "",
        "1. trial_lift_completed = true（TCP trial-lift motion 已完成）；",
        "2. 经 3 步去抖的 bilateral contact = true；",
        "3. gripper aperture > 0.008 m；",
        "4. candidate aperture - current aperture < 0.003 m；",
        "5. robot-table collision = false。",
        "",
        "任一项为 false 时 confirmation hold 清零；timeout 为 4.0 s。",
        "",
        "## 行为不变性与完整性",
        "",
        f"- 8 个 replay 与 Round 0 稳定行为字段全部一致：{_fmt(summary['behavior_invariance']['all_episodes_match'])}",
        f"- 正式 config/split 哈希不变：{_fmt(summary['protected_hashes']['formal_unchanged'])}",
        f"- Round 0 全目录文件哈希不变：{_fmt(summary['protected_hashes']['round_0_unchanged'])}",
        f"- privileged 数据进入控制器：{_fmt(summary['field_boundaries']['privileged_data_entered_controller'])}",
        "",
        "## Episode 诊断",
        "",
        "| seed | method | result | candidate (m) | confirmation end (m) | drop (m) | max drop (m) | hold max/final | final false predicates | bilateral max steps | object/TCP slip (m) | tilt change (rad) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for episode in summary["episodes"]:
        result = episode["failure_reason"] or "completed"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(episode["seed"]),
                    str(episode["method"]),
                    result,
                    _fmt(episode["candidate_aperture"]),
                    _fmt(episode["aperture_at_confirmation_end"]),
                    _fmt(episode["aperture_drop_from_candidate"]),
                    _fmt(episode["maximum_aperture_drop_during_confirmation"]),
                    f"{episode['confirmation_held_steps_maximum']}/{episode['final_confirmation_held_steps']}",
                    _fmt(episode["final_false_predicates"]),
                    str(episode["longest_continuous_bilateral_contact_steps"]),
                    _fmt(episode["maximum_relative_object_tcp_slip"]),
                    _fmt(episode["maximum_object_tilt_change_rad"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 可视化证据",
            "",
            f"- PNG 数：{summary['visualization']['frame_count']}",
            "- 每个 episode 保存 close end、candidate、trial lift end、confirmation success/failure、terminal 的 overhead 与斜侧诊断视角。",
            f"- 视频边界：{summary['visualization']['video_limitation']}",
            "",
            "## Structured evidence review",
            "",
            "本文件初次由运行器生成时只汇总结构化证据；A/B/C 决策须在读取 trace 与 PNG 后写入同目录的 structured review。",
            "",
            "本报告不修改任何 B1 参数，不运行 Round 1，也不宣布冻结 B1。",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_structured_review(
    output_dir: str | Path,
    review_file: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    review_path = Path(review_file).expanduser().resolve()
    summary_path = output_path / "diagnostic_summary.json"
    report_path = output_path / "round_0_5_report.md"
    manifest_path = output_path / "run_manifest.json"
    if not output_path.is_dir():
        raise FileNotFoundError(f"Diagnostic output is missing: {output_path}")
    for path in (summary_path, report_path, manifest_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(f"Structured review input is missing: {path}")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("Structured review must be a JSON object")
    if review.get("assessment_kind") != "structured_evidence_review":
        raise ValueError(
            "assessment_kind must be exactly 'structured_evidence_review'"
        )
    decision = review.get("round_1_decision")
    if decision not in {"A", "B", "C"}:
        raise ValueError("round_1_decision must be exactly A, B, or C")
    adjustments = review.get("round_1_parameter_adjustments", [])
    if not isinstance(adjustments, list) or len(adjustments) > 2:
        raise ValueError("round_1_parameter_adjustments must contain at most two items")
    if decision != "A" and adjustments:
        raise ValueError("Only decision A may propose Round 1 parameter candidates")
    if review.get("b1_parameters_modified") is not False:
        raise ValueError("Structured review must state b1_parameters_modified=false")
    if review.get("round_1_executed") is not False:
        raise ValueError("Structured review must state round_1_executed=false")
    if review.get("b1_frozen") is not False:
        raise ValueError("Structured review must state b1_frozen=false")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_kind") != RUN_KIND:
        raise ValueError("Output is not a Round 0.5 diagnostic replay")
    summary["manual_assessment"] = review
    write_json(summary_path, summary)

    report = report_path.read_text(encoding="utf-8")
    report = report.split("## Final structured evidence review", 1)[0].rstrip()
    report += "\n\n"
    report += "\n".join(
        [
            "## Final structured evidence review",
            "",
            f"- Round 1 决策：{decision}",
            f"- 是否建议当前修改参数：{_fmt(review.get('parameter_change_recommended_now'))}",
            f"- 明确参数问题：{_fmt(review.get('explicit_parameter_issue'))}",
            f"- 几何或算法问题：{_fmt(review.get('geometry_or_algorithm_issue'))}",
            f"- 证据仍不足：{_fmt(review.get('evidence_gap'))}",
            f"- 理由：{review.get('rationale', '—')}",
            f"- B2 方向：{review.get('next_b2_direction', '—')}",
            f"- Round 1 参数候选：{_fmt(adjustments)}",
            "",
            "### 逐 seed 结论",
            "",
            "```json",
            json.dumps(
                review.get("per_seed_findings", {}),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "本任务只完成 Round 0.5 定向诊断。",
            "未修改 B1 参数。",
            "未运行 Round 1。",
            "未冻结 B1。",
            "Round 1 是否执行由用户在审查本报告后决定。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["structured_review_finalized"] = True
    manifest["assessment_kind"] = "structured_evidence_review"
    manifest["round_1_decision"] = decision
    artifact_hashes = {
        path.relative_to(output_path).as_posix(): sha256_file(path)
        for path in sorted(output_path.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest["artifact_hashes"] = artifact_hashes
    manifest["artifact_set_sha256"] = _canonical_sha256(artifact_hashes)
    write_json(manifest_path, manifest)
    return summary


def run_calibration_diagnostics(
    *,
    protocol: ProtocolConfig,
    config_path: str | Path,
    seeds_file: str | Path,
    output_dir: str | Path,
    round_zero_dir: str | Path,
    method_ids: Sequence[str] = REQUIRED_METHODS,
    diagnostics_enabled: bool,
    visualization_enabled: bool,
    require_traceable_source: bool,
    command: Sequence[str] | None = None,
    fingerprint_atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> CalibrationDiagnosticRunResult:
    if not diagnostics_enabled:
        raise ValueError("Round 0.5 requires --diagnostics-enabled")
    if not visualization_enabled:
        raise ValueError("Round 0.5 requires --visualization-artifacts-enabled")
    if not require_traceable_source:
        raise ValueError("Round 0.5 requires --require-traceable-source")

    config_path = Path(config_path).expanduser().resolve()
    seeds_path = Path(seeds_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    round_zero_path = Path(round_zero_dir).expanduser().resolve()
    config, seeds, methods = validate_diagnostic_request(
        protocol=protocol,
        config_path=config_path,
        seeds_path=seeds_path,
        method_ids=method_ids,
    )
    if sha256_file(config_path) != sha256_file(round_zero_path / "config_snapshot.toml"):
        raise ValueError("Baseline config no longer matches the Round 0 snapshot")
    if protocol.sha256 != sha256_file(round_zero_path / "protocol_snapshot.toml"):
        raise ValueError("Protocol no longer matches the Round 0 snapshot")

    protected_paths = tuple((PROJECT_ROOT / path).resolve() for path in PROTECTED_FORMAL_PATHS)
    protected_before = _file_hashes(protected_paths)
    round_zero_before = _round_zero_hashes(round_zero_path)
    source = _source_provenance(PROJECT_ROOT)
    repository = repository_metadata(PROJECT_ROOT)
    if not source.get("source_snapshot_sha256"):
        raise BenchmarkRunError("Source traceability snapshot could not be established")

    _prepare_output_dir(output_path, overwrite=False)
    shutil.copyfile(config_path, output_path / "diagnostic_config_snapshot.toml")
    shutil.copyfile(protocol.path, output_path / "protocol_snapshot.toml")
    write_json(output_path / "source_provenance.json", source)
    write_json(
        output_path / "seeds.json",
        {
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "diagnostic_only": True,
            "formal_split": False,
            "reference_split": "calibration",
            "used_for_formal_success_rate": False,
        },
    )
    logger, log_handler = _logger_for(output_path)
    session = CalibrationDiagnosticSession(
        output_path, visualization_enabled=visualization_enabled
    )
    manifest: dict[str, Any] = {
        "run_kind": RUN_KIND,
        "diagnostic_telemetry_schema_version": DIAGNOSTIC_TELEMETRY_SCHEMA_VERSION,
        "diagnostic_only": True,
        "formal_metrics_included": False,
        "production_metrics_generated": False,
        "calibration_run": False,
        "baseline_frozen": False,
        "automatic_parameter_search": False,
        "round_1_run": False,
        "diagnostics_enabled": True,
        "visualization_enabled": True,
        "command": list(command or []),
        **repository,
        **runtime_metadata(),
        "traceability_kind": source["traceability_kind"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "tracked_diff_sha256": source["tracked_diff_sha256"],
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "metrics_schema_version": protocol.metrics_schema_version,
        "protocol_config_path": str(protocol.path),
        "protocol_config_sha256": protocol.sha256,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "diagnostic_seed_file_path": str(seeds_path),
        "diagnostic_seed_file_sha256": sha256_file(seeds_path),
        "reference_calibration_seed_file_sha256": sha256_file(
            protocol.splits["calibration"].path
        ),
        "reference_split_name": "calibration",
        "split_name": None,
        "seeds": list(seeds),
        "methods": [method.method_id for method in methods],
        "method_execution_order": [method.method_id for method in methods],
        "requested_pairs": len(seeds),
        "requested_episodes": len(seeds) * len(methods),
        "completed_pairs": 0,
        "completed_episodes": 0,
        "invalid_pairs": 0,
        "program_errors": 0,
        "program_error_details": [],
    }
    executions: list[Any] = []
    pair_rows: list[dict[str, Any]] = []
    fatal_error: str | None = None
    execution_index = 0
    try:
        logger.info(
            "calibration_diagnostic_start pairs=%s methods=%s seeds=%s",
            len(seeds),
            list(REQUIRED_METHODS),
            list(seeds),
        )
        for pair_index, seed in enumerate(seeds):
            pair_id = f"diagnostic_pair_{pair_index:04d}_seed_{seed}"
            pair_executions: dict[str, Any] = {}
            for method in methods:
                execution = _execute_episode(
                    method,
                    config,
                    seed,
                    pair_id,
                    execution_index,
                    logger,
                    diagnostic_factory=session,
                )
                execution_index += 1
                executions.append(execution)
                pair_executions[method.method_id] = execution
                if execution.result is not None:
                    manifest["completed_episodes"] += 1
                if execution.program_error:
                    manifest["program_errors"] += 1
                    manifest["program_error_details"].append(
                        {
                            "pair_id": pair_id,
                            "method": method.method_id,
                            "error": execution.program_error,
                        }
                    )
            oracle = pair_executions["b0_oracle"]
            vision = pair_executions["b1_vision"]
            pair_error = _validate_execution_pair(
                oracle, vision, atol=fingerprint_atol
            )
            pair_valid = pair_error is None
            oracle.pair_valid = pair_valid
            vision.pair_valid = pair_valid
            pair_rows.append(_paired_row(pair_id, seed, oracle, vision, pair_error))
            if pair_valid:
                manifest["completed_pairs"] += 1
            else:
                manifest["invalid_pairs"] += 1
                fatal_error = pair_error or "invalid diagnostic pair"
                break
            if oracle.program_error or vision.program_error:
                fatal_error = "Diagnostic episode ended with a program error"
                break
    except Exception:
        fatal_error = traceback.format_exc()
        manifest["program_errors"] += 1
        manifest["program_error_details"].append(
            {"pair_id": None, "method": None, "error": fatal_error}
        )
        logger.exception("calibration_diagnostic_program_error")

    try:
        episode_rows = [
            _episode_row(
                execution,
                protocol=None,
                split_name=None,
                config_sha256=manifest["config_sha256"],
                code_commit=str(repository.get("git_commit") or ""),
            )
            for execution in executions
        ]
        diagnostic_summaries = session.episode_summaries()
        summary_by_key = {
            (row["seed"], row["method"]): row for row in diagnostic_summaries
        }
        for row in episode_rows:
            diagnostic = summary_by_key.get((row.get("seed"), row.get("method_id")))
            if diagnostic is not None:
                row.update(
                    {
                        f"diagnostic.{key}": value
                        for key, value in diagnostic.items()
                        if key not in {"frames"}
                    }
                )
                row["diagnostic.frames"] = diagnostic["frames"]
                row["diagnostic.config_sha256"] = manifest["config_sha256"]
                row["diagnostic.protocol_version"] = protocol.protocol_version
        write_csv(
            output_path / "episode_diagnostics.csv",
            episode_rows,
            episode_fieldnames(episode_rows),
        )
        write_csv(output_path / "paired_results.csv", pair_rows, PAIRED_RESULT_FIELDS)
        trace_rows = session.trace_rows()
        trace_fields = tuple(
            sorted({key for row in trace_rows for key in row})
        )
        write_csv(output_path / "confirmation_trace.csv", trace_rows, trace_fields)
        write_json(output_path / "frames_manifest.json", session.frame_records())

        replay = _round_zero_replay_comparison(diagnostic_summaries, round_zero_path)
        if not replay["all_episodes_match"]:
            fatal_error = fatal_error or (
                "Diagnostics replay changed one or more archived stable behavior fields"
            )
            manifest["program_errors"] += 1
            manifest["program_error_details"].append(
                {"pair_id": None, "method": None, "error": fatal_error}
            )

        protected_after = _file_hashes(protected_paths)
        round_zero_after = _round_zero_hashes(round_zero_path)
        if protected_before != protected_after or round_zero_before != round_zero_after:
            fatal_error = fatal_error or "Protected inputs changed during diagnostics"
            manifest["program_errors"] += 1
            manifest["program_error_details"].append(
                {"pair_id": None, "method": None, "error": fatal_error}
            )
        summary = _build_summary(
            manifest=manifest,
            episode_summaries=diagnostic_summaries,
            replay=replay,
            protected_before=protected_before,
            protected_after=protected_after,
            round_zero_before=round_zero_before,
            round_zero_after=round_zero_after,
            frame_records=session.frame_records(),
        )
        write_json(output_path / "diagnostic_summary.json", summary)
        (output_path / "round_0_5_report.md").write_text(
            _render_report(summary), encoding="utf-8", newline="\n"
        )
    except Exception:
        output_error = traceback.format_exc()
        fatal_error = fatal_error or output_error
        manifest["program_errors"] += 1
        manifest["program_error_details"].append(
            {"pair_id": None, "method": None, "error": output_error}
        )
        logger.exception("calibration_diagnostic_output_error")
    finally:
        try:
            logger.info(
                "calibration_diagnostic_end completed_pairs=%s invalid_pairs=%s errors=%s",
                manifest["completed_pairs"],
                manifest["invalid_pairs"],
                manifest["program_errors"],
            )
            log_handler.flush()
            artifact_hashes = {
                path.relative_to(output_path).as_posix(): sha256_file(path)
                for path in sorted(output_path.rglob("*"))
                if path.is_file() and path.name != "run_manifest.json"
            }
            manifest["artifact_hashes"] = artifact_hashes
            manifest["artifact_set_sha256"] = _canonical_sha256(artifact_hashes)
            write_json(output_path / "run_manifest.json", manifest)
        except Exception:
            fatal_error = fatal_error or traceback.format_exc()
        logger.removeHandler(log_handler)
        log_handler.close()

    exit_code = 1 if fatal_error or manifest["invalid_pairs"] or manifest["program_errors"] else 0
    result = CalibrationDiagnosticRunResult(
        output_dir=output_path,
        requested_pairs=len(seeds),
        completed_pairs=int(manifest["completed_pairs"]),
        invalid_pairs=int(manifest["invalid_pairs"]),
        program_errors=int(manifest["program_errors"]),
        exit_code=exit_code,
    )
    if fatal_error:
        raise BenchmarkRunError(
            "Round 0.5 stopped after writing traceable outputs: " + fatal_error
        )
    return result


__all__ = [
    "CalibrationDiagnosticRunResult",
    "DIAGNOSTIC_TELEMETRY_SCHEMA_VERSION",
    "REQUIRED_DIAGNOSTIC_SEEDS",
    "REQUIRED_METHODS",
    "finalize_structured_review",
    "run_calibration_diagnostics",
    "validate_diagnostic_request",
]
