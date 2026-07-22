from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import traceback
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from controllers import B1DiagnosticSnapshot
from evaluation.protocol import ProtocolConfig
from perception.camera_geometry import ProjectionError, pixels_depth_to_world, world_to_pixel
from perception.image_io import save_depth_preview_png, save_mask_png, save_rgb_png

from .calibration_diagnostics import _DiagnosticRenderer, _source_provenance
from .manifest import repository_metadata, runtime_metadata, sha256_file
from .methods import MethodSpec, assert_static_fairness, resolve_methods
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


DIAGNOSTIC_SCHEMA_VERSION = "1.0.0"
RUN_KIND = "development_d0_5_passive_diagnostic_replay"
REQUIRED_SEEDS = (2225, 3989, 1557, 3301, 2574, 1297, 254, 3794, 1950, 3170)
REQUIRED_METHODS = ("b0_oracle", "b1_vision")
PERCEPTION_FOCUS_SEEDS = frozenset({3301, 2574, 1297, 254, 3170})
TRANSFER_FOCUS_SEEDS = frozenset({3794, 1950, 3170})
SHARED_GRASP_SEEDS = frozenset({2225, 3989, 1557})
EXPECTED_CANDIDATE_SHA256 = "ea81b78cbe9c3b579e694656a9b81c54940ce5ede34a2ce5fe100244a61234e1"
EXPECTED_FROZEN_CONFIG_SHA256 = "6808c142ae8805695fc43d5e4743a9529cdbea15008810456184e40e1c4b7ea9"
EXPECTED_PROTOCOL_SHA256 = "7a47be9ddf3851b06c84068ec29030d5bf25ebf60f37057d55371823b07e10bd"
EXPECTED_DEVELOPMENT_SHA256 = "677ecd23f9e689b971fa7340f7d34d674f07dfca19bfa9cd4634598d497b98d6"
EXPECTED_EXECUTION_COMMIT = "c3a1a699ba77b28ef79ae272047cd49de335914e"
EXPECTED_ANALYSIS_COMMIT = "0dc85b19b449a0f7d37bdef527a2561a917eceb9"

FROZEN_CONFIG_PATH = (PROJECT_ROOT / "configs/baselines/b1_vision_v1.toml").resolve()
PROTOCOL_PATH = (PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml").resolve()
DEVELOPMENT_SPLIT_PATH = (
    PROJECT_ROOT / "configs/splits/evaluation_protocol_v1/development_v1.txt"
).resolve()
SEED_SNAPSHOT_PATH = (
    PROJECT_ROOT / "configs/diagnostics/development_d0_5_seeds.txt"
).resolve()
FORMAL_D0_PATH = (
    PROJECT_ROOT / "outputs/development/b1_vision_v1/development_60"
).resolve()
CANDIDATE_PATH = (FORMAL_D0_PATH / "diagnostic_seed_candidates.json").resolve()
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs/development/b1_vision_v1/development_d0_5"
).resolve()
FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT / "configs/baselines/b1_vision_v1_manifest.json"
).resolve()

EXPECTED_CANDIDATE_IDENTITY = {
    2225: ("shared_grasp_failure", "typical_failure", "both_failed"),
    3989: ("shared_grasp_failure", "extreme_failure", "both_failed"),
    1557: ("shared_grasp_failure", "matched_success_control", "both_success"),
    3301: ("oracle_only_perception_failure", "typical_failure", "oracle_only_success"),
    2574: ("oracle_only_perception_failure", "extreme_failure", "oracle_only_success"),
    1297: ("oracle_only_perception_failure", "discordant_pair", "oracle_only_success"),
    254: ("oracle_only_perception_failure", "matched_success_control", "both_success"),
    3794: ("transfer_drop", "typical_failure", "oracle_only_success"),
    1950: ("transfer_drop", "extreme_failure", "both_failed"),
    3170: ("transfer_drop", "matched_success_control", "both_success"),
}

MILESTONES = (
    "scene_perception",
    "pregrasp_target",
    "first_contact",
    "grasp_candidate",
    "trial_lift_completion",
    "grasp_confirmation",
    "transfer_midpoint",
    "grasp_lost_or_descend_to_place",
    "release",
    "final_verification",
)
DECISIONS = frozenset(
    {"M-CENTERING", "M-ORIENTATION", "M-GATE", "M-COUPLED", "M-INCONCLUSIVE"}
)
CONTROLLER_PREFIX = "controller_observable."
PRIVILEGED_PREFIX = "privileged_diagnostic."
DERIVED_PREFIX = "derived_diagnostic."


@dataclass(frozen=True)
class DevelopmentDiagnosticRunResult:
    output_dir: Path
    requested_pairs: int
    completed_pairs: int
    invalid_pairs: int
    program_errors: int
    exit_code: int


def _strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path.name} contains non-finite token {token}")
        ),
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _directory_hashes(directory: Path) -> dict[str, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory is missing: {directory}")
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _protected_hashes() -> dict[str, str]:
    paths = (
        FROZEN_CONFIG_PATH,
        FREEZE_MANIFEST_PATH,
        PROTOCOL_PATH,
        DEVELOPMENT_SPLIT_PATH,
        SEED_SNAPSHOT_PATH,
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in paths
    }


def _finite(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Diagnostic value contains NaN or Inf")
    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    return value


def _bool_cell(value: str) -> bool | None:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean CSV cell: {value!r}")


def _json_cell(value: str) -> Any:
    return None if value == "" else json.loads(value)


def validate_development_diagnostic_request(
    *,
    protocol: ProtocolConfig,
    config_path: str | Path,
    development_run_dir: str | Path,
    candidate_file: str | Path,
    method_ids: Sequence[str],
    seed_snapshot_path: str | Path = SEED_SNAPSHOT_PATH,
) -> tuple[Any, tuple[int, ...], tuple[MethodSpec, ...], list[dict[str, Any]]]:
    config_file = Path(config_path).expanduser().resolve()
    run_dir = Path(development_run_dir).expanduser().resolve()
    candidate = Path(candidate_file).expanduser().resolve()
    snapshot = Path(seed_snapshot_path).expanduser().resolve()
    if protocol.path.resolve() != PROTOCOL_PATH or protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("D0.5 requires the registered Evaluation Protocol v1 bytes")
    if config_file != FROZEN_CONFIG_PATH or sha256_file(config_file) != EXPECTED_FROZEN_CONFIG_SHA256:
        raise ValueError("D0.5 requires the byte-exact frozen B1-Vision v1 config")
    if run_dir != FORMAL_D0_PATH:
        raise ValueError(f"D0.5 requires the formal Development D0 archive {FORMAL_D0_PATH}")
    if candidate != CANDIDATE_PATH or sha256_file(candidate) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("D0.5 accepts only the registered formal candidate file")
    if snapshot != SEED_SNAPSHOT_PATH:
        raise ValueError("D0.5 rejects custom diagnostic seed snapshots")
    if tuple(method_ids) != REQUIRED_METHODS:
        raise ValueError("D0.5 methods must be exactly b0_oracle then b1_vision")
    if sha256_file(DEVELOPMENT_SPLIT_PATH) != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("Registered Development split hash mismatch")

    seeds = tuple(load_seeds(snapshot))
    if seeds != REQUIRED_SEEDS or len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError(f"D0.5 seeds must be exactly {list(REQUIRED_SEEDS)} in order")
    development = set(load_seeds(DEVELOPMENT_SPLIT_PATH))
    if not set(seeds).issubset(development):
        raise ValueError("Every D0.5 seed must already belong to Development")

    validation = _strict_json(run_dir / "development_run_validation.json")
    if not isinstance(validation, Mapping) or validation.get("status") != "PASS":
        raise ValueError("Formal Development D0 validation is not PASS")
    if validation.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("Formal Development execution commit mismatch")
    if validation.get("analysis_commit") != EXPECTED_ANALYSIS_COMMIT:
        raise ValueError("Formal Development analysis commit mismatch")
    if validation.get("formal_raw_files_unchanged") is not True:
        raise ValueError("Formal Development raw archive is not immutable")
    if validation.get("requested_pairs") != 60 or validation.get("completed_pairs") != 60:
        raise ValueError("Formal Development D0 is incomplete")

    document = _strict_json(candidate)
    candidates = document.get("candidates") if isinstance(document, Mapping) else None
    if not isinstance(candidates, list):
        raise ValueError("Candidate document lacks candidates")
    candidate_seeds = tuple(int(item["seed"]) for item in candidates)
    if candidate_seeds != REQUIRED_SEEDS:
        raise ValueError("Candidate file seed order does not match the fixed snapshot")
    for item in candidates:
        seed = int(item["seed"])
        expected = EXPECTED_CANDIDATE_IDENTITY[seed]
        actual = (
            item.get("problem_family"),
            item.get("selection_role"),
            item.get("pair_category"),
        )
        if actual != expected:
            raise ValueError(f"Candidate identity mismatch for seed {seed}: {actual}")
    if document.get("candidate_count") != 10 or document.get("unique_seed_count") != 10:
        raise ValueError("Candidate file must declare ten unique seeds")
    if document.get("d0_5_episodes_run") != 0:
        raise ValueError("Candidate file no longer represents the pre-D0.5 snapshot")

    config, overrides = _effective_config(config_file)
    if overrides:
        raise ValueError(f"Frozen baseline required effective overrides: {overrides}")
    methods = resolve_methods(list(method_ids))
    assert_static_fairness(methods, config)
    return config, seeds, methods, [dict(item) for item in candidates]


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def _euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = float(math.hypot(rotation[0, 0], rotation[1, 0]))
    if sy > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    value = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(value, np.asarray(rotation, dtype=float).reshape(-1))
    return tuple(float(item) for item in value)


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return np.asarray(fallback if norm <= 1e-12 else vector / norm, dtype=float)


def _mask_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    for start_v, start_u in np.argwhere(mask):
        start = (int(start_v), int(start_u))
        if visited[start]:
            continue
        count += 1
        visited[start] = True
        stack = [start]
        component: list[tuple[int, int]] = []
        while stack:
            v, u = stack.pop()
            component.append((v, u))
            for nv, nu in ((v - 1, u), (v + 1, u), (v, u - 1), (v, u + 1)):
                if 0 <= nv < height and 0 <= nu < width and mask[nv, nu] and not visited[nv, nu]:
                    visited[nv, nu] = True
                    stack.append((nv, nu))
        if len(component) > len(best):
            best = component
    largest = np.zeros_like(mask, dtype=bool)
    if best:
        points = np.asarray(best, dtype=int)
        largest[points[:, 0], points[:, 1]] = True
    return count, largest


def _bbox(mask: np.ndarray) -> list[int] | None:
    v, u = np.nonzero(mask)
    if not len(u):
        return None
    return [int(u.min()), int(v.min()), int(u.max()), int(v.max())]


def _visible_direction(mask: np.ndarray) -> tuple[float | None, float]:
    v, u = np.nonzero(mask)
    if len(u) < 3:
        return None, 0.0
    points = np.column_stack((u, -v)).astype(float)
    covariance = np.cov(points - np.mean(points, axis=0), rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)
    major = vectors[:, order[-1]]
    denominator = max(float(values[order[-1]]), 1e-12)
    anisotropy = float((values[order[-1]] - values[order[-2]]) / denominator)
    angle = float(math.atan2(major[1], major[0])) if anisotropy >= 0.05 else None
    return angle, anisotropy


def _overlay(rgb: np.ndarray, object_mask: np.ndarray, target_mask: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    obj = np.asarray(object_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    image[obj] = np.rint(0.35 * image[obj] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
    image[target] = np.rint(0.35 * image[target] + 0.65 * np.array([0, 255, 255])).astype(np.uint8)
    return image


class DevelopmentEpisodeRecorder:
    """Runner-side, return-value-free recorder for one immutable replay."""

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
        candidate: Mapping[str, Any],
    ) -> None:
        self.env = env
        self.method = method
        self.seed = int(seed)
        self.pair_id = pair_id
        self.execution_index = int(execution_index)
        self.output_dir = output_dir
        self.visualization_enabled = bool(visualization_enabled)
        self.candidate = dict(candidate)
        self.trace_rows: list[dict[str, Any]] = []
        self.contact_rows: list[dict[str, Any]] = []
        self.perception_rows: list[dict[str, Any]] = []
        self.frame_records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.result: Any | None = None
        self.fingerprint: Any | None = None
        self.external_state_metrics: Any | None = None
        self.initial_robot_state: tuple[float, ...] | None = None
        self.provider_call_count = 0
        self.detector_call_count = 0
        self._current_stage = "scene_perception"
        self._renderer: _DiagnosticRenderer | None = None
        self._captured: set[str] = set()
        self._initial_object_position: np.ndarray | None = None
        self._initial_object_rotation: np.ndarray | None = None
        self._candidate_relative_position: np.ndarray | None = None
        self._candidate_relative_rotation: np.ndarray | None = None
        self._peak_object_z: float | None = None
        self._first_left_contact_time: float | None = None
        self._first_right_contact_time: float | None = None
        self._first_contact_classification: str | None = None
        self._last_tcp_position: np.ndarray | None = None
        self._last_tcp_velocity: np.ndarray | None = None
        self._last_trace_time: float | None = None
        self._transfer_start_time: float | None = None
        self._render_checks = 0
        self._render_state_unchanged = True
        self._last_object_bbox: list[int] | None = None
        self._last_object_center_pixel: tuple[float, float] | None = None
        self._last_visible_angle: float | None = None
        self._last_visible_anisotropy: float | None = None
        self._left_body_id = int(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        )
        self._right_body_id = int(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        )
        self._hand_body_id = int(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        )
        self._left_geoms = self._body_geoms(self._left_body_id)
        self._right_geoms = self._body_geoms(self._right_body_id)
        self._hand_geoms = frozenset(
            geom_id
            for geom_id in range(env.model.ngeom)
            if self._is_descendant(int(env.model.geom_bodyid[geom_id]), self._hand_body_id)
        )

    def _is_descendant(self, body_id: int, ancestor: int) -> bool:
        while body_id > 0:
            if body_id == ancestor:
                return True
            body_id = int(self.env.model.body_parentid[body_id])
        return False

    def _body_geoms(self, body_id: int) -> frozenset[int]:
        return frozenset(
            int(geom_id)
            for geom_id in np.flatnonzero(
                (self.env.model.geom_bodyid == body_id)
                & (self.env.model.geom_contype != 0)
            )
        )

    def _finger_geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left = np.asarray(self.env.data.xpos[self._left_body_id], dtype=float).copy()
        right = np.asarray(self.env.data.xpos[self._right_body_id], dtype=float).copy()
        closing = _unit(right - left, np.array([0.0, 1.0, 0.0]))
        horizontal = closing.copy()
        horizontal[2] = 0.0
        horizontal = _unit(horizontal, np.array([0.0, 1.0, 0.0]))
        perpendicular = _unit(
            np.cross(np.array([0.0, 0.0, 1.0]), horizontal),
            np.array([1.0, 0.0, 0.0]),
        )
        return left, right, closing, perpendicular, horizontal

    def _raw_finger_contacts(self) -> tuple[bool, bool]:
        left = False
        right = False
        for index in range(int(self.env.data.ncon)):
            contact = self.env.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.env.object_geom_id:
                other = geom2
            elif geom2 == self.env.object_geom_id:
                other = geom1
            else:
                continue
            left = left or other in self._left_geoms
            right = right or other in self._right_geoms
        return left, right

    def _contact_classification(self, position: np.ndarray, object_position: np.ndarray, object_rotation: np.ndarray) -> str:
        half = float(self.env.config.workspace.object_half_size)
        local = object_rotation.T @ (position - object_position)
        near = int(np.count_nonzero(np.abs(np.abs(local) - half) <= 0.0045))
        if near >= 3:
            return "corner"
        if near == 2:
            return "edge"
        if near == 1:
            return "face"
        return "unknown"

    def _contact_trace(
        self,
        *,
        event: str,
        stage: str,
        object_position: np.ndarray,
        object_rotation: np.ndarray,
    ) -> None:
        for index in range(int(self.env.data.ncon)):
            contact = self.env.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            finger = None
            if geom1 == self.env.object_geom_id:
                other = geom2
            elif geom2 == self.env.object_geom_id:
                other = geom1
            else:
                continue
            if other in self._left_geoms:
                finger = "left"
            elif other in self._right_geoms:
                finger = "right"
            if finger is None:
                continue
            position = np.asarray(contact.pos, dtype=float).copy()
            normal = np.asarray(contact.frame[:3], dtype=float).copy()
            force = np.zeros(6, dtype=float)
            force_available = bool(int(contact.efc_address) >= 0)
            if force_available:
                mujoco.mj_contactForce(self.env.model, self.env.data, index, force)
            classification = self._contact_classification(
                position, object_position, object_rotation
            )
            if self._first_contact_classification is None:
                self._first_contact_classification = f"{finger}:{classification}"
            self.contact_rows.append(
                _finite(
                    {
                        "seed": self.seed,
                        "method": self.method.method_id,
                        "pair_id": self.pair_id,
                        "simulation_time": float(self.env.data.time),
                        "stage": stage,
                        "event": event,
                        "contact_index": index,
                        "finger": finger,
                        "geom1_id": geom1,
                        "geom2_id": geom2,
                        "geom1_name": mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
                        "geom2_name": mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
                        "position_x": position[0],
                        "position_y": position[1],
                        "position_z": position[2],
                        "normal_x": normal[0],
                        "normal_y": normal[1],
                        "normal_z": normal[2],
                        "penetration_distance": float(contact.dist),
                        "solver_force_available": force_available,
                        "normal_force": float(force[0]) if force_available else None,
                        "force_norm": float(np.linalg.norm(force[:3])) if force_available else None,
                        "object_surface_classification": classification,
                        "availability": "privileged_diagnostic_only",
                    }
                )
            )

    def _table_clearance(self) -> float:
        values = [
            float(self.env.data.geom_xpos[geom_id, 2] - self.env.model.geom_rbound[geom_id] - 0.22)
            for geom_id in self._hand_geoms
        ]
        return min(values) if values else 0.0

    def _physics_row(self, snapshot: B1DiagnosticSnapshot) -> dict[str, Any]:
        tcp_position = np.asarray(self.env.data.site_xpos[self.env.tcp_site_id], dtype=float).copy()
        tcp_rotation = np.asarray(
            self.env.data.site_xmat[self.env.tcp_site_id], dtype=float
        ).reshape(3, 3).copy()
        object_position = np.asarray(self.env.data.xpos[self.env.object_body_id], dtype=float).copy()
        object_rotation = np.asarray(
            self.env.data.xmat[self.env.object_body_id], dtype=float
        ).reshape(3, 3).copy()
        object_quaternion = np.asarray(
            self.env.data.xquat[self.env.object_body_id], dtype=float
        ).copy()
        object_velocity = np.asarray(
            self.env.data.qvel[
                self.env.object_dof_address : self.env.object_dof_address + 6
            ],
            dtype=float,
        ).copy()
        if self._initial_object_position is None:
            self._initial_object_position = object_position.copy()
            self._initial_object_rotation = object_rotation.copy()
        assert self._initial_object_rotation is not None
        assert self._initial_object_position is not None
        relative_position = tcp_rotation.T @ (object_position - tcp_position)
        relative_rotation = tcp_rotation.T @ object_rotation
        if snapshot.event == "candidate_sample" and snapshot.grasp_state == "grasp_candidate" and self._candidate_relative_position is None:
            self._candidate_relative_position = relative_position.copy()
            self._candidate_relative_rotation = relative_rotation.copy()
        relative_slip = 0.0
        relative_rotation_change = 0.0
        if self._candidate_relative_position is not None:
            relative_slip = float(np.linalg.norm(relative_position - self._candidate_relative_position))
        if self._candidate_relative_rotation is not None:
            relative_rotation_change = _rotation_angle(
                self._candidate_relative_rotation.T @ relative_rotation
            )
        self._peak_object_z = max(
            float(object_position[2]),
            float(object_position[2]) if self._peak_object_z is None else self._peak_object_z,
        )
        object_drop = float(self._peak_object_z - object_position[2])
        initial_delta_rotation = self._initial_object_rotation.T @ object_rotation
        orientation_angle = _rotation_angle(initial_delta_rotation)
        delta_roll, delta_pitch, delta_yaw = _euler_xyz(initial_delta_rotation)
        tcp_roll, tcp_pitch, tcp_yaw = _euler_xyz(tcp_rotation)
        left, right, closing, perpendicular, closing_horizontal = self._finger_geometry()
        approach = -tcp_rotation[:, 2]
        object_axes = (object_rotation[:, 0], object_rotation[:, 1])
        closing_xy = _unit(closing_horizontal[:2], np.array([0.0, 1.0]))
        face_angles = []
        for axis in object_axes:
            axis_xy = _unit(np.asarray(axis[:2]), np.array([1.0, 0.0]))
            face_angles.append(math.acos(float(np.clip(abs(np.dot(closing_xy, axis_xy)), -1.0, 1.0))))
        symmetry_reduced_angle = float(min(face_angles))
        raw_left, raw_right = self._raw_finger_contacts()
        now = float(snapshot.simulation_time)
        if raw_left and self._first_left_contact_time is None:
            self._first_left_contact_time = now
        if raw_right and self._first_right_contact_time is None:
            self._first_right_contact_time = now
        tcp_velocity = np.zeros(3, dtype=float)
        tcp_acceleration = np.zeros(3, dtype=float)
        if self._last_tcp_position is not None and self._last_trace_time is not None and now > self._last_trace_time:
            dt = now - self._last_trace_time
            tcp_velocity = (tcp_position - self._last_tcp_position) / dt
            if self._last_tcp_velocity is not None:
                tcp_acceleration = (tcp_velocity - self._last_tcp_velocity) / dt
        self._last_tcp_position = tcp_position.copy()
        self._last_tcp_velocity = tcp_velocity.copy()
        self._last_trace_time = now
        aperture = float(sum(snapshot.finger_positions)) if snapshot.gripper_aperture is None else snapshot.gripper_aperture
        row: dict[str, Any] = {
            "seed": self.seed,
            "method": self.method.method_id,
            "pair_id": self.pair_id,
            "execution_index": self.execution_index,
            "event": snapshot.event,
            "simulation_time": now,
            "stage": snapshot.stage,
            "next_stage": snapshot.next_stage,
            "failure_reason": snapshot.failure_reason,
        }
        for key, value in asdict(snapshot).items():
            if key not in {"event", "simulation_time", "stage", "next_stage", "failure_reason"}:
                row[f"{CONTROLLER_PREFIX}{key}"] = value
        row[f"{CONTROLLER_PREFIX}gripper_aperture"] = aperture
        row.update(
            {
                f"{PRIVILEGED_PREFIX}tcp_quaternion": _quaternion(tcp_rotation),
                f"{PRIVILEGED_PREFIX}tcp_rotation": tcp_rotation.reshape(-1),
                f"{PRIVILEGED_PREFIX}object_center": object_position,
                f"{PRIVILEGED_PREFIX}object_quaternion": object_quaternion,
                f"{PRIVILEGED_PREFIX}object_rotation": object_rotation.reshape(-1),
                f"{PRIVILEGED_PREFIX}object_linear_velocity": object_velocity[:3],
                f"{PRIVILEGED_PREFIX}object_angular_velocity": object_velocity[3:],
                f"{PRIVILEGED_PREFIX}left_finger_center": left,
                f"{PRIVILEGED_PREFIX}right_finger_center": right,
                f"{PRIVILEGED_PREFIX}left_finger_object_center_distance": np.linalg.norm(left - object_position),
                f"{PRIVILEGED_PREFIX}right_finger_object_center_distance": np.linalg.norm(right - object_position),
                f"{PRIVILEGED_PREFIX}raw_left_contact": raw_left,
                f"{PRIVILEGED_PREFIX}raw_right_contact": raw_right,
                f"{PRIVILEGED_PREFIX}object_lift_height": object_position[2] - self._initial_object_position[2],
                f"{PRIVILEGED_PREFIX}object_peak_height": self._peak_object_z,
                f"{PRIVILEGED_PREFIX}object_drop_after_peak": object_drop,
                f"{PRIVILEGED_PREFIX}relative_gripper_object_position": relative_position,
                f"{PRIVILEGED_PREFIX}relative_gripper_object_rotation": relative_rotation.reshape(-1),
                f"{PRIVILEGED_PREFIX}minimum_table_clearance_conservative": self._table_clearance(),
                f"{DERIVED_PREFIX}closing_axis": closing,
                f"{DERIVED_PREFIX}approach_axis": approach,
                f"{DERIVED_PREFIX}perpendicular_horizontal_axis": perpendicular,
                f"{DERIVED_PREFIX}tcp_roll": tcp_roll,
                f"{DERIVED_PREFIX}tcp_pitch": tcp_pitch,
                f"{DERIVED_PREFIX}tcp_yaw": tcp_yaw,
                f"{DERIVED_PREFIX}object_orientation_change_angle": orientation_angle,
                f"{DERIVED_PREFIX}object_roll_change": delta_roll,
                f"{DERIVED_PREFIX}object_pitch_change": delta_pitch,
                f"{DERIVED_PREFIX}object_yaw_change": delta_yaw,
                f"{DERIVED_PREFIX}relative_translation_slip_from_candidate": relative_slip,
                f"{DERIVED_PREFIX}relative_rotation_slip_from_candidate": relative_rotation_change,
                f"{DERIVED_PREFIX}cube_symmetry_reduced_closing_axis_angle": symmetry_reduced_angle,
                f"{DERIVED_PREFIX}cube_principal_axis_unique": False,
                f"{DERIVED_PREFIX}visible_edge_angle": self._last_visible_angle,
                f"{DERIVED_PREFIX}visible_edge_anisotropy": self._last_visible_anisotropy,
                f"{DERIVED_PREFIX}tcp_speed": np.linalg.norm(tcp_velocity),
                f"{DERIVED_PREFIX}tcp_acceleration": np.linalg.norm(tcp_acceleration),
            }
        )
        self._contact_trace(
            event=snapshot.event,
            stage=snapshot.stage,
            object_position=object_position,
            object_rotation=object_rotation,
        )
        return _finite(row)

    def _state_vector(self) -> tuple[np.ndarray, ...]:
        return (
            np.asarray([self.env.data.time], dtype=float).copy(),
            self.env.data.qpos.copy(),
            self.env.data.qvel.copy(),
            self.env.data.ctrl.copy(),
        )

    def _capture_milestone(self, milestone: str, simulation_time: float) -> None:
        if milestone in self._captured or not self.visualization_enabled:
            return
        if self._renderer is None:
            self._renderer = _DiagnosticRenderer(self.env)
        base = self.output_dir / "frames" / f"seed_{self.seed}" / self.method.method_id
        before = self._state_vector()
        for view, renderer_view in (("diagnostic_top", "overhead_rgb"), ("diagnostic_side", "diagnostic_side")):
            image = self._renderer.capture(renderer_view)
            path = base / f"{milestone}__{view}.png"
            save_rgb_png(path, image)
            self.frame_records.append(
                {
                    "seed": self.seed,
                    "method": self.method.method_id,
                    "milestone": milestone,
                    "status": "captured",
                    "view": view,
                    "source": "diagnostic_render_only",
                    "simulation_time": simulation_time,
                    "path": path.relative_to(self.output_dir).as_posix(),
                }
            )
        after = self._state_vector()
        unchanged = all(np.array_equal(left, right) for left, right in zip(before, after))
        self._render_checks += 1
        self._render_state_unchanged = self._render_state_unchanged and unchanged
        if not unchanged:
            raise RuntimeError("Diagnostic rendering changed time/qpos/qvel/ctrl")
        self._captured.add(milestone)

    def observe(self, snapshot: B1DiagnosticSnapshot) -> None:
        try:
            if not isinstance(snapshot, B1DiagnosticSnapshot):
                raise TypeError("Controller supplied a non-diagnostic snapshot")
            self._current_stage = snapshot.stage
            row = self._physics_row(snapshot)
            self.trace_rows.append(row)
            if snapshot.event == "episode_reset":
                self._capture_milestone("scene_perception", snapshot.simulation_time)
            if snapshot.event == "stage_transition" and snapshot.stage == "scene_perception":
                self._capture_milestone("pregrasp_target", snapshot.simulation_time)
            if snapshot.event == "close_sample" and (
                row[f"{PRIVILEGED_PREFIX}raw_left_contact"]
                or row[f"{PRIVILEGED_PREFIX}raw_right_contact"]
            ):
                self._capture_milestone("first_contact", snapshot.simulation_time)
            if snapshot.event == "candidate_sample" and snapshot.grasp_state == "grasp_candidate":
                self._capture_milestone("grasp_candidate", snapshot.simulation_time)
            if snapshot.event == "stage_transition" and snapshot.stage == "trial_lift":
                self._capture_milestone("trial_lift_completion", snapshot.simulation_time)
            if snapshot.event == "confirmation_sample" and snapshot.grasp_state == "grasp_confirmed":
                self._capture_milestone("grasp_confirmation", snapshot.simulation_time)
            if snapshot.event == "transport_sample" and snapshot.stage == "transfer":
                if self._transfer_start_time is None:
                    self._transfer_start_time = snapshot.simulation_time
                if snapshot.simulation_time - self._transfer_start_time >= 2.0:
                    self._capture_milestone("transfer_midpoint", snapshot.simulation_time)
            if snapshot.event == "stage_transition" and snapshot.stage == "transfer":
                self._capture_milestone("grasp_lost_or_descend_to_place", snapshot.simulation_time)
            if snapshot.event == "release_sample":
                self._capture_milestone("release", snapshot.simulation_time)
            if snapshot.event == "episode_end":
                if snapshot.stage in {"grasp_confirmation", "transfer", "trial_lift"}:
                    self._capture_milestone("grasp_confirmation", snapshot.simulation_time)
                if snapshot.stage == "transfer":
                    self._capture_milestone("grasp_lost_or_descend_to_place", snapshot.simulation_time)
                if snapshot.stage == "final_visual_verification" or snapshot.next_stage == "completed":
                    self._capture_milestone("final_verification", snapshot.simulation_time)
            if snapshot.event == "stage_transition" and snapshot.next_stage is not None:
                self._current_stage = snapshot.next_stage
        except Exception:
            self.errors.append(traceback.format_exc())

    def _color_mask(self, rgb: np.ndarray, kind: str, config: Any) -> np.ndarray:
        colors = rgb.astype(np.float32)
        red, green, blue = colors[..., 0], colors[..., 1], colors[..., 2]
        if kind == "object":
            minimum = np.asarray(config.object_min_rgb, dtype=float)
            ratio = float(config.object_dominance_ratio)
            return (
                (red >= minimum[0])
                & (green >= minimum[1])
                & (blue >= minimum[2])
                & (red >= ratio * green)
                & (red >= ratio * blue)
            )
        minimum = np.asarray(config.target_min_rgb, dtype=float)
        ratio = float(config.target_dominance_ratio)
        return (
            (red >= minimum[0])
            & (green >= minimum[1])
            & (blue >= minimum[2])
            & (green >= ratio * red)
            & (green >= ratio * blue)
        )

    def _audit_detection(
        self,
        *,
        frame: Any,
        detection: Any,
        kind: str,
        detector: Any,
        call_index: int,
        estimate: Any,
    ) -> dict[str, Any]:
        config = detector.config
        color = self._color_mask(frame.rgb, kind, config)
        valid_depth = (
            np.isfinite(frame.depth)
            & (frame.depth >= config.minimum_depth)
            & (frame.depth <= config.maximum_depth)
        )
        color_depth = color & valid_depth
        projected_count = 0
        z_filtered_count = 0
        component_count = 0
        selected = np.zeros_like(color, dtype=bool)
        z_range = config.object_world_z_range if kind == "object" else config.target_world_z_range
        v_all, u_all = np.nonzero(color_depth)
        projection_error = None
        if len(u_all):
            try:
                world = pixels_depth_to_world(
                    u_all,
                    v_all,
                    frame.depth[v_all, u_all],
                    frame.intrinsics,
                    frame.extrinsics,
                )
                projected_count = int(len(world))
                valid_z = (world[:, 2] >= z_range[0]) & (world[:, 2] <= z_range[1])
                geometry = np.zeros_like(color, dtype=bool)
                geometry[v_all[valid_z], u_all[valid_z]] = True
                z_filtered_count = int(np.count_nonzero(geometry))
                component_count, selected = _mask_components(geometry)
            except ProjectionError as exc:
                projection_error = str(exc)
        selected_bbox = _bbox(detection.mask)
        direction, anisotropy = _visible_direction(detection.mask)
        if kind == "object":
            self._last_object_bbox = selected_bbox
            self._last_object_center_pixel = detection.center_pixel
            self._last_visible_angle = direction
            self._last_visible_anisotropy = anisotropy
        privileged = np.asarray(
            self.env.data.xpos[self.env.object_body_id]
            if kind == "object"
            else self.env.data.site_xpos[self.env.place_target_site_id],
            dtype=float,
        ).copy()
        position = detection.position
        error = None if position is None else float(np.linalg.norm(np.asarray(position) - privileged))
        projected_truth = None
        truth_in_view = False
        try:
            projected_truth = world_to_pixel(
                privileged,
                frame.intrinsics,
                frame.extrinsics,
                require_inside=False,
            )
            truth_in_view = bool(
                0 <= projected_truth[0] < frame.width
                and 0 <= projected_truth[1] < frame.height
            )
        except ProjectionError:
            pass
        if not truth_in_view:
            occlusion = "truth_outside_camera_view"
        elif detection.pixel_count == 0:
            occlusion = "truth_in_view_no_selected_color_depth_component"
        elif detection.pixel_count < (config.minimum_object_pixels if kind == "object" else config.minimum_target_pixels):
            occlusion = "truth_in_view_component_below_pixel_threshold"
        elif detection.failure_reason == "perception_low_confidence":
            occlusion = "visible_component_low_confidence"
        else:
            occlusion = "selected_component_available"
        return _finite(
            {
                "seed": self.seed,
                "method": self.method.method_id,
                "pair_id": self.pair_id,
                "stage": self._current_stage,
                "provider_call_index": call_index,
                "frame_index_within_stage": sum(
                    1
                    for row in self.perception_rows
                    if row.get("stage") == self._current_stage
                    and row.get("component") == kind
                ),
                "simulation_time": float(frame.simulation_time),
                "component": kind,
                "camera_name": frame.camera_name,
                "image_width": frame.width,
                "image_height": frame.height,
                "color_mask_pixel_count": int(np.count_nonzero(color)),
                "valid_depth_pixel_count": int(np.count_nonzero(color_depth)),
                "depth_invalid_count_within_color_mask": int(np.count_nonzero(color & ~valid_depth)),
                "projection_input_count": int(len(u_all)),
                "projection_output_count": projected_count,
                "world_z_filter_input_count": projected_count,
                "world_z_filter_output_count": z_filtered_count,
                "workspace_filter_input_count": z_filtered_count,
                "workspace_filter_output_count": z_filtered_count,
                "workspace_filter_applied": False,
                "connected_component_count": component_count,
                "selected_component_pixel_count_audit": int(np.count_nonzero(selected)),
                "mask_pixel_count": int(detection.pixel_count),
                "bounding_box": selected_bbox,
                "centroid": detection.center_pixel,
                "confidence": float(detection.confidence),
                "success": bool(detection.success),
                "rejection_reason": detection.failure_reason,
                "projection_error": projection_error,
                "provider_position": position,
                "provider_component_valid": (
                    estimate.object_valid if kind == "object" else estimate.target_valid
                ),
                "final_provider_result": estimate.failure_reason,
                f"{DERIVED_PREFIX}visible_geometry_direction_rad": direction,
                f"{DERIVED_PREFIX}visible_geometry_anisotropy": anisotropy,
                f"{DERIVED_PREFIX}occlusion_diagnostic": occlusion,
                f"{PRIVILEGED_PREFIX}same_time_position": privileged,
                f"{DERIVED_PREFIX}position_error": error,
                f"{DERIVED_PREFIX}truth_projected_pixel": projected_truth,
                f"{DERIVED_PREFIX}truth_in_camera_view": truth_in_view,
                "availability": "online_observable_existing_perception_call",
            }
        )

    def _save_perception_frame(self, raw_provider: Any, call_index: int) -> None:
        frame = raw_provider.last_frame
        obj = raw_provider.last_object_detection
        target = raw_provider.last_target_detection
        if frame is None or obj is None or target is None:
            return
        base = self.output_dir / "frames" / f"seed_{self.seed}" / self.method.method_id / "perception"
        stem = f"{call_index:02d}_{self._current_stage}"
        artifacts = (
            ("control_rgb", base / f"{stem}__control_rgb.png", frame.rgb),
            ("control_depth_preview", base / f"{stem}__control_depth_preview.png", frame.depth),
            (
                "object_target_mask_overlay",
                base / f"{stem}__object_target_mask_overlay.png",
                _overlay(frame.rgb, obj.mask, target.mask),
            ),
            ("object_mask", base / f"{stem}__object_mask.png", obj.mask),
            ("target_mask", base / f"{stem}__target_mask.png", target.mask),
        )
        for view, path, data in artifacts:
            if view == "control_depth_preview":
                save_depth_preview_png(path, data)
            elif view.endswith("_mask"):
                save_mask_png(path, data)
            else:
                save_rgb_png(path, data)
            self.frame_records.append(
                {
                    "seed": self.seed,
                    "method": self.method.method_id,
                    "milestone": self._current_stage,
                    "status": "captured",
                    "view": view,
                    "source": "existing_control_perception_call",
                    "simulation_time": float(frame.simulation_time),
                    "provider_call_index": call_index,
                    "path": path.relative_to(self.output_dir).as_posix(),
                }
            )

    def observe_provider(self, *, estimate: Any, raw_provider: Any, call_index: int) -> None:
        try:
            self.provider_call_count += 1
            frame = getattr(raw_provider, "last_frame", None)
            object_detection = getattr(raw_provider, "last_object_detection", None)
            target_detection = getattr(raw_provider, "last_target_detection", None)
            detector = getattr(raw_provider, "detector", None)
            if frame is None or detector is None:
                privileged_object = np.asarray(
                    self.env.data.xpos[self.env.object_body_id], dtype=float
                ).copy()
                position = estimate.object_position
                self.perception_rows.append(
                    _finite(
                        {
                            "seed": self.seed,
                            "method": self.method.method_id,
                            "pair_id": self.pair_id,
                            "stage": self._current_stage,
                            "provider_call_index": call_index,
                            "frame_index_within_stage": sum(
                                1
                                for row in self.perception_rows
                                if row.get("stage") == self._current_stage
                            ),
                            "simulation_time": float(self.env.data.time),
                            "component": "oracle_task_state",
                            "success": bool(estimate.valid),
                            "confidence": float(estimate.confidence),
                            "rejection_reason": estimate.failure_reason,
                            "provider_position": position,
                            f"{PRIVILEGED_PREFIX}same_time_position": privileged_object,
                            f"{DERIVED_PREFIX}position_error": (
                                None
                                if position is None
                                else float(np.linalg.norm(np.asarray(position) - privileged_object))
                            ),
                            "availability": "oracle_control_observable_not_deployable",
                        }
                    )
                )
                return
            self.detector_call_count += 2
            self.perception_rows.append(
                self._audit_detection(
                    frame=frame,
                    detection=object_detection,
                    kind="object",
                    detector=detector,
                    call_index=call_index,
                    estimate=estimate,
                )
            )
            self.perception_rows.append(
                self._audit_detection(
                    frame=frame,
                    detection=target_detection,
                    kind="target",
                    detector=detector,
                    call_index=call_index,
                    estimate=estimate,
                )
            )
            if self.visualization_enabled and self.seed in PERCEPTION_FOCUS_SEEDS:
                self._save_perception_frame(raw_provider, call_index)
            if self._current_stage == "final_visual_verification":
                self._capture_milestone("final_verification", float(frame.simulation_time))
        except Exception:
            self.errors.append(traceback.format_exc())

    def _enrich_centering(self) -> None:
        if self.result is None:
            return
        provider_target = self.result.pregrasp_corrected_object_position
        if provider_target is None:
            provider_target = self.result.initial_object_position
        if provider_target is None:
            return
        object_target = np.asarray(provider_target, dtype=float)
        tcp_target = object_target + np.array(
            [0.0, 0.0, float(self.env.config.controller.grasp_z_offset)]
        )
        half = float(self.env.config.workspace.object_half_size)
        for row in self.trace_rows:
            object_center = np.asarray(
                row[f"{PRIVILEGED_PREFIX}object_center"], dtype=float
            )
            closing = np.asarray(row[f"{DERIVED_PREFIX}closing_axis"], dtype=float)
            perpendicular = np.asarray(
                row[f"{DERIVED_PREFIX}perpendicular_horizontal_axis"], dtype=float
            )
            delta = object_target - object_center
            tcp_delta = tcp_target - object_center
            closing_offset = float(np.dot(delta, closing))
            perpendicular_offset = float(np.dot(delta, perpendicular))
            row.update(
                _finite(
                    {
                        f"{CONTROLLER_PREFIX}provider_object_target_position": object_target,
                        f"{CONTROLLER_PREFIX}grasp_target_position": object_target,
                        f"{CONTROLLER_PREFIX}tcp_target_position": tcp_target,
                        f"{DERIVED_PREFIX}grasp_target_minus_object_center": delta,
                        f"{DERIVED_PREFIX}tcp_target_minus_object_center": tcp_delta,
                        f"{DERIVED_PREFIX}world_xy_centering_offset": np.linalg.norm(delta[:2]),
                        f"{DERIVED_PREFIX}closing_axis_offset": closing_offset,
                        f"{DERIVED_PREFIX}perpendicular_horizontal_offset": perpendicular_offset,
                        f"{DERIVED_PREFIX}vertical_offset": float(delta[2]),
                        f"{DERIVED_PREFIX}expected_left_contact_margin": half - closing_offset,
                        f"{DERIVED_PREFIX}expected_right_contact_margin": half + closing_offset,
                    }
                )
            )
        if self._last_object_bbox is not None and self._last_object_center_pixel is not None:
            x0, y0, x1, y1 = self._last_object_bbox
            width = max(x1 - x0, 1)
            height = max(y1 - y0, 1)
            normalized = (
                (self._last_object_center_pixel[0] - x0) / width,
                (self._last_object_center_pixel[1] - y0) / height,
            )
            for row in self.trace_rows:
                row[f"{DERIVED_PREFIX}visible_object_bounding_box"] = self._last_object_bbox
                row[f"{DERIVED_PREFIX}grasp_point_normalized_in_visible_bbox"] = normalized

        for stage in {str(row.get("stage")) for row in self.perception_rows}:
            for component in {str(row.get("component")) for row in self.perception_rows}:
                selected = [
                    row
                    for row in self.perception_rows
                    if row.get("stage") == stage
                    and row.get("component") == component
                    and row.get("provider_position") is not None
                ]
                if not selected:
                    continue
                positions = np.asarray(
                    [row["provider_position"] for row in selected], dtype=float
                )
                median = np.median(positions, axis=0)
                spread = float(np.max(np.linalg.norm(positions - median, axis=1)))
                for row in selected:
                    row[f"{DERIVED_PREFIX}position_spread"] = spread

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
        if provider_call_count is not None and int(provider_call_count) != self.provider_call_count:
            self.errors.append(
                f"Provider wrapper counted {provider_call_count}, recorder counted {self.provider_call_count}"
            )
        self._enrich_centering()
        for milestone in MILESTONES:
            if milestone not in self._captured:
                self.frame_records.append(
                    {
                        "seed": self.seed,
                        "method": self.method.method_id,
                        "milestone": milestone,
                        "status": "not_reached",
                        "view": None,
                        "source": None,
                        "simulation_time": None,
                        "path": None,
                    }
                )
        if self.errors:
            raise RuntimeError(
                "Passive diagnostic recorder failed after controller completion:\n"
                + "\n".join(self.errors)
            )

    @staticmethod
    def _max_abs(rows: Sequence[Mapping[str, Any]], field: str) -> float:
        values = [abs(float(row[field])) for row in rows if row.get(field) is not None]
        return max(values, default=0.0)

    def summary(self) -> dict[str, Any]:
        if self.result is None or self.fingerprint is None:
            raise RuntimeError("Diagnostic episode did not finish")
        candidate_rows = [
            row
            for row in self.trace_rows
            if row["event"] == "candidate_sample"
            and row.get(f"{CONTROLLER_PREFIX}grasp_state") == "grasp_candidate"
        ]
        candidate = candidate_rows[0] if candidate_rows else None
        confirmation = [
            row for row in self.trace_rows if row["event"] == "confirmation_sample"
        ]
        transfer = [
            row
            for row in self.trace_rows
            if row["stage"] in {"transfer", "descend_to_place"}
        ]
        final_confirmation = confirmation[-1] if confirmation else {}
        predicate_names = (
            "commanded_closing_predicate",
            "minimum_aperture_predicate",
            "contact_predicate",
            "lift_predicate",
            "aperture_retention_predicate",
            "collision_free_predicate",
            "combined_predicate",
        )
        final_false = [
            name
            for name in predicate_names
            if final_confirmation.get(f"{CONTROLLER_PREFIX}{name}") is False
        ]
        apertures = [
            float(row[f"{CONTROLLER_PREFIX}gripper_aperture"])
            for row in self.trace_rows
            if row.get(f"{CONTROLLER_PREFIX}gripper_aperture") is not None
        ]
        candidate_aperture = (
            None
            if candidate is None
            else candidate.get(f"{CONTROLLER_PREFIX}candidate_aperture")
        )
        final_aperture = apertures[-1] if apertures else None
        first_delta = None
        if self._first_left_contact_time is not None and self._first_right_contact_time is not None:
            first_delta = self._first_right_contact_time - self._first_left_contact_time
        expected_provider_calls = sum(
            int(value or 0)
            for value in (
                self.result.initial_perception_frame_count,
                self.result.pregrasp_perception_frame_count,
                self.result.final_visual_frame_count,
            )
        )
        return _finite(
            {
                "seed": self.seed,
                "method": self.method.method_id,
                "pair_id": self.pair_id,
                "execution_index": self.execution_index,
                "problem_family": self.candidate["problem_family"],
                "selection_role": self.candidate["selection_role"],
                "formal_pair_category": self.candidate["pair_category"],
                "episode_fingerprint": self.fingerprint.digest,
                "final_stage": self.result.final_stage,
                "failure_reason": self.result.failure_reason,
                "controller_reported_success": self.result.controller_reported_success,
                "privileged_ground_truth_success": self.result.privileged_ground_truth_success,
                "placement_success": bool(self.result.privileged_ground_truth_success),
                "safe_task_success": bool(
                    self.result.privileged_ground_truth_success
                    and self.result.collision_count == 0
                    and self.result.simulation_time <= self.env.config.simulation.episode_timeout
                ),
                "collision": bool(self.result.collision_count),
                "collision_count": self.result.collision_count,
                "simulation_time": self.result.simulation_time,
                "candidate_aperture": candidate_aperture,
                "final_recorded_aperture": final_aperture,
                "maximum_aperture_drop": (
                    None
                    if candidate_aperture is None or not apertures
                    else max(float(candidate_aperture) - value for value in apertures)
                ),
                "maximum_absolute_aperture_velocity": self._max_abs(
                    self.trace_rows, f"{CONTROLLER_PREFIX}gripper_aperture_velocity"
                ),
                "maximum_confirmation_hold_steps": max(
                    (
                        int(row.get(f"{CONTROLLER_PREFIX}confirmation_hold_steps") or 0)
                        for row in confirmation
                    ),
                    default=0,
                ),
                "final_false_predicates": final_false,
                "first_left_contact_time": self._first_left_contact_time,
                "first_right_contact_time": self._first_right_contact_time,
                "first_contact_time_delta_right_minus_left": first_delta,
                "first_contact_surface": self._first_contact_classification,
                "candidate_world_xy_centering_offset": (
                    None if candidate is None else candidate.get(f"{DERIVED_PREFIX}world_xy_centering_offset")
                ),
                "candidate_closing_axis_offset": (
                    None if candidate is None else candidate.get(f"{DERIVED_PREFIX}closing_axis_offset")
                ),
                "candidate_perpendicular_offset": (
                    None if candidate is None else candidate.get(f"{DERIVED_PREFIX}perpendicular_horizontal_offset")
                ),
                "candidate_orientation_mismatch": (
                    None if candidate is None else candidate.get(f"{DERIVED_PREFIX}cube_symmetry_reduced_closing_axis_angle")
                ),
                "candidate_visible_edge_angle": (
                    None if candidate is None else candidate.get(f"{DERIVED_PREFIX}visible_edge_angle")
                ),
                "maximum_relative_slip": self._max_abs(
                    self.trace_rows, f"{DERIVED_PREFIX}relative_translation_slip_from_candidate"
                ),
                "maximum_relative_rotation_slip": self._max_abs(
                    self.trace_rows, f"{DERIVED_PREFIX}relative_rotation_slip_from_candidate"
                ),
                "maximum_object_orientation_change": self._max_abs(
                    self.trace_rows, f"{DERIVED_PREFIX}object_orientation_change_angle"
                ),
                "maximum_yaw_change": self._max_abs(self.trace_rows, f"{DERIVED_PREFIX}object_yaw_change"),
                "maximum_roll_change": self._max_abs(self.trace_rows, f"{DERIVED_PREFIX}object_roll_change"),
                "maximum_pitch_change": self._max_abs(self.trace_rows, f"{DERIVED_PREFIX}object_pitch_change"),
                "face_flip_approximately_90_degrees": bool(
                    math.radians(75)
                    <= self._max_abs(self.trace_rows, f"{DERIVED_PREFIX}object_orientation_change_angle")
                    <= math.radians(105)
                ),
                "maximum_object_drop_after_peak": self._max_abs(
                    self.trace_rows, f"{PRIVILEGED_PREFIX}object_drop_after_peak"
                ),
                "maximum_tcp_acceleration": self._max_abs(
                    transfer, f"{DERIVED_PREFIX}tcp_acceleration"
                ),
                "minimum_table_clearance_conservative": min(
                    (
                        float(row[f"{PRIVILEGED_PREFIX}minimum_table_clearance_conservative"])
                        for row in self.trace_rows
                    ),
                    default=None,
                ),
                "actual_target_ik_reachability": (
                    "ik_not_converged"
                    if self.result.failure_reason == "ik_not_converged"
                    else "actual_planned_orientation_reached_or_failure_elsewhere"
                ),
                "provider_call_count": self.provider_call_count,
                "expected_provider_call_count_from_controller_result": expected_provider_calls,
                "provider_call_count_unchanged": self.provider_call_count == expected_provider_calls,
                "detector_call_count": self.detector_call_count,
                "expected_detector_call_count": (
                    2 * self.provider_call_count if self.method.method_id == "b1_vision" else 0
                ),
                "detector_call_count_unchanged": self.detector_call_count
                == (2 * self.provider_call_count if self.method.method_id == "b1_vision" else 0),
                "render_state_checks": self._render_checks,
                "render_state_unchanged": self._render_state_unchanged,
                "trace_row_count": len(self.trace_rows),
                "contact_row_count": len(self.contact_rows),
                "perception_row_count": len(self.perception_rows),
                "frame_record_count": len(self.frame_records),
                "online_pretransfer_fields": [
                    "gripper_aperture",
                    "gripper_aperture_velocity",
                    "left_contact",
                    "right_contact",
                    "bilateral_contact",
                    "contact_hold_steps",
                    "candidate_aperture",
                    "aperture_drop",
                    "confirmation_predicates",
                    "RGB-D detection validity/confidence/spread",
                ],
                "online_only_after_failure_fields": [
                    "sustained contact loss",
                    "grasp_lost state",
                    "final visual rejection",
                ],
                "privileged_diagnostic_only_fields": [
                    "object pose/velocity",
                    "contact position/normal/solver force",
                    "relative object-gripper transform",
                    "ground-truth centering and slip",
                ],
            }
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class DevelopmentDiagnosticSession:
    def __init__(
        self,
        output_dir: Path,
        *,
        visualization_enabled: bool,
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        self.output_dir = output_dir
        self.visualization_enabled = visualization_enabled
        self.candidates = {int(item["seed"]): dict(item) for item in candidates}
        self.recorders: list[DevelopmentEpisodeRecorder] = []

    def start_episode(self, *, env: Any, method: MethodSpec, seed: int, pair_id: str, execution_index: int) -> DevelopmentEpisodeRecorder:
        recorder = DevelopmentEpisodeRecorder(
            env=env,
            method=method,
            seed=seed,
            pair_id=pair_id,
            execution_index=execution_index,
            output_dir=self.output_dir,
            visualization_enabled=self.visualization_enabled,
            candidate=self.candidates[int(seed)],
        )
        self.recorders.append(recorder)
        return recorder

    def episode_summaries(self) -> list[dict[str, Any]]:
        return [
            recorder.summary()
            for recorder in self.recorders
            if recorder.result is not None and recorder.fingerprint is not None
        ]

    def trace_rows(self) -> list[dict[str, Any]]:
        return [row for recorder in self.recorders for row in recorder.trace_rows]

    def contact_rows(self) -> list[dict[str, Any]]:
        return [row for recorder in self.recorders for row in recorder.contact_rows]

    def perception_rows(self) -> list[dict[str, Any]]:
        return [row for recorder in self.recorders for row in recorder.perception_rows]

    def frame_records(self) -> list[dict[str, Any]]:
        return [row for recorder in self.recorders for row in recorder.frame_records]


def _formal_rows(run_dir: Path) -> dict[tuple[int, str], dict[str, str]]:
    with (run_dir / "episodes.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(int(row["seed"]), row["method_id"]): row for row in rows}


def _replay_comparison(
    summaries: Sequence[Mapping[str, Any]],
    executions: Sequence[Any],
    run_dir: Path,
) -> dict[str, Any]:
    archived = _formal_rows(run_dir)
    execution_by_key = {
        (int(item.seed), str(item.method.method_id)): item for item in executions
    }
    comparisons: list[dict[str, Any]] = []
    for summary in summaries:
        key = (int(summary["seed"]), str(summary["method"]))
        expected = archived.get(key)
        execution = execution_by_key.get(key)
        if expected is None or execution is None or execution.result is None:
            raise ValueError(f"Formal D0 reference or replay execution is missing for {key}")
        result = execution.result
        exact_checks = {
            "seed": int(result.seed) == int(expected["seed"]),
            "method": key[1] == expected["method_id"],
            "episode_fingerprint": summary["episode_fingerprint"] == expected["episode_fingerprint"],
            "sampled_pick_position": list(result.sampled_pick_position or ()) == _json_cell(expected["sampled_pick_position"]),
            "sampled_place_position": list(result.sampled_place_position or ()) == _json_cell(expected["sampled_place_position"]),
            "sampled_mass": result.sampled_mass == float(expected["sampled_mass"]),
            "sampled_friction": list(result.sampled_friction or ()) == _json_cell(expected["sampled_friction"]),
            "pick_region": result.pick_region == expected["pick_region"],
            "place_region": result.place_region == expected["place_region"],
            "final_stage": result.final_stage == expected["final_stage"],
            "failure_reason": (result.failure_reason or "") == expected["failure_reason"],
            "controller_reported_success": result.controller_reported_success == _bool_cell(expected["controller_reported_success"]),
            "privileged_ground_truth_success": result.privileged_ground_truth_success == _bool_cell(expected["privileged_ground_truth_success"]),
            "safe_task_success": summary["safe_task_success"] == _bool_cell(expected["safe_task_success"]),
            "placement_success": summary["placement_success"] == _bool_cell(expected["placement_success"]),
            "collision_count": int(result.collision_count) == int(expected["collision_count"]),
        }
        time_difference = abs(float(result.simulation_time) - float(expected["simulation_time"]))
        time_match = time_difference <= 0.0020000001
        comparisons.append(
            {
                "seed": key[0],
                "method": key[1],
                "exact_checks": exact_checks,
                "simulation_time_check": time_match,
                "simulation_time_tolerance_seconds": 0.0020000001,
                "formal_simulation_time": float(expected["simulation_time"]),
                "replay_simulation_time": float(result.simulation_time),
                "simulation_time_absolute_difference": time_difference,
                "all_behavior_fields_match": all(exact_checks.values()) and time_match,
            }
        )
    return {
        "reference": "formal Development D0 episodes.csv",
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "all_episodes_match": len(comparisons) == 20
        and all(item["all_behavior_fields_match"] for item in comparisons),
    }


def _mechanism_comparison(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_seed_method = {
        f"{item['seed']}:{item['method']}": dict(item) for item in summaries
    }
    shared_fields = (
        "failure_reason",
        "candidate_world_xy_centering_offset",
        "candidate_closing_axis_offset",
        "candidate_perpendicular_offset",
        "candidate_orientation_mismatch",
        "first_contact_surface",
        "first_contact_time_delta_right_minus_left",
        "candidate_aperture",
        "maximum_aperture_drop",
        "maximum_relative_slip",
        "maximum_object_orientation_change",
        "maximum_object_drop_after_peak",
        "final_false_predicates",
        "maximum_confirmation_hold_steps",
    )
    shared = {
        key: {field: value.get(field) for field in shared_fields}
        for key, value in by_seed_method.items()
        if int(value["seed"]) in SHARED_GRASP_SEEDS
    }
    perception = {
        key: {
            "failure_reason": value["failure_reason"],
            "provider_call_count": value["provider_call_count"],
            "detector_call_count": value["detector_call_count"],
            "privileged_ground_truth_success": value["privileged_ground_truth_success"],
        }
        for key, value in by_seed_method.items()
        if int(value["seed"]) in PERCEPTION_FOCUS_SEEDS
    }
    transfer = {
        key: {
            "failure_reason": value["failure_reason"],
            "maximum_relative_slip": value["maximum_relative_slip"],
            "maximum_object_drop_after_peak": value["maximum_object_drop_after_peak"],
            "maximum_aperture_drop": value["maximum_aperture_drop"],
            "maximum_tcp_acceleration": value["maximum_tcp_acceleration"],
            "controller_reported_success": value["controller_reported_success"],
            "privileged_ground_truth_success": value["privileged_ground_truth_success"],
        }
        for key, value in by_seed_method.items()
        if int(value["seed"]) in TRANSFER_FOCUS_SEEDS
    }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "shared_grasp": shared,
        "perception": perception,
        "transfer": transfer,
        "cube_orientation_interpretation": (
            "The object is a cube. Yaw and horizontal principal axes are equivalent "
            "modulo 90 degrees; the reported mismatch is symmetry-reduced and must "
            "not be interpreted as a unique object yaw. Face/edge/corner contact and "
            "relative gripper orientation remain diagnostic."
        ),
        "decision": None,
        "decision_status": "pending_structured_visual_review",
    }


def _perception_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in (3301, 2574, 1297, 254, 3170):
        selected = [
            row
            for row in rows
            if int(row["seed"]) == seed
            and row["method"] == "b1_vision"
            and row.get("component") == "object"
        ]
        result[str(seed)] = {
            "sample_count": len(selected),
            "stages": sorted({str(row["stage"]) for row in selected}),
            "rejection_reason_counts": {
                str(reason): sum(1 for row in selected if row.get("rejection_reason") == reason)
                for reason in sorted({row.get("rejection_reason") for row in selected}, key=str)
            },
            "mask_pixel_counts": [row.get("mask_pixel_count") for row in selected],
            "confidence": [row.get("confidence") for row in selected],
            "occlusion_diagnostics": [row.get(f"{DERIVED_PREFIX}occlusion_diagnostic") for row in selected],
            "position_errors": [row.get(f"{DERIVED_PREFIX}position_error") for row in selected],
        }
    return result


def _b2_evidence(
    summaries: Sequence[Mapping[str, Any]],
    perception_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    success_controls = [
        item
        for item in summaries
        if item["method"] == "b1_vision"
        and int(item["seed"]) in {1557, 254}
        and item["controller_reported_success"]
    ]
    failed_grasps = [
        item
        for item in summaries
        if item["method"] == "b1_vision"
        and int(item["seed"]) in {2225, 3989}
    ]
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "decision": None,
        "decision_status": "pending_structured_visual_review",
        "b2_implemented": False,
        "formal_b2_config_generated": False,
        "centering": {
            "failed_seed_offsets": [item["candidate_world_xy_centering_offset"] for item in failed_grasps],
            "successful_control_offsets": [item["candidate_world_xy_centering_offset"] for item in success_controls],
            "interpretation_pending_visual_review": True,
        },
        "orientation": {
            "failed_seed_symmetry_reduced_angles": [item["candidate_orientation_mismatch"] for item in failed_grasps],
            "failed_seed_contact_surfaces": [item["first_contact_surface"] for item in failed_grasps],
            "cube_principal_axis_unique": False,
            "interpretation_pending_visual_review": True,
        },
        "quality_gate": {
            "failed_seed_final_false_predicates": [item["final_false_predicates"] for item in failed_grasps],
            "failed_seed_aperture_drops": [item["maximum_aperture_drop"] for item in failed_grasps],
            "successful_control_aperture_drops": [item["maximum_aperture_drop"] for item in success_controls],
            "gate_can_only_reject_or_trigger_retry": True,
            "gate_cannot_generate_a_better_grasp": True,
            "interpretation_pending_visual_review": True,
        },
        "perception": _perception_evidence(perception_rows),
        "availability_boundary": {
            "online_observable_before_transfer": [
                "aperture and aperture velocity",
                "debounced left/right/bilateral contact",
                "contact timing and hold counters",
                "candidate/confirmation predicate values",
                "existing RGB-D masks, confidence, validity, and position spread",
            ],
            "online_observable_only_after_failure": [
                "sustained transfer contact loss",
                "grasp_lost transition",
                "final visual rejection",
            ],
            "privileged_diagnostic_only": [
                "ground-truth object pose and velocity",
                "contact location/normal/solver force",
                "ground-truth centering, slip, and orientation change",
            ],
        },
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


def _render_report(
    *,
    manifest: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    formal_unchanged: bool,
) -> str:
    lines = [
        "# Development D0.5 Passive Diagnostics and B2 Mechanism Disambiguation",
        "",
        "## 身份与边界",
        "",
        "- 本运行是 10 个既有 Development seed 的定向机制诊断，不是正式成功率评测。",
        "- 20 个 replay 不进入 Development 60、production metrics 或 pair 分类。",
        "- 正式 B1 safe task success 仍为 34/60；B0 仍为 39/60。",
        "- 未修改冻结 B1、protocol 或 split；未运行 Development 100 或 Held-out；未实现 B2。",
        f"- execution commit：{manifest['git_commit']}",
        f"- frozen config SHA-256：{manifest['config_sha256']}",
        f"- protocol：{manifest['protocol_id']} / {manifest['protocol_version']}",
        f"- seeds：{_fmt(manifest['seeds'])}",
        "",
        "## 完整性与行为复现",
        "",
        f"- completed pairs：{manifest['completed_pairs']}/{manifest['requested_pairs']}",
        f"- completed episodes：{manifest['completed_episodes']}/20",
        f"- invalid pairs：{manifest['invalid_pairs']}",
        f"- program errors：{manifest['program_errors']}",
        f"- 与正式 D0 稳定行为字段一致：{_fmt(replay['all_episodes_match'])}",
        f"- 正式 D0 全目录 hash 不变：{_fmt(formal_unchanged)}",
        "- simulation time 只使用仓库既有的 0.0020000001 s 单步容差；其余指定行为字段精确比较。",
        "",
        "## Episode 机制摘要",
        "",
        "| seed | method | formal role | result | center XY m | closing m | perpendicular m | symmetry angle rad | first contact | candidate m | max drop m | slip m | rotation rad | false predicates | hold |",
        "|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item["seed"]),
                    str(item["method"]),
                    str(item["selection_role"]),
                    str(item["failure_reason"] or "success"),
                    _fmt(item["candidate_world_xy_centering_offset"]),
                    _fmt(item["candidate_closing_axis_offset"]),
                    _fmt(item["candidate_perpendicular_offset"]),
                    _fmt(item["candidate_orientation_mismatch"]),
                    _fmt(item["first_contact_surface"]),
                    _fmt(item["candidate_aperture"]),
                    _fmt(item["maximum_aperture_drop"]),
                    _fmt(item["maximum_relative_slip"]),
                    _fmt(item["maximum_object_orientation_change"]),
                    _fmt(item["final_false_predicates"]),
                    str(item["maximum_confirmation_hold_steps"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 数据隔离",
            "",
            "- `controller_observable.*`：控制器本来已经计算的 aperture、contact、predicate、hold、TCP 与 provider estimate。",
            "- `privileged_diagnostic.*`：runner-side 只读 object/contact/solver/pose truth；从未返回 controller。",
            "- `derived_diagnostic.*`：由上述只读样本离线计算的 centering、相对 slip、orientation、mask 几何与 motion 指标。",
            "- provider/detector 调用次数与 EpisodeResult 既有 frame count 逐 episode 交叉核对；未新增调用。",
            "- observer 只在既有 step 后执行；没有额外 reset、simulation step 或控制循环。",
            "- 诊断渲染逐次验证 time/qpos/qvel/ctrl bitwise 不变。",
            "",
            "## Cube orientation 解释",
            "",
            "物体是立方体，水平 yaw 与主轴只在 90° 模意义下可辨。报告只给出 symmetry-reduced face alignment；不虚构唯一 yaw。face/edge/corner 接触、物体翻面及相对夹爪姿态仍有诊断意义。",
            "",
            "## 结构化机制结论",
            "",
            "当前状态：等待实际图像审查后通过同一工具写入唯一 M-* 结论。不得把本节的待审状态解释为 B2 路线已经确定。",
            "",
            "## 结论边界",
            "",
            "D0.5 不能修改正式 Development 34/60，不能与 D0 合并，不能替代 Held-out，也不能证明 B2 最终性能。候选问题族有重叠，不能把 failure count 相加。centering、orientation 与 gate 可能处于同一因果链的不同层；本任务只建议 B2.1 首个设计问题，不选择最终完整算法。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifests(output_path: Path, manifest: dict[str, Any]) -> None:
    artifact_path = output_path / "artifact_manifest.json"
    hashes = {
        path.relative_to(output_path).as_posix(): sha256_file(path)
        for path in sorted(output_path.rglob("*"))
        if path.is_file()
        and path.name not in {"artifact_manifest.json", "run_manifest.json"}
    }
    artifact = {
        "schema_version": "1.0.0",
        "hash_algorithm": "sha256",
        "excluded_self": "artifact_manifest.json",
        "excluded_run_manifest": "run_manifest.json",
        "artifacts": hashes,
        "artifact_set_sha256": _canonical_sha256(hashes),
        "self_check_pass": all(
            sha256_file(output_path / relative) == digest
            for relative, digest in hashes.items()
        ),
    }
    write_json(artifact_path, artifact)
    all_hashes = {
        path.relative_to(output_path).as_posix(): sha256_file(path)
        for path in sorted(output_path.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest["artifact_hashes"] = all_hashes
    manifest["artifact_set_sha256"] = _canonical_sha256(all_hashes)
    manifest["artifact_manifest_self_check_pass"] = artifact["self_check_pass"]
    write_json(output_path / "run_manifest.json", manifest)


def finalize_mechanism_review(
    output_dir: str | Path,
    review_file: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_dir).expanduser().resolve()
    review_path = Path(review_file).expanduser().resolve()
    if output_path != OUTPUT_PATH or not output_path.is_dir():
        raise ValueError("Structured review must target the registered D0.5 output")
    review = _strict_json(review_path)
    if not isinstance(review, Mapping):
        raise ValueError("Mechanism review must be a JSON object")
    decision = review.get("decision")
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    if review.get("b1_modified") is not False:
        raise ValueError("Review must state b1_modified=false")
    if review.get("b2_implemented") is not False:
        raise ValueError("Review must state b2_implemented=false")
    if review.get("held_out_accessed") is not False:
        raise ValueError("Review must state held_out_accessed=false")
    if not isinstance(review.get("b2_1_first_problem"), str):
        raise ValueError("Review must name the B2.1 first design problem")

    manifest_path = output_path / "run_manifest.json"
    manifest = _strict_json(manifest_path)
    if not (
        manifest.get("completed_pairs") == 10
        and manifest.get("completed_episodes") == 20
        and manifest.get("invalid_pairs") == 0
        and manifest.get("program_errors") == 0
        and manifest.get("behavior_replay_match") is True
        and manifest.get("formal_d0_unchanged") is True
    ):
        raise ValueError("Cannot finalize an incomplete or behavior-inconsistent D0.5 run")
    mechanism_path = output_path / "mechanism_comparison.json"
    evidence_path = output_path / "b2_mechanism_evidence.json"
    mechanism = _strict_json(mechanism_path)
    evidence = _strict_json(evidence_path)
    mechanism["decision"] = decision
    mechanism["decision_status"] = "final_structured_visual_review"
    mechanism["structured_review"] = dict(review)
    evidence["decision"] = decision
    evidence["decision_status"] = "final_structured_visual_review"
    evidence["b2_1_first_problem"] = review["b2_1_first_problem"]
    evidence["later_b2_candidates"] = review.get("later_b2_candidates", [])
    evidence["rationale"] = review.get("rationale")
    evidence["per_seed_findings"] = review.get("per_seed_findings", {})
    write_json(mechanism_path, mechanism)
    write_json(evidence_path, evidence)

    report_path = output_path / "development_d0_5_report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.split("## Final structured visual review", 1)[0].rstrip()
    report += "\n\n" + "\n".join(
        [
            "## Final structured visual review",
            "",
            f"- 唯一主结论：{decision}",
            f"- B2.1 首个设计问题：{review['b2_1_first_problem']}",
            f"- 后续 B2.x 候选：{_fmt(review.get('later_b2_candidates', []))}",
            f"- 理由：{review.get('rationale', '—')}",
            f"- 实际人工审查：{review.get('visual_review_scope', '—')}",
            "",
            "### Per-seed findings",
            "",
            "```json",
            json.dumps(review.get("per_seed_findings", {}), ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "这只是 B2.1 首个设计问题的证据建议，不是最终算法选择，也不宣布 B2 路线已经确定。",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8", newline="\n")
    manifest["mechanism_decision"] = decision
    manifest["structured_visual_review_finalized"] = True
    manifest["b2_1_first_problem"] = review["b2_1_first_problem"]
    _write_artifact_manifests(output_path, manifest)
    return dict(evidence)


def run_development_diagnostics(
    *,
    protocol: ProtocolConfig,
    config_path: str | Path,
    development_run_dir: str | Path,
    candidate_file: str | Path,
    output_dir: str | Path,
    method_ids: Sequence[str] = REQUIRED_METHODS,
    diagnostics_enabled: bool,
    visualization_enabled: bool,
    require_traceable_source: bool,
    command: Sequence[str] | None = None,
    fingerprint_atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> DevelopmentDiagnosticRunResult:
    if not diagnostics_enabled:
        raise ValueError("Development D0.5 requires --diagnostics-enabled")
    if not visualization_enabled:
        raise ValueError("Development D0.5 requires --visualization-artifacts-enabled")
    if not require_traceable_source:
        raise ValueError("Development D0.5 requires --require-traceable-source")
    output_path = Path(output_dir).expanduser().resolve()
    if output_path != OUTPUT_PATH:
        raise ValueError(f"D0.5 output must be exactly {OUTPUT_PATH}")
    config, seeds, methods, candidates = validate_development_diagnostic_request(
        protocol=protocol,
        config_path=config_path,
        development_run_dir=development_run_dir,
        candidate_file=candidate_file,
        method_ids=method_ids,
    )
    repository = repository_metadata(PROJECT_ROOT)
    if repository.get("git_dirty"):
        raise BenchmarkRunError("Development D0.5 requires a clean checkpoint worktree")
    source = _source_provenance(PROJECT_ROOT)
    formal_before = _directory_hashes(FORMAL_D0_PATH)
    protected_before = _protected_hashes()
    _prepare_output_dir(output_path, overwrite=False)
    shutil.copyfile(FROZEN_CONFIG_PATH, output_path / "diagnostic_config_snapshot.toml")
    shutil.copyfile(PROTOCOL_PATH, output_path / "protocol_snapshot.toml")
    write_json(output_path / "source_provenance.json", source)
    write_json(
        output_path / "candidate_snapshot.json",
        {
            "diagnostic_only": True,
            "formal_split": False,
            "reference_split": "development",
            "used_for_formal_success_rate": False,
            "candidate_file": str(CANDIDATE_PATH),
            "candidate_file_sha256": sha256_file(CANDIDATE_PATH),
            "seed_snapshot_file": str(SEED_SNAPSHOT_PATH),
            "seed_snapshot_sha256": sha256_file(SEED_SNAPSHOT_PATH),
            "seeds": list(seeds),
            "candidates": candidates,
        },
    )
    write_json(
        output_path / "formal_d0_reference_hashes.json",
        {
            "formal_d0_path": str(FORMAL_D0_PATH),
            "before": formal_before,
            "after": None,
            "unchanged": None,
        },
    )
    logger, log_handler = _logger_for(output_path)
    session = DevelopmentDiagnosticSession(
        output_path,
        visualization_enabled=visualization_enabled,
        candidates=candidates,
    )
    manifest: dict[str, Any] = {
        "run_kind": RUN_KIND,
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "formal_metrics_included": False,
        "production_metrics_generated": False,
        "development_run": False,
        "development_100_run": False,
        "held_out_data_read": False,
        "held_out_test_run": False,
        "b2_implemented": False,
        "b1_modified": False,
        "automatic_algorithm_selection": False,
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
        "protocol_sha256": protocol.sha256,
        "config_sha256": sha256_file(FROZEN_CONFIG_PATH),
        "formal_d0_execution_commit": EXPECTED_EXECUTION_COMMIT,
        "formal_d0_analysis_commit": EXPECTED_ANALYSIS_COMMIT,
        "formal_d0_path": str(FORMAL_D0_PATH),
        "candidate_file_sha256": sha256_file(CANDIDATE_PATH),
        "development_split_sha256": sha256_file(DEVELOPMENT_SPLIT_PATH),
        "seeds": list(seeds),
        "methods": list(REQUIRED_METHODS),
        "method_execution_order": list(REQUIRED_METHODS),
        "requested_pairs": 10,
        "requested_episodes": 20,
        "completed_pairs": 0,
        "completed_episodes": 0,
        "b0_episode_count": 0,
        "b1_episode_count": 0,
        "invalid_pairs": 0,
        "program_errors": 0,
        "program_error_details": [],
        "behavior_replay_match": False,
        "formal_d0_unchanged": False,
        "provider_calls_unchanged": False,
        "detector_calls_unchanged": False,
        "simulation_steps_unchanged": False,
        "render_state_unchanged": False,
        "structured_visual_review_finalized": False,
        "mechanism_decision": None,
    }
    executions: list[Any] = []
    pair_rows: list[dict[str, Any]] = []
    fatal_error: str | None = None
    execution_index = 0
    try:
        logger.info(
            "development_d0_5_start pairs=%s methods=%s seeds=%s",
            len(seeds),
            list(REQUIRED_METHODS),
            list(seeds),
        )
        for pair_index, seed in enumerate(seeds):
            pair_id = f"development_d0_5_pair_{pair_index:04d}_seed_{seed}"
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
                    manifest[f"{method.method_id.split('_')[0]}_episode_count"] += 1
                if execution.program_error:
                    manifest["program_errors"] += 1
                    manifest["program_error_details"].append(
                        {
                            "seed": seed,
                            "method": method.method_id,
                            "error": execution.program_error,
                        }
                    )
                    fatal_error = "A specified diagnostic episode ended with a program error"
                    break
            if fatal_error:
                break
            oracle = pair_executions["b0_oracle"]
            vision = pair_executions["b1_vision"]
            pair_error = _validate_execution_pair(oracle, vision, atol=fingerprint_atol)
            valid = pair_error is None
            oracle.pair_valid = valid
            vision.pair_valid = valid
            pair_rows.append(_paired_row(pair_id, seed, oracle, vision, pair_error))
            if valid:
                manifest["completed_pairs"] += 1
            else:
                manifest["invalid_pairs"] += 1
                fatal_error = pair_error or "Invalid diagnostic pair"
                break
    except Exception:
        fatal_error = traceback.format_exc()
        manifest["program_errors"] += 1
        manifest["program_error_details"].append(
            {"seed": None, "method": None, "error": fatal_error}
        )
        logger.exception("development_d0_5_program_error")

    try:
        summaries = session.episode_summaries()
        episode_rows = [
            _episode_row(
                execution,
                protocol=protocol,
                split_name="development_d0_5_diagnostic_reference",
                config_sha256=manifest["config_sha256"],
                code_commit=str(repository.get("git_commit") or ""),
            )
            for execution in executions
        ]
        formal_rows = _formal_rows(FORMAL_D0_PATH)
        summary_by_key = {(row["seed"], row["method"]): row for row in summaries}
        for row in episode_rows:
            key = (int(row["seed"]), str(row["method_id"])) if row.get("seed") is not None else None
            summary = summary_by_key.get(key) if key is not None else None
            reference = formal_rows.get(key) if key is not None else None
            if summary is not None:
                row.update({f"diagnostic.{name}": value for name, value in summary.items()})
            if reference is not None:
                row.update(
                    {
                        "formal_d0.pair_id": reference["pair_id"],
                        "formal_d0.failure_reason": reference["failure_reason"],
                        "formal_d0.final_stage": reference["final_stage"],
                        "formal_d0.safe_task_success": reference["safe_task_success"],
                        "formal_d0.controller_reported_success": reference["controller_reported_success"],
                        "formal_d0.privileged_ground_truth_success": reference["privileged_ground_truth_success"],
                    }
                )
        write_csv(
            output_path / "episode_diagnostics.csv",
            episode_rows,
            episode_fieldnames(episode_rows),
        )
        if pair_rows:
            write_csv(output_path / "paired_results.csv", pair_rows, PAIRED_RESULT_FIELDS)
        trace_rows = session.trace_rows()
        contact_rows = session.contact_rows()
        perception_rows = session.perception_rows()
        write_csv(
            output_path / "diagnostic_trace.csv",
            trace_rows,
            tuple(sorted({key for row in trace_rows for key in row})),
        )
        write_csv(
            output_path / "contact_trace.csv",
            contact_rows,
            tuple(sorted({key for row in contact_rows for key in row})),
        )
        write_csv(
            output_path / "perception_trace.csv",
            perception_rows,
            tuple(sorted({key for row in perception_rows for key in row})),
        )
        write_csv(
            output_path / "mechanism_summary.csv",
            summaries,
            tuple(sorted({key for row in summaries for key in row})),
        )
        write_json(output_path / "frames_manifest.json", session.frame_records())

        replay = _replay_comparison(summaries, executions, FORMAL_D0_PATH)
        formal_after = _directory_hashes(FORMAL_D0_PATH)
        protected_after = _protected_hashes()
        formal_unchanged = formal_before == formal_after
        protected_unchanged = protected_before == protected_after
        write_json(
            output_path / "formal_d0_reference_hashes.json",
            {
                "formal_d0_path": str(FORMAL_D0_PATH),
                "before": formal_before,
                "after": formal_after,
                "unchanged": formal_unchanged,
                "protected_inputs_before": protected_before,
                "protected_inputs_after": protected_after,
                "protected_inputs_unchanged": protected_unchanged,
            },
        )
        manifest["behavior_replay_match"] = bool(replay["all_episodes_match"])
        manifest["formal_d0_unchanged"] = formal_unchanged
        manifest["protected_inputs_unchanged"] = protected_unchanged
        manifest["provider_calls_unchanged"] = bool(
            summaries and all(row["provider_call_count_unchanged"] for row in summaries)
        )
        manifest["detector_calls_unchanged"] = bool(
            summaries and all(row["detector_call_count_unchanged"] for row in summaries)
        )
        manifest["simulation_steps_unchanged"] = bool(replay["all_episodes_match"])
        manifest["render_state_unchanged"] = bool(
            summaries and all(row["render_state_unchanged"] for row in summaries)
        )
        write_json(output_path / "replay_comparison.json", replay)
        write_json(output_path / "mechanism_comparison.json", _mechanism_comparison(summaries))
        write_json(output_path / "b2_mechanism_evidence.json", _b2_evidence(summaries, perception_rows))
        (output_path / "development_d0_5_report.md").write_text(
            _render_report(
                manifest=manifest,
                summaries=summaries,
                replay=replay,
                formal_unchanged=formal_unchanged,
            ),
            encoding="utf-8",
            newline="\n",
        )
        if len(summaries) != 20:
            fatal_error = fatal_error or "D0.5 did not complete all 20 diagnostic episodes"
        if not replay["all_episodes_match"]:
            fatal_error = fatal_error or "D0.5 replay differs from formal Development D0"
        if not formal_unchanged or not protected_unchanged:
            fatal_error = fatal_error or "Protected formal inputs changed during D0.5"
        if not manifest["provider_calls_unchanged"] or not manifest["detector_calls_unchanged"]:
            fatal_error = fatal_error or "Passive diagnostics changed provider/detector call counts"
        if not manifest["render_state_unchanged"]:
            fatal_error = fatal_error or "Diagnostic rendering changed simulation state"
        if fatal_error and not manifest["program_errors"]:
            manifest["program_errors"] = 1
            manifest["program_error_details"].append(
                {"seed": None, "method": None, "error": fatal_error}
            )
    except Exception:
        output_error = traceback.format_exc()
        fatal_error = fatal_error or output_error
        manifest["program_errors"] += 1
        manifest["program_error_details"].append(
            {"seed": None, "method": None, "error": output_error}
        )
        logger.exception("development_d0_5_output_error")
    finally:
        try:
            logger.info(
                "development_d0_5_end completed_pairs=%s invalid_pairs=%s errors=%s",
                manifest["completed_pairs"],
                manifest["invalid_pairs"],
                manifest["program_errors"],
            )
            log_handler.flush()
            _write_artifact_manifests(output_path, manifest)
        except Exception:
            fatal_error = fatal_error or traceback.format_exc()
        logger.removeHandler(log_handler)
        log_handler.close()

    result = DevelopmentDiagnosticRunResult(
        output_dir=output_path,
        requested_pairs=10,
        completed_pairs=int(manifest["completed_pairs"]),
        invalid_pairs=int(manifest["invalid_pairs"]),
        program_errors=int(manifest["program_errors"]),
        exit_code=1 if fatal_error or manifest["invalid_pairs"] or manifest["program_errors"] else 0,
    )
    if fatal_error:
        raise BenchmarkRunError(
            "Development D0.5 stopped after writing traceable outputs: " + fatal_error
        )
    return result


__all__ = [
    "DECISIONS",
    "DevelopmentDiagnosticRunResult",
    "DevelopmentEpisodeRecorder",
    "REQUIRED_METHODS",
    "REQUIRED_SEEDS",
    "finalize_mechanism_review",
    "run_development_diagnostics",
    "validate_development_diagnostic_request",
]
