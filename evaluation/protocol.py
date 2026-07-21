from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from environments import EnvConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMVER_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
REQUIRED_CORE_METRICS = frozenset(
    {
        "safe_task_success_rate",
        "first_attempt_placement_success_rate",
        "collision_episode_rate",
        "safe_successful_simulation_time",
        "unexplained_failure_rate",
    }
)


class ProtocolValidationError(ValueError):
    """Raised when an Evaluation Protocol configuration is incomplete."""


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path
    size: int
    allows_b1_tuning: bool


@dataclass(frozen=True)
class CalibrationParameter:
    path: str
    current_value: Any
    unit: str
    legal_range: str
    responsibility: str
    calibration_allowed: bool
    required_evidence: str
    modifiable_after_freeze: bool


@dataclass(frozen=True)
class ProtocolConfig:
    path: Path
    raw: Mapping[str, Any]
    environment: EnvConfig
    protocol_id: str
    protocol_version: str
    environment_version: str
    metrics_schema_version: str
    split_id: str
    calibration_policy_id: str
    split_manifest_path: Path
    splits: Mapping[str, SplitSpec]
    core_metrics: tuple[str, ...]
    allowed_calibration_parameters: tuple[str, ...]
    baseline_frozen: bool
    episode_timeout: float
    released_stages: tuple[str, ...]
    declared_stages: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"Missing or invalid [{name}] section")
    return value


def _root_path(value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{field_name} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def config_value(config: EnvConfig, dotted_path: str) -> Any:
    current: Any = config
    for component in dotted_path.split("."):
        if not is_dataclass(current):
            raise ProtocolValidationError(
                f"Calibration parameter does not resolve through a dataclass: {dotted_path}"
            )
        names = {field.name for field in fields(current)}
        if component not in names:
            raise ProtocolValidationError(
                f"Calibration parameter does not exist: {dotted_path}"
            )
        current = getattr(current, component)
    return current


def _parameter_metadata(path: str, value: Any) -> CalibrationParameter:
    section, name = path.split(".", 1)
    if section == "perception":
        responsibility = "RGB-D color/depth segmentation and 3-D position estimation"
        evidence = "Calibration RGB-D masks, valid-frame counts, confidence, and position errors"
    elif section == "controller":
        responsibility = "Fixed-DLS solve, waypoint geometry, motion timing, or gripper command"
        evidence = "Calibration stage logs, pose/velocity errors, timeouts, and collision records"
    elif section == "b1":
        responsibility = "B1 event, contact, grasp, reacquisition, release, or final verification"
        evidence = "Calibration stage transitions, sensor traces, hold counts, and failure reasons"
    else:
        raise ProtocolValidationError(f"Unsupported calibration section: {section}")

    unit = "unitless"
    if "frames" in name or "steps" in name or "iterations" in name:
        unit = "count"
    elif "pixels" in name:
        unit = "pixels"
    elif "duration" in name or "timeout" in name:
        unit = "s"
    elif "orientation" in name:
        unit = "rad"
    elif any(
        token in name
        for token in (
            "depth",
            "height",
            "distance",
            "tolerance",
            "spread",
            "correction",
            "offset",
            "aperture",
            "lift",
            "joint_step",
        )
    ):
        unit = "m"
    elif name.endswith("_rgb"):
        unit = "RGB 0..255"
    elif name.endswith("_control"):
        unit = "actuator command"
    elif "velocity" in name:
        unit = "rad/s"

    legal = "must satisfy environments.config.validate_config"
    if isinstance(value, bool):
        legal = "boolean"
    elif isinstance(value, int):
        legal = "positive integer and cross-field protocol constraints"
    elif name.endswith("_rgb"):
        legal = "three values in [0, 255]"
    elif name.endswith("_world_z_range"):
        legal = "two finite metres with lower < upper"
    elif name.endswith("_observation_offset"):
        legal = "finite 3-vector with norm <= 0.20 m"
    elif name == "minimum_confidence":
        legal = "[0, 1]"
    elif "dominance_ratio" in name:
        legal = "> 1"
    elif name == "release_aperture_threshold":
        legal = "> minimum_grasp_aperture and <= 0.08 m"
    elif "aperture" in name:
        legal = "> 0 and <= 0.08 m, with configured ordering constraints"
    elif isinstance(value, (int, float)):
        legal = "finite and positive unless validate_config explicitly permits zero"

    return CalibrationParameter(
        path=path,
        current_value=value,
        unit=unit,
        legal_range=legal,
        responsibility=responsibility,
        calibration_allowed=True,
        required_evidence=evidence,
        modifiable_after_freeze=False,
    )


def calibration_parameter_catalog(protocol: ProtocolConfig) -> tuple[CalibrationParameter, ...]:
    return tuple(
        _parameter_metadata(path, config_value(protocol.environment, path))
        for path in protocol.allowed_calibration_parameters
    )


def load_protocol(
    path: str | Path,
    *,
    validate_splits: bool = True,
) -> ProtocolConfig:
    protocol_path = Path(path).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol configuration does not exist: {protocol_path}")
    with protocol_path.open("rb") as stream:
        raw = tomllib.load(stream)

    protocol_raw = _section(raw, "protocol")
    splits_raw = _section(raw, "splits")
    metrics_raw = _section(raw, "metrics")
    calibration_raw = _section(raw, "calibration")
    success_raw = _section(raw, "success")
    _section(raw, "collision")
    _section(raw, "program_error")
    _section(raw, "freeze")
    environment = load_config(protocol_path)

    split_names = ("calibration", "development", "held_out_test", "calibration_smoke")
    split_specs: dict[str, SplitSpec] = {}
    for name in split_names:
        path_key = f"{name}_path"
        size_key = f"{name}_size"
        if path_key not in splits_raw or size_key not in splits_raw:
            raise ProtocolValidationError(f"[splits] must define {path_key} and {size_key}")
        allows_tuning = name == "calibration"
        if name == "development":
            allows_tuning = bool(calibration_raw.get("development_allows_b1_tuning"))
        elif name == "held_out_test":
            allows_tuning = bool(calibration_raw.get("held_out_allows_b1_tuning"))
        split_specs[name] = SplitSpec(
            name=name,
            path=_root_path(splits_raw[path_key], f"splits.{path_key}"),
            size=int(splits_raw[size_key]),
            allows_b1_tuning=allows_tuning,
        )

    allowed = calibration_raw.get("allowed_parameters")
    if not isinstance(allowed, list) or not allowed:
        raise ProtocolValidationError("calibration.allowed_parameters must be non-empty")
    core_metrics = metrics_raw.get("core_metrics")
    if not isinstance(core_metrics, list):
        raise ProtocolValidationError("metrics.core_metrics must be a list")
    released_stages = success_raw.get("released_stages")
    declared_stages = success_raw.get("declared_stages")
    if not isinstance(released_stages, list) or not released_stages:
        raise ProtocolValidationError("success.released_stages must be non-empty")
    if not isinstance(declared_stages, list) or not declared_stages:
        raise ProtocolValidationError("success.declared_stages must be non-empty")

    result = ProtocolConfig(
        path=protocol_path,
        raw=raw,
        environment=environment,
        protocol_id=str(protocol_raw.get("protocol_id", "")),
        protocol_version=str(protocol_raw.get("protocol_version", "")),
        environment_version=str(protocol_raw.get("environment_version", "")),
        metrics_schema_version=str(protocol_raw.get("metrics_schema_version", "")),
        split_id=str(splits_raw.get("split_id", "")),
        calibration_policy_id=str(calibration_raw.get("policy_id", "")),
        split_manifest_path=_root_path(
            splits_raw.get("manifest_path"), "splits.manifest_path"
        ),
        splits=split_specs,
        core_metrics=tuple(str(value) for value in core_metrics),
        allowed_calibration_parameters=tuple(str(value) for value in allowed),
        baseline_frozen=bool(calibration_raw.get("baseline_frozen")),
        episode_timeout=environment.simulation.episode_timeout,
        released_stages=tuple(str(value) for value in released_stages),
        declared_stages=tuple(str(value) for value in declared_stages),
    )
    validate_protocol(result, validate_splits=validate_splits)
    return result


def validate_protocol(protocol: ProtocolConfig, *, validate_splits: bool = True) -> None:
    # Import lazily so evaluation.protocol remains usable while benchmark.runner
    # imports the protocol-aware metrics modules.
    from benchmark.seed_io import load_seeds

    if protocol.protocol_id != "evaluation_protocol":
        raise ProtocolValidationError("protocol_id must be 'evaluation_protocol'")
    for field_name, value in (
        ("protocol_version", protocol.protocol_version),
        ("metrics_schema_version", protocol.metrics_schema_version),
    ):
        if SEMVER_PATTERN.fullmatch(value) is None:
            raise ProtocolValidationError(f"{field_name} must be semantic version x.y.z")
    if not protocol.environment_version:
        raise ProtocolValidationError("environment_version must be non-empty")
    modes = (
        protocol.environment.pick.mode,
        protocol.environment.place.mode,
        protocol.environment.physics.mode,
    )
    if modes != ("random", "random", "random"):
        raise ProtocolValidationError(
            "Evaluation Protocol v1 requires random pick, place, and physics modes"
        )
    if protocol.environment.controller.type != "sensor_event_b1":
        raise ProtocolValidationError("Protocol task config must select sensor_event_b1")
    if protocol.environment.observation.source != "perception":
        raise ProtocolValidationError("Protocol task config must use perception observation")
    expected_sizes = {"calibration": 30, "development": 60, "held_out_test": 100}
    for name, expected in expected_sizes.items():
        if protocol.splits[name].size != expected:
            raise ProtocolValidationError(
                f"{name} split size must be {expected}, got {protocol.splits[name].size}"
            )
    if protocol.splits["development"].allows_b1_tuning:
        raise ProtocolValidationError("Development must forbid B1 parameter tuning")
    if protocol.splits["held_out_test"].allows_b1_tuning:
        raise ProtocolValidationError("Held-out Test must forbid B1 parameter tuning")
    if protocol.baseline_frozen:
        raise ProtocolValidationError("Evaluation Protocol tooling must start baseline_frozen=false")
    if not REQUIRED_CORE_METRICS.issubset(protocol.core_metrics):
        missing = sorted(REQUIRED_CORE_METRICS - set(protocol.core_metrics))
        raise ProtocolValidationError(f"Missing core metric definitions: {missing}")
    success_raw = protocol.raw["success"]
    if not math_isclose(
        success_raw.get("placement_xy_tolerance"),
        protocol.environment.b1.final_place_xy_tolerance,
    ) or not math_isclose(
        success_raw.get("placement_height_tolerance"),
        protocol.environment.b1.final_place_height_tolerance,
    ):
        raise ProtocolValidationError(
            "Protocol placement tolerances must match the independent B1 evaluator"
        )
    if len(protocol.allowed_calibration_parameters) != len(
        set(protocol.allowed_calibration_parameters)
    ):
        raise ProtocolValidationError("Calibration allowlist contains duplicates")
    forbidden_prefixes = (
        "workspace.",
        "pick.",
        "place.",
        "physics.",
        "camera.",
        "simulation.",
        "environment.",
    )
    for path in protocol.allowed_calibration_parameters:
        if path.startswith(forbidden_prefixes):
            raise ProtocolValidationError(
                f"Environment/protocol field cannot enter calibration allowlist: {path}"
            )
        config_value(protocol.environment, path)
    catalog = calibration_parameter_catalog(protocol)
    if len(catalog) != len(protocol.allowed_calibration_parameters):
        raise ProtocolValidationError("Calibration parameter catalog is incomplete")

    if not validate_splits:
        return
    all_seeds: dict[str, list[int]] = {}
    for name in ("calibration", "development", "held_out_test", "calibration_smoke"):
        spec = protocol.splits[name]
        seeds = load_seeds(spec.path)
        if len(seeds) != spec.size:
            raise ProtocolValidationError(
                f"{name} split contains {len(seeds)} seeds, expected {spec.size}"
            )
        all_seeds[name] = seeds
    for left_name, left in all_seeds.items():
        for right_name, right in all_seeds.items():
            if left_name >= right_name:
                continue
            overlap = sorted(set(left) & set(right))
            if overlap:
                raise ProtocolValidationError(
                    f"Splits {left_name} and {right_name} overlap: {overlap[:5]}"
                )
    if not protocol.split_manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest does not exist: {protocol.split_manifest_path}"
        )
    manifest = json.loads(protocol.split_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_config_sha256") != protocol.sha256:
        raise ProtocolValidationError("Split manifest protocol config SHA-256 mismatch")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(payload):
        raise ProtocolValidationError("Split manifest self hash mismatch")
    for name in ("calibration", "development", "held_out_test"):
        file_record = manifest.get("files", {}).get(name, {})
        if file_record.get("seed_count") != protocol.splits[name].size:
            raise ProtocolValidationError(f"Split manifest count mismatch for {name}")
        if file_record.get("sha256") != sha256_file(protocol.splits[name].path):
            raise ProtocolValidationError(f"Split manifest file hash mismatch for {name}")


def math_isclose(value: Any, expected: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return abs(float(value) - float(expected)) <= 1e-12


def validate_baseline_compatibility(
    protocol: ProtocolConfig,
    baseline: EnvConfig,
) -> None:
    protected = (
        "workspace",
        "pick",
        "place",
        "physics",
        "simulation",
        "camera",
    )
    mismatches = [
        name
        for name in protected
        if getattr(protocol.environment, name) != getattr(baseline, name)
    ]
    if mismatches:
        raise ProtocolValidationError(
            "Baseline changes protocol-protected environment fields: "
            + ", ".join(mismatches)
        )
    if baseline.controller.type != "sensor_event_b1":
        raise ProtocolValidationError("Calibration baseline must use sensor_event_b1")
    if baseline.observation.source != "perception":
        raise ProtocolValidationError("Calibration baseline must use perception")


__all__ = [
    "CalibrationParameter",
    "ProtocolConfig",
    "ProtocolValidationError",
    "SplitSpec",
    "calibration_parameter_catalog",
    "canonical_sha256",
    "config_value",
    "load_protocol",
    "sha256_file",
    "validate_baseline_compatibility",
    "validate_protocol",
]
