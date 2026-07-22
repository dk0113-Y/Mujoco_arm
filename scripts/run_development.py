from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import FORMAL_METHOD_IDS, run_benchmark
from benchmark.manifest import sha256_file
from benchmark.seed_io import load_seeds
from environments import load_config
from evaluation.protocol import load_protocol, validate_baseline_compatibility


PROTOCOL_PATH = PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml"
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_v1.toml"
DEVELOPMENT_SEEDS_PATH = (
    PROJECT_ROOT / "configs/splits/evaluation_protocol_v1/development_v1.txt"
)
FREEZE_MANIFEST_PATH = (
    PROJECT_ROOT / "configs/baselines/b1_vision_v1_manifest.json"
)
DEVELOPMENT_OUTPUT_PATH = (
    PROJECT_ROOT / "outputs/development/b1_vision_v1/development_60"
)
EXPECTED_PROTOCOL_SHA256 = (
    "7a47be9ddf3851b06c84068ec29030d5bf25ebf60f37057d55371823b07e10bd"
)
EXPECTED_FROZEN_CONFIG_SHA256 = (
    "6808c142ae8805695fc43d5e4743a9529cdbea15008810456184e40e1c4b7ea9"
)
EXPECTED_DEVELOPMENT_SPLIT_SHA256 = (
    "677ecd23f9e689b971fa7340f7d34d674f07dfca19bfa9cd4634598d497b98d6"
)
EXPECTED_VERIFIED_BEHAVIOR_COMMIT = (
    "bf6a07945f396f7b98f5c24cf94d1a97b8dc7f9d"
)
EXPECTED_FREEZE_PACKAGE_COMMIT = (
    "129036f8eacc4d24aa892d7510c51dec33407c47"
)
ALLOWED_FREEZE_STATES = frozenset(
    {"committed_pending_tag", "tagged", "frozen"}
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class DevelopmentRunValidationError(ValueError):
    """Raised before any Development output or controller execution occurs."""


def _strict_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                DevelopmentRunValidationError(
                    f"{path.name} contains a non-finite value: {token}"
                )
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentRunValidationError(
            f"Cannot parse freeze manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise DevelopmentRunValidationError("Freeze manifest must be a JSON object")
    return value


def _require_exact_path(actual: str | Path, expected: Path, label: str) -> Path:
    path = Path(actual).expanduser().resolve()
    if path != expected.resolve():
        raise DevelopmentRunValidationError(
            f"{label} must be the registered file {expected.resolve()}, got {path}"
        )
    if not path.is_file():
        raise DevelopmentRunValidationError(f"{label} does not exist: {path}")
    return path


def _manifest_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DevelopmentRunValidationError(f"Freeze manifest {label} is missing")
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_PATTERN.fullmatch(value) is None:
        raise DevelopmentRunValidationError(
            f"Freeze manifest {label} must be a full lowercase Git commit"
        )
    return value


def validate_development_request(
    *,
    protocol_path: str | Path,
    frozen_config_path: str | Path,
    seeds_file: str | Path,
    freeze_manifest_path: str | Path,
    method_ids: Sequence[str],
) -> tuple[Any, dict[str, Any]]:
    protocol_file = _require_exact_path(protocol_path, PROTOCOL_PATH, "protocol")
    config_file = _require_exact_path(
        frozen_config_path, FROZEN_CONFIG_PATH, "frozen config"
    )
    seeds_path = _require_exact_path(
        seeds_file, DEVELOPMENT_SEEDS_PATH, "Development seed file"
    )
    manifest_file = _require_exact_path(
        freeze_manifest_path, FREEZE_MANIFEST_PATH, "freeze manifest"
    )
    if tuple(method_ids) != tuple(FORMAL_METHOD_IDS):
        raise DevelopmentRunValidationError(
            "Development methods must be exactly b0_oracle then b1_vision"
        )

    if sha256_file(protocol_file) != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentRunValidationError("Evaluation Protocol SHA-256 mismatch")
    config_hash = sha256_file(config_file)
    if config_hash != EXPECTED_FROZEN_CONFIG_SHA256:
        raise DevelopmentRunValidationError("Frozen config SHA-256 mismatch")
    split_hash = sha256_file(seeds_path)
    if split_hash != EXPECTED_DEVELOPMENT_SPLIT_SHA256:
        raise DevelopmentRunValidationError("Development split SHA-256 mismatch")
    seeds = load_seeds(seeds_path)
    if len(seeds) != 60 or len(set(seeds)) != 60:
        raise DevelopmentRunValidationError(
            "Development split must contain exactly 60 unique seeds"
        )

    freeze = _strict_json(manifest_file)
    if freeze.get("baseline_id") != "b1_vision_v1":
        raise DevelopmentRunValidationError("Freeze manifest baseline_id mismatch")
    if freeze.get("behavior_frozen") is not True:
        raise DevelopmentRunValidationError("Freeze manifest does not freeze behavior")
    if freeze.get("freeze_state") not in ALLOWED_FREEZE_STATES:
        raise DevelopmentRunValidationError(
            "Freeze manifest state is not committed_pending_tag or a later formal state"
        )
    if _manifest_path(
        freeze.get("frozen_config_path"), "frozen_config_path"
    ) != config_file:
        raise DevelopmentRunValidationError("Freeze manifest frozen config path mismatch")
    if freeze.get("frozen_config_sha256") != config_hash:
        raise DevelopmentRunValidationError("Freeze manifest frozen config hash mismatch")
    if freeze.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentRunValidationError("Freeze manifest protocol hash mismatch")
    protected = freeze.get("protected_input_hashes")
    if not isinstance(protected, Mapping) or protected.get(
        "development_split_sha256"
    ) != split_hash:
        raise DevelopmentRunValidationError(
            "Freeze manifest Development split hash mismatch"
        )
    verified_commit = _commit(
        freeze.get("verified_behavior_commit"), "verified_behavior_commit"
    )
    package_commit = _commit(
        freeze.get("freeze_package_commit") or freeze.get("final_git_commit"),
        "freeze_package_commit",
    )
    final_commit = _commit(freeze.get("final_git_commit"), "final_git_commit")
    if verified_commit != EXPECTED_VERIFIED_BEHAVIOR_COMMIT:
        raise DevelopmentRunValidationError("Verified behavior commit mismatch")
    if package_commit != EXPECTED_FREEZE_PACKAGE_COMMIT or final_commit != package_commit:
        raise DevelopmentRunValidationError("Freeze package commit mismatch")

    protocol = load_protocol(protocol_file, validate_splits=False)
    if protocol.splits["development"].path.resolve() != seeds_path:
        raise DevelopmentRunValidationError("Protocol Development path mismatch")
    if protocol.splits["development"].size != 60:
        raise DevelopmentRunValidationError("Protocol Development size mismatch")
    baseline = load_config(config_file)
    validate_baseline_compatibility(protocol, baseline)
    if baseline.simulation.viewer:
        raise DevelopmentRunValidationError("Frozen config enables visualization")

    metadata = {
        "frozen_baseline_id": "b1_vision_v1",
        "frozen_config_path": str(config_file),
        "frozen_config_sha256": config_hash,
        "freeze_manifest_path": str(manifest_file),
        "freeze_manifest_sha256": sha256_file(manifest_file),
        "verified_behavior_commit": verified_commit,
        "freeze_package_commit": package_commit,
    }
    return protocol, metadata


def validate_output_directory(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output != DEVELOPMENT_OUTPUT_PATH.resolve():
        raise DevelopmentRunValidationError(
            f"Development output must be {DEVELOPMENT_OUTPUT_PATH.resolve()}"
        )
    if output.exists() and not output.is_dir():
        raise DevelopmentRunValidationError(
            f"Development output exists but is not a directory: {output}"
        )
    if output.exists() and any(output.iterdir()):
        raise DevelopmentRunValidationError(
            f"Development output directory must be empty: {output}"
        )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the registered Development 60 split with B0-Oracle followed by "
            "the frozen B1-Vision v1 baseline. No overrides or diagnostics exist."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(FORMAL_METHOD_IDS),
        help="Must remain exactly: b0_oracle b1_vision",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol, frozen_metadata = validate_development_request(
            protocol_path=args.protocol,
            frozen_config_path=args.frozen_config,
            seeds_file=args.seeds_file,
            freeze_manifest_path=args.freeze_manifest,
            method_ids=args.methods,
        )
        output = validate_output_directory(args.output_dir)
        command_arguments = list(sys.argv[1:] if argv is None else argv)
        result = run_benchmark(
            config_path=args.frozen_config,
            method_ids=args.methods,
            seeds_file=args.seeds_file,
            output_dir=output,
            overwrite=False,
            continue_on_error=False,
            require_clean_git=args.require_clean_git,
            command=[sys.executable, str(Path(__file__).resolve()), *command_arguments],
            protocol=protocol,
            split_name="development",
            calibration_run=False,
            baseline_frozen=True,
            development_run=True,
            frozen_baseline_metadata=frozen_metadata,
        )
    except Exception as exc:
        print(f"Development run error: {exc}", file=sys.stderr)
        return 1
    print(
        "Development 60 finished: "
        f"completed_pairs={result.completed_pairs}/{result.requested_pairs}, "
        f"invalid_pairs={result.invalid_pairs}, "
        f"program_errors={result.program_errors}, "
        "development_run=true, baseline_frozen=true, "
        f"output={result.output_dir}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
