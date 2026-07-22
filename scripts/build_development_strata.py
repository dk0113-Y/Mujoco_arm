from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.manifest import repository_metadata, sha256_file
from benchmark.schemas import write_json
from benchmark.seed_io import load_seeds
from evaluation.protocol import load_protocol
from evaluation.split_analysis import TaskSample, collect_task_samples


PROTOCOL_PATH = PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml"
DEVELOPMENT_SEEDS_PATH = (
    PROJECT_ROOT / "configs/splits/evaluation_protocol_v1/development_v1.txt"
)
EXPECTED_PROTOCOL_SHA256 = (
    "7a47be9ddf3851b06c84068ec29030d5bf25ebf60f37057d55371823b07e10bd"
)
EXPECTED_DEVELOPMENT_SPLIT_SHA256 = (
    "677ecd23f9e689b971fa7340f7d34d674f07dfca19bfa9cd4634598d497b98d6"
)


class DevelopmentStrataError(ValueError):
    """Raised when registered Development metadata cannot be stratified."""


def _exact_file(path: str | Path, expected: Path, label: str) -> Path:
    actual = Path(path).expanduser().resolve()
    if actual != expected.resolve() or not actual.is_file():
        raise DevelopmentStrataError(
            f"{label} must be the registered file {expected.resolve()}"
        )
    return actual


def _edges(low: float, high: float) -> list[float]:
    width = (high - low) / 4.0
    return [low + index * width for index in range(5)]


def _bin_record(value: float, edges: Sequence[float]) -> dict[str, Any]:
    if len(edges) != 5 or any(not math.isfinite(float(item)) for item in edges):
        raise DevelopmentStrataError("A four-bin edge sequence must contain five values")
    if any(float(edges[index]) > float(edges[index + 1]) for index in range(4)):
        raise DevelopmentStrataError("Bin edges must be nondecreasing")
    if value < edges[0] - 1e-12 or value > edges[-1] + 1e-12:
        raise DevelopmentStrataError(
            f"Value {value} is outside declared bin range [{edges[0]}, {edges[-1]}]"
        )
    index = 0
    while index < 3 and value > edges[index + 1]:
        index += 1
    return {
        "index": index,
        "name": f"Q{index + 1}",
        "lower": float(edges[index]),
        "upper": float(edges[index + 1]),
        "upper_inclusive": True,
    }


def _require_complete_samples(
    samples: Iterable[TaskSample], expected_seeds: Sequence[int]
) -> list[TaskSample]:
    values = list(samples)
    if [sample.seed for sample in values] != list(expected_seeds):
        raise DevelopmentStrataError(
            "Environment-reset samples do not preserve Development seed order"
        )
    if any(sample.illegal_reasons for sample in values):
        invalid = {
            sample.seed: list(sample.illegal_reasons)
            for sample in values
            if sample.illegal_reasons
        }
        raise DevelopmentStrataError(f"Development reset metadata is invalid: {invalid}")
    for sample in values:
        if (
            sample.pick_region is None
            or sample.place_region is None
            or sample.pick_position is None
            or sample.place_position is None
            or sample.pick_place_distance is None
            or sample.mass is None
            or sample.friction is None
        ):
            raise DevelopmentStrataError(
                f"Development seed {sample.seed} has incomplete reset metadata"
            )
    return values


def build_development_strata(
    *,
    protocol_path: str | Path,
    seeds_file: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    protocol_file = _exact_file(protocol_path, PROTOCOL_PATH, "protocol")
    seeds_path = _exact_file(
        seeds_file, DEVELOPMENT_SEEDS_PATH, "Development seed file"
    )
    if sha256_file(protocol_file) != EXPECTED_PROTOCOL_SHA256:
        raise DevelopmentStrataError("Evaluation Protocol SHA-256 mismatch")
    if sha256_file(seeds_path) != EXPECTED_DEVELOPMENT_SPLIT_SHA256:
        raise DevelopmentStrataError("Development split SHA-256 mismatch")
    protocol = load_protocol(protocol_file, validate_splits=False)
    seeds = load_seeds(seeds_path)
    if len(seeds) != 60 or len(set(seeds)) != 60:
        raise DevelopmentStrataError(
            "Development split must contain exactly 60 unique seeds"
        )
    samples = _require_complete_samples(
        collect_task_samples(protocol, seeds), seeds
    )

    physics = protocol.environment.physics
    mass_edges = _edges(*physics.mass_range)
    friction_edges = [
        _edges(physics.friction_min[index], physics.friction_max[index])
        for index in range(3)
    ]
    distances = sorted(float(sample.pick_place_distance) for sample in samples)
    distance_cuts = statistics.quantiles(distances, n=4, method="inclusive")
    distance_edges = [distances[0], *distance_cuts, distances[-1]]

    records: list[dict[str, Any]] = []
    for sample in samples:
        assert sample.pick_position is not None
        assert sample.place_position is not None
        assert sample.mass is not None
        assert sample.friction is not None
        assert sample.pick_place_distance is not None
        records.append(
            {
                "seed": sample.seed,
                "pick_region": sample.pick_region,
                "place_region": sample.place_region,
                "region_pair": sample.region_pair,
                "same_cross": (
                    "same_region" if sample.same_region else "cross_region"
                ),
                "pick_position": list(sample.pick_position),
                "place_position": list(sample.place_position),
                "mass": sample.mass,
                "friction": list(sample.friction),
                "pick_place_distance": sample.pick_place_distance,
                "mass_bin": _bin_record(sample.mass, mass_edges),
                "sliding_friction_bin": _bin_record(
                    sample.friction[0], friction_edges[0]
                ),
                "torsional_friction_bin": _bin_record(
                    sample.friction[1], friction_edges[1]
                ),
                "rolling_friction_bin": _bin_record(
                    sample.friction[2], friction_edges[2]
                ),
                "pick_place_distance_bin": _bin_record(
                    sample.pick_place_distance, distance_edges
                ),
            }
        )

    repository = repository_metadata(PROJECT_ROOT)
    result = {
        "strata_schema_version": "1.0.0",
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "split_id": protocol.split_id,
        "split_name": "development",
        "seed_count": len(seeds),
        "input_files": {
            "protocol_path": str(protocol_file),
            "protocol_sha256": sha256_file(protocol_file),
            "seeds_file": str(seeds_path),
            "seeds_file_sha256": sha256_file(seeds_path),
        },
        "binning": {
            "mass": {
                "rule": "four equal-width bins over protocol range",
                "edges": mass_edges,
            },
            "sliding_friction": {
                "rule": "four equal-width bins over protocol range",
                "edges": friction_edges[0],
            },
            "torsional_friction": {
                "rule": "four equal-width bins over protocol range",
                "edges": friction_edges[1],
            },
            "rolling_friction": {
                "rule": "four equal-width bins over protocol range",
                "edges": friction_edges[2],
            },
            "pick_place_distance": {
                "rule": "Development metadata inclusive quartiles",
                "edges": distance_edges,
                "cut_points": distance_cuts,
            },
        },
        "generation_code_commit": repository["git_commit"],
        "generation_git_dirty": repository["git_dirty"],
        "controller_outcomes_used": False,
        "calibration_outcomes_used": False,
        "development_outcomes_used": False,
        "held_out_data_read": False,
        "renderer_created": False,
        "ik_executed": False,
        "strata": records,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predeclare Development strata from registered seeds and renderer-free "
            "environment-reset metadata. No episode outcomes are read."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seeds-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_development_strata(
            protocol_path=args.protocol,
            seeds_file=args.seeds_file,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"Development strata error: {exc}", file=sys.stderr)
        return 1
    print(
        "Development strata built: "
        f"seeds={result['seed_count']}, "
        "controller_outcomes_used=false, renderer_created=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
