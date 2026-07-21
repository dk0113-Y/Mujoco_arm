from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import numpy as np

from environments import PandaUTableEnv
from environments.panda_u_table_env import InvalidResetError

from .protocol import PROJECT_ROOT, ProtocolConfig, canonical_sha256, sha256_file


REGIONS = ("front", "left", "right")
REGION_PAIRS = tuple((pick, place) for pick in REGIONS for place in REGIONS)
SPLIT_ORDER = ("calibration", "development", "held_out_test")
MASK_64 = (1 << 64) - 1


@dataclass(frozen=True)
class TaskSample:
    seed: int
    pick_region: str | None
    place_region: str | None
    pick_position: tuple[float, float, float] | None
    place_position: tuple[float, float, float] | None
    pick_place_distance: float | None
    mass: float | None
    friction: tuple[float, float, float] | None
    settled_object_table_penetration: float | None
    illegal_reasons: tuple[str, ...] = ()

    @property
    def region_pair(self) -> str | None:
        if self.pick_region is None or self.place_region is None:
            return None
        return f"{self.pick_region}->{self.place_region}"

    @property
    def same_region(self) -> bool | None:
        if self.pick_region is None or self.place_region is None:
            return None
        return self.pick_region == self.place_region

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["region_pair"] = self.region_pair
        value["same_region"] = self.same_region
        return value


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK_64
    return (value ^ (value >> 31)) & MASK_64


def _candidate_seeds(protocol: ProtocolConfig) -> list[int]:
    raw = protocol.raw["split_generation"]
    start = int(raw["candidate_start"])
    stop = int(raw["candidate_stop"])
    count = int(raw["candidate_count"])
    generator_seed = int(raw["generator_seed"])
    if start < 0 or stop <= start or not 0 < count <= stop - start:
        raise ValueError("Invalid deterministic split candidate range/count")
    ordered = sorted(
        range(start, stop),
        key=lambda seed: (_splitmix64(seed ^ generator_seed), seed),
    )
    return ordered[:count]


def _sample_reasons(env: PandaUTableEnv) -> tuple[str, ...]:
    episode = env.current_episode
    if episode is None:
        return ("missing_episode_metadata",)
    reasons: list[str] = []
    pick = np.asarray(episode.pick_position, dtype=float)
    place = np.asarray(episode.place_position, dtype=float)
    if not np.all(np.isfinite(pick)) or not np.all(np.isfinite(place)):
        reasons.append("non_finite_position")
    if not env.workspace.region(episode.pick_region).contains_xy(
        pick[:2], env.config.pick.edge_margin
    ):
        reasons.append("pick_outside_region")
    if not env.workspace.region(episode.place_region).contains_xy(
        place[:2], env.config.place.edge_margin
    ):
        reasons.append("place_outside_region")
    if not env.workspace.is_clear_of_base(pick[:2]):
        reasons.append("pick_inside_base_clearance")
    if not env.workspace.is_clear_of_base(place[:2]):
        reasons.append("place_inside_base_clearance")
    distance_xy = float(np.linalg.norm(pick[:2] - place[:2]))
    if distance_xy + 1e-12 < env.config.place.minimum_xy_distance:
        reasons.append("pick_place_distance_below_minimum")
    expected_pick_z = env.workspace.object_spawn_z(episode.pick_region)
    expected_place_z = env.workspace.target_z(episode.place_region)
    if not math.isclose(pick[2], expected_pick_z, rel_tol=0.0, abs_tol=1e-9):
        reasons.append("pick_z_mismatch")
    if not math.isclose(place[2], expected_place_z, rel_tol=0.0, abs_tol=1e-9):
        reasons.append("place_z_mismatch")
    target_radius = float(env.model.site_size[env.place_target_site_id, 0])
    if distance_xy < env.config.workspace.object_half_size + target_radius - 1e-12:
        reasons.append("object_target_overlap")
    actual_bottom = float(
        env.data.xpos[env.object_body_id, 2] - env.config.workspace.object_half_size
    )
    table_top = env.workspace.region(episode.pick_region).top_z
    # After the configured 1 s settling period MuJoCo maintains a small contact
    # solver overlap for a resting body.  Treat only overlap beyond the original
    # 1 mm spawn clearance as a geometric penetration defect.
    penetration_tolerance = max(1e-3, env.config.workspace.spawn_clearance)
    if actual_bottom < table_top - penetration_tolerance:
        reasons.append("initial_object_penetration")
    low_mass, high_mass = env.config.physics.mass_range
    if not math.isfinite(episode.mass) or not low_mass <= episode.mass <= high_mass:
        reasons.append("mass_out_of_range")
    for index, (value, low, high) in enumerate(
        zip(
            episode.friction,
            env.config.physics.friction_min,
            env.config.physics.friction_max,
        )
    ):
        if not math.isfinite(value) or not low <= value <= high or value <= 0.0:
            reasons.append(f"friction_{index}_out_of_range")
    return tuple(sorted(set(reasons)))


def collect_task_samples(
    protocol: ProtocolConfig,
    seeds: Iterable[int],
) -> list[TaskSample]:
    """Reset one renderer-free environment and record only sampled task metadata."""

    env = PandaUTableEnv(protocol.environment)
    samples: list[TaskSample] = []
    try:
        for seed in seeds:
            try:
                env.reset(seed=int(seed))
            except InvalidResetError as exc:
                samples.append(
                    TaskSample(
                        seed=int(seed),
                        pick_region=None,
                        place_region=None,
                        pick_position=None,
                        place_position=None,
                        pick_place_distance=None,
                        mass=None,
                        friction=None,
                        settled_object_table_penetration=None,
                        illegal_reasons=(f"invalid_reset:{exc}",),
                    )
                )
                continue
            episode = env.current_episode
            if episode is None:
                raise RuntimeError("Environment reset omitted episode metadata")
            pick = tuple(float(value) for value in episode.pick_position)
            place = tuple(float(value) for value in episode.place_position)
            samples.append(
                TaskSample(
                    seed=int(seed),
                    pick_region=episode.pick_region,
                    place_region=episode.place_region,
                    pick_position=pick,
                    place_position=place,
                    pick_place_distance=math.dist(pick, place),
                    mass=float(episode.mass),
                    friction=tuple(float(value) for value in episode.friction),
                    settled_object_table_penetration=max(
                        0.0,
                        env.workspace.region(episode.pick_region).top_z
                        - float(
                            env.data.xpos[env.object_body_id, 2]
                            - env.config.workspace.object_half_size
                        ),
                    ),
                    illegal_reasons=_sample_reasons(env),
                )
            )
    finally:
        env.close()
    return samples


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
    }


def distribution_summary(samples: Iterable[TaskSample]) -> dict[str, Any]:
    values = list(samples)
    legal = [sample for sample in values if not sample.illegal_reasons]
    pick_counts = Counter(sample.pick_region for sample in legal)
    place_counts = Counter(sample.place_region for sample in legal)
    pair_counts = Counter(sample.region_pair for sample in legal)
    same_counts = Counter(
        "same_region" if sample.same_region else "cross_region" for sample in legal
    )
    reason_counts = Counter(
        reason for sample in values for reason in sample.illegal_reasons
    )
    friction_summaries = []
    for index in range(3):
        friction_summaries.append(
            _numeric_summary(
                sample.friction[index]
                for sample in legal
                if sample.friction is not None
            )
        )
    return {
        "requested_seed_count": len(values),
        "legal_sample_count": len(legal),
        "illegal_sample_count": len(values) - len(legal),
        "illegal_reason_counts": dict(sorted(reason_counts.items())),
        "illegal_seeds": [sample.seed for sample in values if sample.illegal_reasons],
        "pick_region_counts": {name: pick_counts[name] for name in REGIONS},
        "place_region_counts": {name: place_counts[name] for name in REGIONS},
        "region_pair_counts": {
            f"{pick}->{place}": pair_counts[f"{pick}->{place}"]
            for pick, place in REGION_PAIRS
        },
        "same_cross_counts": {
            "same_region": same_counts["same_region"],
            "cross_region": same_counts["cross_region"],
        },
        "pick_place_distance": _numeric_summary(
            sample.pick_place_distance
            for sample in legal
            if sample.pick_place_distance is not None
        ),
        "mass": _numeric_summary(
            sample.mass for sample in legal if sample.mass is not None
        ),
        "friction": friction_summaries,
        "settled_object_table_penetration": _numeric_summary(
            sample.settled_object_table_penetration
            for sample in legal
            if sample.settled_object_table_penetration is not None
        ),
    }


def _bin(value: float, low: float, high: float, count: int = 4) -> int:
    if high <= low:
        return 0
    normalized = min(1.0, max(0.0, (value - low) / (high - low)))
    return min(count - 1, int(normalized * count))


def _features(
    sample: TaskSample,
    protocol: ProtocolConfig,
    distance_range: tuple[float, float],
) -> tuple[str, ...]:
    if sample.mass is None or sample.friction is None or sample.pick_place_distance is None:
        return ()
    physics = protocol.environment.physics
    result = [
        f"pick:{sample.pick_region}",
        f"place:{sample.place_region}",
        f"pair:{sample.region_pair}",
        f"same:{sample.same_region}",
        f"distance_bin:{_bin(sample.pick_place_distance, *distance_range)}",
        f"mass_bin:{_bin(sample.mass, *physics.mass_range)}",
    ]
    for index, value in enumerate(sample.friction):
        result.append(
            f"friction_{index}_bin:"
            f"{_bin(value, physics.friction_min[index], physics.friction_max[index])}"
        )
    return tuple(result)


def _select_split(
    candidates: list[TaskSample],
    *,
    size: int,
    used: set[int],
    protocol: ProtocolConfig,
    distance_range: tuple[float, float],
) -> list[TaskSample]:
    available = [
        sample
        for sample in candidates
        if sample.seed not in used and not sample.illegal_reasons
    ]
    order = {sample.seed: index for index, sample in enumerate(candidates)}
    chosen: list[TaskSample] = []
    seen_features: set[str] = set()
    pair_counts: Counter[str] = Counter()

    def choose(pool: list[TaskSample]) -> TaskSample:
        if not pool:
            raise RuntimeError("Candidate pool cannot satisfy protocol coverage")
        return max(
            pool,
            key=lambda sample: (
                sum(
                    feature not in seen_features
                    for feature in _features(sample, protocol, distance_range)
                ),
                -order[sample.seed],
            ),
        )

    def add(sample: TaskSample) -> None:
        chosen.append(sample)
        used.add(sample.seed)
        seen_features.update(_features(sample, protocol, distance_range))
        pair_counts[str(sample.region_pair)] += 1

    for pick, place in REGION_PAIRS:
        pair_name = f"{pick}->{place}"
        pool = [
            sample
            for sample in available
            if sample.seed not in used and sample.region_pair == pair_name
        ]
        add(choose(pool))

    while len(chosen) < size:
        minimum_count = min(pair_counts[f"{pick}->{place}"] for pick, place in REGION_PAIRS)
        pool = [
            sample
            for sample in available
            if sample.seed not in used
            and pair_counts[str(sample.region_pair)] == minimum_count
        ]
        add(choose(pool))
    return chosen


def seed_file_text(seeds: Iterable[int]) -> str:
    return "".join(f"{int(seed)}\n" for seed in seeds)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def generate_split_plan(
    protocol: ProtocolConfig,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    candidate_seeds = _candidate_seeds(protocol)
    candidate_samples = collect_task_samples(protocol, candidate_seeds)
    legal_distances = [
        sample.pick_place_distance
        for sample in candidate_samples
        if not sample.illegal_reasons and sample.pick_place_distance is not None
    ]
    if not legal_distances:
        raise RuntimeError("No legal candidate tasks were sampled")
    distance_range = (min(legal_distances), max(legal_distances))
    used: set[int] = set()
    split_samples: dict[str, list[TaskSample]] = {}
    for name in SPLIT_ORDER:
        split_samples[name] = _select_split(
            candidate_samples,
            size=protocol.splits[name].size,
            used=used,
            protocol=protocol,
            distance_range=distance_range,
        )
    seeds_by_split = {
        name: [sample.seed for sample in split_samples[name]] for name in SPLIT_ORDER
    }
    generation = protocol.raw["split_generation"]
    files: dict[str, Any] = {}
    for name in SPLIT_ORDER:
        spec = protocol.splits[name]
        content = seed_file_text(seeds_by_split[name]).encode("utf-8")
        import hashlib

        files[name] = {
            "path": _relative(spec.path),
            "seed_count": len(seeds_by_split[name]),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    payload: dict[str, Any] = {
        "split_id": protocol.split_id,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_config_path": _relative(protocol.path),
        "protocol_config_sha256": protocol.sha256,
        "generator_version": str(generation["generator_version"]),
        "generator_seed": int(generation["generator_seed"]),
        "generation_date": str(generation["generation_date"]),
        "candidate_start": int(generation["candidate_start"]),
        "candidate_stop": int(generation["candidate_stop"]),
        "candidate_count": int(generation["candidate_count"]),
        "selection_algorithm": (
            "SplitMix64 candidate ordering; environment-reset metadata only; "
            "nine region-pair coverage followed by balanced pair counts and "
            "greedy mass/friction/distance-bin coverage"
        ),
        "controller_outcomes_used": False,
        "files": files,
        "coverage": {
            name: distribution_summary(split_samples[name]) for name in SPLIT_ORDER
        },
        "candidate_illegal_sample_count": sum(
            bool(sample.illegal_reasons) for sample in candidate_samples
        ),
    }
    manifest = {**payload, "manifest_sha256": canonical_sha256(payload)}
    return seeds_by_split, manifest


def validate_manifest_files(protocol: ProtocolConfig, manifest: Mapping[str, Any]) -> None:
    for name in SPLIT_ORDER:
        expected = manifest["files"][name]
        path = protocol.splits[name].path
        if not path.is_file():
            raise FileNotFoundError(f"Split file does not exist: {path}")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"{name} split SHA-256 differs from deterministic plan")
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != canonical_sha256(payload):
        raise ValueError("Generated split manifest hash is inconsistent")


__all__ = [
    "REGIONS",
    "REGION_PAIRS",
    "SPLIT_ORDER",
    "TaskSample",
    "collect_task_samples",
    "distribution_summary",
    "generate_split_plan",
    "seed_file_text",
    "validate_manifest_files",
]
