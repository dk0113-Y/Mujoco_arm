from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


DEFAULT_FINGERPRINT_ATOL = 1e-12

OUTCOME_BOTH_SUCCESS = "both_success"
OUTCOME_ORACLE_ONLY_SUCCESS = "oracle_only_success"
OUTCOME_VISION_ONLY_SUCCESS = "vision_only_success"
OUTCOME_BOTH_FAILED = "both_failed"
OUTCOME_INVALID_PAIR = "invalid_pair"
OUTCOME_PROGRAM_ERROR = "program_error"


class FingerprintError(ValueError):
    """Raised when episode parameters cannot form a valid fingerprint."""


class PairMismatchError(ValueError):
    """Raised when two supposedly paired episodes have different parameters."""

    def __init__(
        self,
        oracle: "EpisodeFingerprint",
        vision: "EpisodeFingerprint",
        differences: Mapping[str, tuple[Any, Any]],
        *,
        atol: float,
    ) -> None:
        self.oracle = oracle
        self.vision = vision
        self.differences = dict(differences)
        self.atol = atol
        details = "; ".join(
            f"{name}: oracle={values[0]!r}, vision={values[1]!r}"
            for name, values in self.differences.items()
        )
        super().__init__(
            "Paired episode fingerprints do not match "
            f"(absolute tolerance={atol:g}): {details}"
        )


def _mapping_from(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    raise FingerprintError(f"{label} must be a mapping or expose episode fields")


def _required(mapping: Mapping[str, Any], name: str, *, fallback: str | None = None) -> Any:
    if name in mapping:
        return mapping[name]
    if fallback is not None and fallback in mapping:
        return mapping[fallback]
    raise FingerprintError(f"Missing fingerprint field: {name}")


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FingerprintError(f"seed must be an integer, got {value!r}")
    if value < 0:
        raise FingerprintError(f"seed must be non-negative, got {value}")
    return value


def _region(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FingerprintError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise FingerprintError(f"{name} must be a finite number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FingerprintError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(result):
        raise FingerprintError(f"{name} must be finite, got {value!r}")
    return result


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FingerprintError(f"{name} must contain exactly three numbers")
    if len(value) != 3:
        raise FingerprintError(f"{name} must contain exactly three numbers")
    converted = tuple(
        _finite_float(component, f"{name}[{index}]")
        for index, component in enumerate(value)
    )
    return converted  # type: ignore[return-value]


@dataclass(frozen=True)
class EpisodeFingerprint:
    """Canonical description of the randomized state shared by a method pair."""

    seed: int
    pick_position: tuple[float, float, float]
    place_position: tuple[float, float, float]
    pick_region: str
    place_region: str
    mass: float
    friction: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _seed(self.seed))
        object.__setattr__(
            self, "pick_position", _vector(self.pick_position, "pick_position")
        )
        object.__setattr__(
            self, "place_position", _vector(self.place_position, "place_position")
        )
        object.__setattr__(self, "pick_region", _region(self.pick_region, "pick_region"))
        object.__setattr__(
            self, "place_region", _region(self.place_region, "place_region")
        )
        object.__setattr__(self, "mass", _finite_float(self.mass, "mass"))
        object.__setattr__(self, "friction", _vector(self.friction, "friction"))

    @classmethod
    def from_episode_result(cls, result: Any) -> "EpisodeFingerprint":
        """Build a fingerprint from an ``EpisodeResult`` or equivalent mapping."""

        values = _mapping_from(result, label="episode result")
        return cls(
            seed=_seed(_required(values, "seed")),
            pick_position=_vector(
                _required(values, "sampled_pick_position", fallback="pick_position"),
                "sampled_pick_position",
            ),
            place_position=_vector(
                _required(values, "sampled_place_position", fallback="place_position"),
                "sampled_place_position",
            ),
            pick_region=_region(_required(values, "pick_region"), "pick_region"),
            place_region=_region(_required(values, "place_region"), "place_region"),
            mass=_finite_float(
                _required(values, "sampled_mass", fallback="mass"), "sampled_mass"
            ),
            friction=_vector(
                _required(values, "sampled_friction", fallback="friction"),
                "sampled_friction",
            ),
        )

    @classmethod
    def from_episode_parameters(cls, parameters: Any) -> "EpisodeFingerprint":
        """Build a fingerprint from ``EpisodeParameters`` or its ``as_dict`` output."""

        values = _mapping_from(parameters, label="episode parameters")
        return cls(
            seed=_seed(_required(values, "seed")),
            pick_position=_vector(_required(values, "pick_position"), "pick_position"),
            place_position=_vector(
                _required(values, "place_position"), "place_position"
            ),
            pick_region=_region(_required(values, "pick_region"), "pick_region"),
            place_region=_region(_required(values, "place_region"), "place_region"),
            mass=_finite_float(_required(values, "mass"), "mass"),
            friction=_vector(_required(values, "friction"), "friction"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "pick_position": list(self.pick_position),
            "place_position": list(self.place_position),
            "pick_region": self.pick_region,
            "place_region": self.place_region,
            "mass": self.mass,
            "friction": list(self.friction),
        }

    def canonical_json(self) -> str:
        """Return stable, whitespace-free JSON used as the hash input."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of :meth:`canonical_json`."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def _validate_atol(atol: float) -> float:
    tolerance = _finite_float(atol, "atol")
    if tolerance < 0.0:
        raise ValueError("atol must be non-negative")
    return tolerance


def fingerprint_differences(
    oracle: EpisodeFingerprint,
    vision: EpisodeFingerprint,
    *,
    atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> dict[str, tuple[Any, Any]]:
    """Return every exact or numeric field mismatch between a method pair."""

    tolerance = _validate_atol(atol)
    differences: dict[str, tuple[Any, Any]] = {}
    for name in ("seed", "pick_region", "place_region"):
        left, right = getattr(oracle, name), getattr(vision, name)
        if left != right:
            differences[name] = (left, right)
    for name in ("pick_position", "place_position", "friction"):
        left, right = getattr(oracle, name), getattr(vision, name)
        if any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
            for a, b in zip(left, right)
        ):
            differences[name] = (left, right)
    if not math.isclose(oracle.mass, vision.mass, rel_tol=0.0, abs_tol=tolerance):
        differences["mass"] = (oracle.mass, vision.mass)
    return differences


def fingerprints_match(
    oracle: EpisodeFingerprint,
    vision: EpisodeFingerprint,
    *,
    atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> bool:
    """Compare identifiers exactly and randomized floating values by absolute tolerance."""

    return not fingerprint_differences(oracle, vision, atol=atol)


def validate_pair(
    oracle: EpisodeFingerprint,
    vision: EpisodeFingerprint,
    *,
    atol: float = DEFAULT_FINGERPRINT_ATOL,
) -> None:
    """Raise :class:`PairMismatchError` unless fingerprints describe one episode."""

    tolerance = _validate_atol(atol)
    differences = fingerprint_differences(oracle, vision, atol=tolerance)
    if differences:
        raise PairMismatchError(oracle, vision, differences, atol=tolerance)


def classify_outcome(
    oracle_ground_truth_success: bool | None,
    vision_ground_truth_success: bool | None,
    *,
    pair_valid: bool = True,
    program_error: bool = False,
) -> str:
    """Classify a pair without assuming that the Oracle method succeeds."""

    if program_error:
        return OUTCOME_PROGRAM_ERROR
    if not pair_valid:
        return OUTCOME_INVALID_PAIR
    if not isinstance(oracle_ground_truth_success, bool) or not isinstance(
        vision_ground_truth_success, bool
    ):
        raise ValueError(
            "Valid completed pairs require boolean ground-truth outcomes for both methods"
        )
    if oracle_ground_truth_success and vision_ground_truth_success:
        return OUTCOME_BOTH_SUCCESS
    if oracle_ground_truth_success:
        return OUTCOME_ORACLE_ONLY_SUCCESS
    if vision_ground_truth_success:
        return OUTCOME_VISION_ONLY_SUCCESS
    return OUTCOME_BOTH_FAILED


__all__ = [
    "DEFAULT_FINGERPRINT_ATOL",
    "EpisodeFingerprint",
    "FingerprintError",
    "OUTCOME_BOTH_FAILED",
    "OUTCOME_BOTH_SUCCESS",
    "OUTCOME_INVALID_PAIR",
    "OUTCOME_ORACLE_ONLY_SUCCESS",
    "OUTCOME_PROGRAM_ERROR",
    "OUTCOME_VISION_ONLY_SUCCESS",
    "PairMismatchError",
    "classify_outcome",
    "fingerprint_differences",
    "fingerprints_match",
    "validate_pair",
]
