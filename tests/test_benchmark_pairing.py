from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unittest

from benchmark.pairing import (
    EpisodeFingerprint,
    FingerprintError,
    PairMismatchError,
    classify_outcome,
    fingerprint_differences,
    fingerprints_match,
    validate_pair,
)
from environments.randomization import EpisodeParameters


def result_mapping(*, seed: int = 42) -> dict[str, object]:
    return {
        "seed": seed,
        "sampled_pick_position": (0.50, 0.12, 0.246),
        "sampled_place_position": (0.50, -0.20, 0.222),
        "pick_region": "front",
        "place_region": "right",
        "sampled_mass": 0.10,
        "sampled_friction": (1.0, 0.005, 0.0001),
    }


def fingerprint(*, seed: int = 42) -> EpisodeFingerprint:
    return EpisodeFingerprint.from_episode_result(result_mapping(seed=seed))


class BenchmarkPairingTests(unittest.TestCase):
    def test_result_and_episode_parameters_build_the_same_fingerprint(self) -> None:
        result_fingerprint = fingerprint()
        parameters = EpisodeParameters(
            seed=42,
            pick_region="front",
            place_region="right",
            pick_position=(0.50, 0.12, 0.246),
            place_position=(0.50, -0.20, 0.222),
            mass=0.10,
            friction=(1.0, 0.005, 0.0001),
        )
        parameter_fingerprint = EpisodeFingerprint.from_episode_parameters(parameters)

        self.assertEqual(result_fingerprint, parameter_fingerprint)
        self.assertEqual(result_fingerprint.seed, 42)
        self.assertEqual(result_fingerprint.pick_region, "front")
        self.assertEqual(result_fingerprint.place_region, "right")
        self.assertTrue(fingerprints_match(result_fingerprint, parameter_fingerprint))
        self.assertIsNone(validate_pair(result_fingerprint, parameter_fingerprint))

    def test_canonical_json_and_digest_are_stable_and_auditable(self) -> None:
        first = EpisodeFingerprint.from_episode_result(result_mapping())
        reordered = dict(reversed(tuple(result_mapping().items())))
        second = EpisodeFingerprint.from_episode_result(reordered)

        self.assertEqual(first.canonical_json(), second.canonical_json())
        decoded = json.loads(first.canonical_json())
        self.assertEqual(decoded, first.to_dict())
        self.assertNotIn(" ", first.canonical_json())
        self.assertEqual(
            first.digest,
            hashlib.sha256(first.canonical_json().encode("utf-8")).hexdigest(),
        )
        self.assertEqual(len(first.digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in first.digest))

    def test_same_seed_is_repeatable_and_different_seed_changes_fingerprint(self) -> None:
        first = fingerprint(seed=42)
        repeated = fingerprint(seed=42)
        different = fingerprint(seed=43)
        self.assertEqual(first, repeated)
        self.assertEqual(first.digest, repeated.digest)
        self.assertNotEqual(first, different)
        self.assertNotEqual(first.digest, different.digest)
        self.assertEqual(fingerprint_differences(first, different), {"seed": (42, 43)})

    def test_float_comparison_uses_explicit_absolute_tolerance(self) -> None:
        oracle = fingerprint()
        within_tolerance = replace(
            oracle,
            pick_position=(oracle.pick_position[0] + 0.5e-12, *oracle.pick_position[1:]),
            mass=oracle.mass + 0.5e-12,
        )
        outside_tolerance = replace(
            oracle,
            pick_position=(oracle.pick_position[0] + 2.0e-12, *oracle.pick_position[1:]),
            mass=oracle.mass + 2.0e-12,
        )

        self.assertTrue(fingerprints_match(oracle, within_tolerance, atol=1e-12))
        self.assertEqual(fingerprint_differences(oracle, within_tolerance), {})
        differences = fingerprint_differences(oracle, outside_tolerance, atol=1e-12)
        self.assertEqual(set(differences), {"pick_position", "mass"})
        self.assertFalse(fingerprints_match(oracle, outside_tolerance, atol=1e-12))

    def test_pair_mismatch_reports_every_differing_field_and_rejects_pair(self) -> None:
        oracle = fingerprint()
        vision = EpisodeFingerprint(
            seed=43,
            pick_position=(0.51, 0.12, 0.246),
            place_position=(0.50, -0.21, 0.222),
            pick_region="left",
            place_region="front",
            mass=0.11,
            friction=(0.9, 0.006, 0.0002),
        )
        expected_fields = {
            "seed",
            "pick_position",
            "place_position",
            "pick_region",
            "place_region",
            "mass",
            "friction",
        }

        differences = fingerprint_differences(oracle, vision)
        self.assertEqual(set(differences), expected_fields)
        with self.assertRaises(PairMismatchError) as caught:
            validate_pair(oracle, vision)
        self.assertEqual(set(caught.exception.differences), expected_fields)
        self.assertIs(caught.exception.oracle, oracle)
        self.assertIs(caught.exception.vision, vision)
        self.assertIn("fingerprints do not match", str(caught.exception))
        self.assertIn("seed: oracle=42, vision=43", str(caught.exception))

    def test_invalid_tolerance_is_rejected(self) -> None:
        same = fingerprint()
        for invalid in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, FingerprintError)):
                    fingerprints_match(same, same, atol=invalid)

    def test_invalid_fingerprint_inputs_fail_fast(self) -> None:
        invalid_cases = {
            "boolean seed": {**result_mapping(), "seed": True},
            "negative seed": {**result_mapping(), "seed": -1},
            "missing seed": {
                key: value for key, value in result_mapping().items() if key != "seed"
            },
            "short pick vector": {
                **result_mapping(),
                "sampled_pick_position": (0.5, 0.1),
            },
            "non-finite mass": {**result_mapping(), "sampled_mass": math.nan},
            "non-finite friction": {
                **result_mapping(),
                "sampled_friction": (1.0, math.inf, 0.001),
            },
            "empty region": {**result_mapping(), "pick_region": ""},
        }
        for label, values in invalid_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(FingerprintError):
                    EpisodeFingerprint.from_episode_result(values)

    def test_all_outcome_categories_and_precedence(self) -> None:
        cases = (
            (True, True, "both_success"),
            (True, False, "oracle_only_success"),
            (False, True, "vision_only_success"),
            (False, False, "both_failed"),
        )
        for oracle_success, vision_success, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    classify_outcome(oracle_success, vision_success),
                    expected,
                )

        self.assertEqual(
            classify_outcome(None, None, pair_valid=False),
            "invalid_pair",
        )
        self.assertEqual(
            classify_outcome(None, None, pair_valid=False, program_error=True),
            "program_error",
        )

    def test_valid_pair_requires_two_boolean_ground_truth_outcomes(self) -> None:
        for oracle_success, vision_success in ((None, True), (False, None), (1, True)):
            with self.subTest(
                oracle_success=oracle_success,
                vision_success=vision_success,
            ):
                with self.assertRaisesRegex(ValueError, "boolean ground-truth"):
                    classify_outcome(oracle_success, vision_success)


if __name__ == "__main__":
    unittest.main()
