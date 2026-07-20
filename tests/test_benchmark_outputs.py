from __future__ import annotations

import csv
from dataclasses import fields
from enum import Enum
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.schemas import (
    EPISODE_METADATA_FIELDS,
    EPISODE_RESULT_FIELDS,
    FAILURE_COUNT_FIELDS,
    PAIRED_RESULT_FIELDS,
    episode_fieldnames,
    validate_finite_json,
    write_csv,
    write_json,
)
from evaluation import EpisodeResult


class _OutputKind(str, Enum):
    PILOT = "pilot"


class BenchmarkOutputSchemaTests(unittest.TestCase):
    def test_declared_schemas_have_required_stable_order(self) -> None:
        self.assertEqual(
            EPISODE_METADATA_FIELDS,
            (
                "benchmark_name",
                "pair_id",
                "method_id",
                "external_state_source",
                "execution_index",
                "episode_fingerprint",
                "pair_valid",
                "program_error",
            ),
        )
        expected_result_fields = tuple(
            field.name
            for field in fields(EpisodeResult)
            if field.name not in {"key_errors", "stage_durations"}
        )
        self.assertEqual(EPISODE_RESULT_FIELDS, expected_result_fields)
        self.assertEqual(
            PAIRED_RESULT_FIELDS,
            (
                "pair_id",
                "seed",
                "pair_valid",
                "pair_error",
                "fingerprint",
                "oracle_ground_truth_success",
                "vision_ground_truth_success",
                "oracle_controller_reported_success",
                "vision_controller_reported_success",
                "oracle_failure_reason",
                "vision_failure_reason",
                "oracle_final_stage",
                "vision_final_stage",
                "oracle_simulation_time",
                "vision_simulation_time",
                "oracle_collision_count",
                "vision_collision_count",
                "vision_object_position_error",
                "vision_target_position_error",
                "outcome_category",
            ),
        )
        self.assertEqual(
            FAILURE_COUNT_FIELDS,
            ("method_id", "failure_reason", "count"),
        )
        for schema in (
            EPISODE_METADATA_FIELDS,
            EPISODE_RESULT_FIELDS,
            PAIRED_RESULT_FIELDS,
            FAILURE_COUNT_FIELDS,
        ):
            with self.subTest(schema=schema[:2]):
                self.assertEqual(len(schema), len(set(schema)))

    def test_episode_fieldnames_append_dynamic_columns_in_sorted_order(self) -> None:
        rows = [
            {
                "method_id": "b0_oracle",
                "stage_duration.withdraw": 1.0,
                "key_error.z_error": 0.2,
            },
            {
                "method_id": "b1_vision",
                "key_error.a_error": 0.1,
            },
        ]
        expected_prefix = EPISODE_METADATA_FIELDS + EPISODE_RESULT_FIELDS
        expected_dynamic = (
            "key_error.a_error",
            "key_error.z_error",
            "stage_duration.withdraw",
        )
        self.assertEqual(episode_fieldnames(rows), expected_prefix + expected_dynamic)
        self.assertEqual(
            episode_fieldnames(reversed(rows)), expected_prefix + expected_dynamic
        )

    def test_csv_writer_preserves_rows_and_json_encodes_composite_values(self) -> None:
        fieldnames = (
            "method_id",
            "sampled_pick_position",
            "payload",
            "pair_valid",
            "optional",
        )
        rows = [
            {
                "method_id": "b0_oracle",
                "sampled_pick_position": (0.5, 0.1, 0.246),
                "payload": {"z": 2, "a": 1},
                "pair_valid": True,
                "optional": None,
            },
            {
                "method_id": "b1_vision",
                "sampled_pick_position": [0.5, 0.1, 0.246],
                "payload": {"unicode": "视觉"},
                "pair_valid": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            write_csv(path, rows, fieldnames)
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                loaded = list(reader)
                loaded_fieldnames = tuple(reader.fieldnames or ())

        self.assertEqual(loaded_fieldnames, fieldnames)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["method_id"], "b0_oracle")
        self.assertEqual(
            json.loads(loaded[0]["sampled_pick_position"]),
            [0.5, 0.1, 0.246],
        )
        self.assertEqual(loaded[0]["payload"], '{"a":1,"z":2}')
        self.assertEqual(json.loads(loaded[1]["payload"]), {"unicode": "视觉"})
        self.assertEqual(loaded[0]["pair_valid"], "True")
        self.assertEqual(loaded[0]["optional"], "")
        self.assertEqual(loaded[1]["optional"], "")

    def test_csv_writer_rejects_unknown_fields_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            with self.assertRaisesRegex(ValueError, "absent from schema"):
                write_csv(path, [{"known": 1, "unexpected": 2}], ("known",))
            self.assertFalse(path.exists())

    def test_empty_csv_still_has_exact_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired_results.csv"
            write_csv(path, [], PAIRED_RESULT_FIELDS)
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
        self.assertEqual(rows, [list(PAIRED_RESULT_FIELDS)])

    def test_non_finite_values_are_rejected_at_every_nesting_level(self) -> None:
        cases = (
            float("nan"),
            float("inf"),
            -float("inf"),
            {"nested": [1.0, float("nan")]},
            (0.0, {"nested": float("inf")}),
        )
        for value in cases:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    validate_finite_json(value)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                write_csv(path, [{"value": float("nan")}], ("value",))

    def test_json_validation_normalizes_supported_values_and_rejects_objects(self) -> None:
        value = {
            "kind": _OutputKind.PILOT,
            "path": Path("configs/benchmark0/pilot_seeds.txt"),
            "tuple": (1, 2.5, True, None),
        }
        self.assertEqual(
            validate_finite_json(value),
            {
                "kind": "pilot",
                "path": str(Path("configs/benchmark0/pilot_seeds.txt")),
                "tuple": [1, 2.5, True, None],
            },
        )
        with self.assertRaisesRegex(TypeError, "not JSON serializable"):
            validate_finite_json(object())

    def test_json_writer_is_strict_parseable_and_deterministic(self) -> None:
        payload = {"z": (3, 2, 1), "a": {"pilot": True}}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, payload)
            write_json(second, payload)
            first_text = first.read_text(encoding="utf-8")
            second_text = second.read_text(encoding="utf-8")

        self.assertEqual(first_text, second_text)
        self.assertTrue(first_text.endswith("\n"))
        self.assertEqual(json.loads(first_text), {"a": {"pilot": True}, "z": [3, 2, 1]})
        self.assertNotIn("NaN", first_text)
        self.assertNotIn("Infinity", first_text)


if __name__ == "__main__":
    unittest.main()
