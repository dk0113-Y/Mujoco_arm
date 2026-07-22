from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock


from benchmark.pairing import EpisodeFingerprint
from benchmark.schemas import EPISODE_METADATA_FIELDS, EPISODE_RESULT_FIELDS
from environments import load_config
from evaluation.protocol import load_protocol, validate_baseline_compatibility
from scripts import verify_b1_freeze as freeze


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs/protocols/evaluation_protocol_v1.toml"
TEMPLATE_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_calibration_template.toml"
FROZEN_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_v1.toml"
MANIFEST_PATH = PROJECT_ROOT / "configs/baselines/b1_vision_v1_manifest.json"


def _row(seed: int, method: str) -> dict[str, str]:
    task = {
        "seed": seed,
        "sampled_pick_position": [0.3, 0.1, 0.246],
        "sampled_place_position": [0.6, -0.1, 0.222],
        "pick_region": "front",
        "place_region": "front",
        "sampled_mass": 0.1,
        "sampled_friction": [1.0, 0.01, 0.001],
    }
    digest = EpisodeFingerprint.from_episode_result(task).digest
    return {
        "pair_id": f"pair_{seed}",
        "method_id": method,
        "external_state_source": "oracle" if method == "b0_oracle" else "vision",
        "episode_fingerprint": digest,
        "pair_valid": "True",
        "program_error": "",
        "seed": str(seed),
        "sampled_pick_position": "[0.3,0.1,0.246]",
        "sampled_place_position": "[0.6,-0.1,0.222]",
        "pick_region": "front",
        "place_region": "front",
        "sampled_mass": "0.1",
        "sampled_friction": "[1.0,0.01,0.001]",
        "final_stage": "completed",
        "failure_reason": "",
        "controller_reported_success": "True",
        "privileged_ground_truth_success": "True",
        "placement_success": "True",
        "safe_task_success": "True",
        "collision_count": "0",
        "collision_episode": "False",
        "false_positive": "False",
        "false_negative": "False",
        "unexplained_failure": "False",
        "result_fields_complete": "True",
        "simulation_time": "18.0",
    }


def _rows() -> list[dict[str, str]]:
    return [
        _row(seed, method)
        for seed in (1, 2)
        for method in freeze.METHOD_IDS
    ]


class FrozenConfigTests(unittest.TestCase):
    def test_frozen_config_loads_and_is_behavior_equivalent(self) -> None:
        template = load_config(TEMPLATE_PATH)
        frozen = load_config(FROZEN_PATH)
        self.assertEqual(frozen, template)
        self.assertEqual(frozen.controller, template.controller)
        self.assertEqual(frozen.b1, template.b1)
        self.assertEqual(FROZEN_PATH.read_bytes(), TEMPLATE_PATH.read_bytes())

    def test_success_and_protocol_protected_fields_are_unchanged(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        frozen = load_config(FROZEN_PATH)
        validate_baseline_compatibility(protocol, frozen)
        self.assertEqual(
            frozen.b1.final_place_xy_tolerance,
            protocol.environment.b1.final_place_xy_tolerance,
        )
        self.assertEqual(
            frozen.b1.final_place_height_tolerance,
            protocol.environment.b1.final_place_height_tolerance,
        )
        for name in ("workspace", "pick", "place", "physics", "simulation", "camera"):
            self.assertEqual(
                getattr(frozen, name), getattr(protocol.environment, name), name
            )

    def test_manifest_has_required_committed_pending_tag_state(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        required = {
            "artifact_schema_version",
            "baseline_id",
            "display_name",
            "freeze_state",
            "behavior_frozen",
            "final_git_commit",
            "freeze_package_commit",
            "tag_name",
            "tag_created",
            "verified_behavior_commit",
            "frozen_config_path",
            "frozen_config_sha256",
            "source_template_path",
            "source_template_sha256",
            "behavior_config_equivalent",
            "protocol_id",
            "protocol_version",
            "metrics_schema_version",
            "protocol_sha256",
            "split_id",
            "calibration_split_sha256",
            "freeze_verification_output_path",
            "reference_round_0_path",
            "round_0_5_report_path",
            "created_at",
        }
        self.assertFalse(required - set(manifest))
        self.assertEqual(manifest["baseline_id"], "b1_vision_v1")
        self.assertEqual(manifest["freeze_state"], "committed_pending_tag")
        self.assertTrue(manifest["behavior_frozen"])
        self.assertTrue(manifest["behavior_config_equivalent"])
        self.assertEqual(
            manifest["final_git_commit"],
            "129036f8eacc4d24aa892d7510c51dec33407c47",
        )
        self.assertEqual(
            manifest["freeze_package_commit"], manifest["final_git_commit"]
        )
        self.assertIsNone(manifest["tag_name"])
        self.assertFalse(manifest["tag_created"])
        self.assertEqual(
            manifest["verified_behavior_commit"],
            "bf6a07945f396f7b98f5c24cf94d1a97b8dc7f9d",
        )


class FreezeComparisonTests(unittest.TestCase):
    def test_equal_rows_pass(self) -> None:
        rows = _rows()
        comparisons = freeze.compare_episode_rows(rows, copy.deepcopy(rows), (1, 2))
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(row["all_exact_fields_match"] for row in comparisons))

    def test_rejects_missing_and_duplicate_seed_method_rows(self) -> None:
        rows = _rows()
        with self.assertRaisesRegex(freeze.FreezeVerificationError, "coverage mismatch"):
            freeze.compare_episode_rows(rows, rows[:-1], (1, 2))
        duplicate = copy.deepcopy(rows) + [copy.deepcopy(rows[0])]
        with self.assertRaisesRegex(freeze.FreezeVerificationError, "duplicate episode"):
            freeze.compare_episode_rows(rows, duplicate, (1, 2))

    def test_rejects_fingerprint_failure_success_and_collision_mismatches(self) -> None:
        mutations = {
            "fingerprint mismatch": ("episode_fingerprint", "0" * 64),
            "failure_reason": ("failure_reason", "grasp_not_confirmed"),
            "controller success": ("controller_reported_success", "False"),
            "safe success": ("safe_task_success", "False"),
            "collision": ("collision_count", "1"),
        }
        reference = _rows()
        for label, (field, value) in mutations.items():
            with self.subTest(label=label):
                candidate = copy.deepcopy(reference)
                candidate[0][field] = value
                with self.assertRaises(freeze.FreezeVerificationError):
                    freeze.compare_episode_rows(reference, candidate, (1, 2))

    def test_rejects_program_error(self) -> None:
        reference = _rows()
        candidate = copy.deepcopy(reference)
        candidate[0]["program_error"] = "unexpected_exception"
        with self.assertRaisesRegex(freeze.FreezeVerificationError, "program error"):
            freeze.compare_episode_rows(reference, candidate, (1, 2))

    def test_simulation_time_uses_only_round_0_5_tolerance(self) -> None:
        reference = _rows()
        candidate = copy.deepcopy(reference)
        candidate[0]["simulation_time"] = str(
            18.0 + 0.99 * freeze.SIMULATION_TIME_ABSOLUTE_TOLERANCE
        )
        freeze.compare_episode_rows(reference, candidate, (1, 2))
        candidate[0]["simulation_time"] = str(
            18.0 + 1.01 * freeze.SIMULATION_TIME_ABSOLUTE_TOLERANCE
        )
        with self.assertRaisesRegex(freeze.FreezeVerificationError, "simulation_time"):
            freeze.compare_episode_rows(reference, candidate, (1, 2))
        self.assertEqual(freeze.SIMULATION_TIME_ABSOLUTE_TOLERANCE, 0.0020000001)

    def test_rejects_production_metric_mismatch(self) -> None:
        reference = {
            "safe_task_success_count": 17,
            "safe_successful_simulation_time_median": 18.0,
        }
        candidate = dict(reference, safe_task_success_count=18)
        with self.assertRaisesRegex(
            freeze.FreezeVerificationError, "Production metric mismatch"
        ):
            freeze.compare_nested_metrics(reference, candidate)

    def test_verifier_does_not_read_development_or_held_out(self) -> None:
        accessed: list[Path] = []
        real_load_seeds = freeze.load_seeds

        def recording_load(path: str | Path) -> list[int]:
            accessed.append(Path(path).resolve())
            return real_load_seeds(path)

        with mock.patch.object(freeze, "load_seeds", side_effect=recording_load):
            protocol, seeds = freeze._load_protocol_and_calibration(PROTOCOL_PATH)
        self.assertEqual(len(seeds), 30)
        self.assertEqual(accessed, [protocol.splits["calibration"].path.resolve()])
        self.assertNotIn(protocol.splits["development"].path.resolve(), accessed)
        self.assertNotIn(protocol.splits["held_out_test"].path.resolve(), accessed)

    def test_diagnostic_fields_are_absent_from_formal_episode_schema(self) -> None:
        fields = EPISODE_METADATA_FIELDS + EPISODE_RESULT_FIELDS
        self.assertFalse(
            any(
                field.startswith("diagnostic.")
                or field.startswith("privileged_diagnostic.")
                for field in fields
            )
        )


if __name__ == "__main__":
    unittest.main()
