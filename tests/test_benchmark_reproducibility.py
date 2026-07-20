from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from benchmark.methods import FORMAL_METHOD_IDS, METHOD_SPECS
from benchmark.pairing import EpisodeFingerprint, validate_pair
from environments import PandaUTableEnv, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"
FIXED_TEST_SEEDS = (7, 19, 42)
FINGERPRINT_FIELDS = (
    "seed",
    "pick_position",
    "place_position",
    "pick_region",
    "place_region",
    "mass",
    "friction",
)
NONDETERMINISTIC_RESULT_FIELDS = frozenset(
    {
        "perception_latency_ms",
        "initial_perception_latency_ms",
        "pregrasp_perception_latency_ms",
        "final_visual_latency_ms",
        "wall_clock_time",
    }
)


def _episode_fingerprint(env: PandaUTableEnv) -> EpisodeFingerprint:
    episode = env.current_episode
    if episode is None:
        raise AssertionError("Environment reset did not create episode parameters")
    return EpisodeFingerprint.from_episode_parameters(episode)


def _deterministic_projection(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in NONDETERMINISTIC_RESULT_FIELDS
    }


class BenchmarkReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(CONFIG_PATH).with_modes(
            pick_mode="random",
            place_mode="random",
            physics_mode="random",
            observation_source="perception",
            controller_type="sensor_event_b1",
            viewer=False,
        )

    def _collect_structured_records(
        self, *, latency_offset: float
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed in FIXED_TEST_SEEDS:
            for method_id in FORMAL_METHOD_IDS:
                env = PandaUTableEnv(self.config)
                try:
                    env.reset(seed=seed)
                    fingerprint = _episode_fingerprint(env)
                    rows.append(
                        {
                            "execution_index": len(rows),
                            "seed": seed,
                            "method_id": method_id,
                            "external_state_source": METHOD_SPECS[
                                method_id
                            ].external_state_source,
                            "controller_type": "sensor_event_b1",
                            "episode_fingerprint": fingerprint.digest,
                            "episode_parameters": fingerprint.to_dict(),
                            # Deliberately differs between collections and must not
                            # participate in deterministic comparisons.
                            "perception_latency_ms": latency_offset + len(rows),
                        }
                    )
                finally:
                    env.close()
        return rows

    def test_same_seed_pair_uses_independent_envs_with_identical_fingerprint(self) -> None:
        for seed in FIXED_TEST_SEEDS:
            with self.subTest(seed=seed):
                oracle_env = PandaUTableEnv(self.config)
                vision_env = PandaUTableEnv(self.config)
                try:
                    self.assertIsNot(oracle_env, vision_env)
                    self.assertIsNot(oracle_env.model, vision_env.model)
                    self.assertIsNot(oracle_env.data, vision_env.data)
                    oracle_env.reset(seed=seed)
                    vision_env.reset(seed=seed)

                    oracle_fingerprint = _episode_fingerprint(oracle_env)
                    vision_fingerprint = _episode_fingerprint(vision_env)
                    self.assertEqual(
                        tuple(oracle_fingerprint.to_dict()), FINGERPRINT_FIELDS
                    )
                    self.assertEqual(oracle_fingerprint, vision_fingerprint)
                    self.assertEqual(
                        oracle_fingerprint.canonical_json(),
                        vision_fingerprint.canonical_json(),
                    )
                    self.assertEqual(
                        oracle_fingerprint.digest, vision_fingerprint.digest
                    )
                    validate_pair(oracle_fingerprint, vision_fingerprint)
                finally:
                    oracle_env.close()
                    vision_env.close()

    def test_repeated_collections_preserve_structure_values_and_order(self) -> None:
        with patch(
            "mujoco.Renderer",
            side_effect=AssertionError("Reproducibility test must stay renderer-free"),
        ):
            first = self._collect_structured_records(latency_offset=1.0)
            second = self._collect_structured_records(latency_offset=1000.0)

        expected_order = [
            (seed, method_id)
            for seed in FIXED_TEST_SEEDS
            for method_id in FORMAL_METHOD_IDS
        ]
        self.assertEqual(
            [(row["seed"], row["method_id"]) for row in first], expected_order
        )
        self.assertEqual(
            [(row["seed"], row["method_id"]) for row in second], expected_order
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            [_deterministic_projection(row) for row in first],
            [_deterministic_projection(row) for row in second],
        )

        for pair_index, seed in enumerate(FIXED_TEST_SEEDS):
            pair_rows = first[2 * pair_index : 2 * pair_index + 2]
            self.assertEqual([row["seed"] for row in pair_rows], [seed, seed])
            self.assertEqual(
                len({row["episode_fingerprint"] for row in pair_rows}), 1
            )
            self.assertEqual(
                pair_rows[0]["episode_parameters"],
                pair_rows[1]["episode_parameters"],
            )

    def test_fixed_seed_fingerprints_are_repeatable_and_seed_specific(self) -> None:
        first = self._collect_structured_records(latency_offset=0.0)
        second = self._collect_structured_records(latency_offset=0.0)
        oracle_first = {
            int(row["seed"]): str(row["episode_fingerprint"])
            for row in first
            if row["method_id"] == "b0_oracle"
        }
        oracle_second = {
            int(row["seed"]): str(row["episode_fingerprint"])
            for row in second
            if row["method_id"] == "b0_oracle"
        }
        self.assertEqual(oracle_first, oracle_second)
        self.assertEqual(set(oracle_first), set(FIXED_TEST_SEEDS))
        self.assertEqual(len(set(oracle_first.values())), len(FIXED_TEST_SEEDS))


if __name__ == "__main__":
    unittest.main()
