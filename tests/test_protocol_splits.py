from __future__ import annotations

import json
from pathlib import Path
import unittest

from benchmark.seed_io import load_seeds
from evaluation.protocol import load_protocol
from evaluation.split_analysis import SPLIT_ORDER, generate_split_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"


class ProtocolSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)
        cls.committed = {
            name: load_seeds(cls.protocol.splits[name].path) for name in SPLIT_ORDER
        }

    def test_split_sizes_uniqueness_disjointness_and_stable_order(self) -> None:
        self.assertEqual([len(self.committed[name]) for name in SPLIT_ORDER], [30, 60, 100])
        for name, seeds in self.committed.items():
            with self.subTest(name=name):
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertTrue(all(isinstance(seed, int) and seed >= 0 for seed in seeds))
                self.assertEqual(seeds, load_seeds(self.protocol.splits[name].path))
        for index, left_name in enumerate(SPLIT_ORDER):
            for right_name in SPLIT_ORDER[index + 1 :]:
                self.assertFalse(set(self.committed[left_name]) & set(self.committed[right_name]))

    def test_split_generation_reproduces_files_order_and_manifest_hash(self) -> None:
        regenerated, manifest = generate_split_plan(self.protocol)
        self.assertEqual(regenerated, self.committed)
        committed_manifest = json.loads(
            self.protocol.split_manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(manifest, committed_manifest)
        self.assertEqual(manifest["controller_outcomes_used"], False)
        self.assertEqual(len(manifest["manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
