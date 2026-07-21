from __future__ import annotations

from pathlib import Path
import unittest

from benchmark.seed_io import load_seeds
from evaluation.protocol import load_protocol
from evaluation.split_analysis import (
    REGIONS,
    REGION_PAIRS,
    SPLIT_ORDER,
    collect_task_samples,
    distribution_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "protocols" / "evaluation_protocol_v1.toml"


class ProtocolDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = load_protocol(PROTOCOL_PATH)
        cls.summaries = {}
        for name in SPLIT_ORDER:
            samples = collect_task_samples(
                cls.protocol, load_seeds(cls.protocol.splits[name].path)
            )
            cls.summaries[name] = distribution_summary(samples)

    def test_every_split_has_region_pair_and_same_cross_coverage(self) -> None:
        for name, summary in self.summaries.items():
            with self.subTest(name=name):
                self.assertEqual(summary["illegal_sample_count"], 0)
                self.assertTrue(all(summary["pick_region_counts"][region] > 0 for region in REGIONS))
                self.assertTrue(all(summary["place_region_counts"][region] > 0 for region in REGIONS))
                self.assertTrue(
                    all(
                        summary["region_pair_counts"][f"{pick}->{place}"] > 0
                        for pick, place in REGION_PAIRS
                    )
                )
                self.assertGreater(summary["same_cross_counts"]["same_region"], 0)
                self.assertGreater(summary["same_cross_counts"]["cross_region"], 0)

    def test_mass_friction_and_distance_are_legal_and_cover_ranges(self) -> None:
        physics = self.protocol.environment.physics
        for name, summary in self.summaries.items():
            with self.subTest(name=name):
                self.assertGreaterEqual(summary["mass"]["minimum"], physics.mass_range[0])
                self.assertLessEqual(summary["mass"]["maximum"], physics.mass_range[1])
                self.assertLess(summary["mass"]["minimum"], 0.08)
                self.assertGreater(summary["mass"]["maximum"], 0.18)
                self.assertGreaterEqual(
                    summary["pick_place_distance"]["minimum"],
                    self.protocol.environment.place.minimum_xy_distance,
                )
                self.assertGreater(summary["pick_place_distance"]["maximum"], 1.0)
                for index, friction in enumerate(summary["friction"]):
                    self.assertGreaterEqual(friction["minimum"], physics.friction_min[index])
                    self.assertLessEqual(friction["maximum"], physics.friction_max[index])


if __name__ == "__main__":
    unittest.main()
