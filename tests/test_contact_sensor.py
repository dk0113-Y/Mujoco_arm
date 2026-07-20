from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
import unittest

import mujoco
import numpy as np

from environments import PandaUTableEnv
from sensors import ContactFeedback, ContactSensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


@dataclass(frozen=True)
class _FakeContact:
    geom1: int
    geom2: int


class _FakeContactData:
    def __init__(self) -> None:
        self.time = 0.0
        self.contact: list[_FakeContact] = []

    @property
    def ncon(self) -> int:
        return len(self.contact)


class ContactSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = PandaUTableEnv(CONFIG_PATH)
        cls.env.reset(seed=42)
        model = cls.env.model
        left_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        right_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        cls.left_geom = int(
            np.flatnonzero(
                (model.geom_bodyid == left_body) & (model.geom_contype != 0)
            )[0]
        )
        cls.right_geom = int(
            np.flatnonzero(
                (model.geom_bodyid == right_body) & (model.geom_contype != 0)
            )[0]
        )
        cls.left_visual_geom = int(
            np.flatnonzero(
                (model.geom_bodyid == left_body) & (model.geom_contype == 0)
            )[0]
        )
        cls.object_geom = cls.env.object_geom_id

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def setUp(self) -> None:
        self.data = _FakeContactData()
        self.sensor = ContactSensor(
            self.env.model,
            self.data,
            present_debounce_steps=2,
            absent_debounce_steps=2,
        )

    def _sample(
        self, time: float, contacts: list[tuple[int, int]]
    ) -> ContactFeedback:
        self.data.time = time
        self.data.contact = [_FakeContact(*pair) for pair in contacts]
        return self.sensor.read()

    def test_left_right_and_bilateral_pairs_are_order_independent(self) -> None:
        left_pairs = [(self.left_geom, self.object_geom)]
        first_left = self._sample(0.01, left_pairs)
        stable_left = self._sample(0.02, left_pairs)
        self.assertFalse(first_left.left_finger_object_contact)
        self.assertTrue(stable_left.left_finger_object_contact)
        self.assertFalse(stable_left.right_finger_object_contact)
        self.assertFalse(stable_left.bilateral_contact)

        both_pairs = [
            (self.object_geom, self.left_geom),
            (self.right_geom, self.object_geom),
        ]
        first_both = self._sample(0.03, both_pairs)
        stable_both = self._sample(0.04, both_pairs)
        self.assertFalse(first_both.bilateral_contact)
        self.assertTrue(stable_both.left_finger_object_contact)
        self.assertTrue(stable_both.right_finger_object_contact)
        self.assertTrue(stable_both.bilateral_contact)
        self.assertEqual(stable_both.contact_duration, 0.0)

        held = self._sample(0.14, both_pairs)
        self.assertAlmostEqual(held.contact_duration, 0.10, places=12)

    def test_present_and_absent_debounce_reject_single_step_jitter(self) -> None:
        both_pairs = [
            (self.left_geom, self.object_geom),
            (self.object_geom, self.right_geom),
        ]
        transient = self._sample(0.01, both_pairs)
        cleared = self._sample(0.02, [])
        self.assertFalse(transient.bilateral_contact)
        self.assertFalse(cleared.bilateral_contact)

        self._sample(0.03, both_pairs)
        stable = self._sample(0.04, both_pairs)
        one_missing = self._sample(0.05, [])
        absent = self._sample(0.06, [])
        self.assertTrue(stable.bilateral_contact)
        self.assertTrue(one_missing.bilateral_contact)
        self.assertFalse(absent.bilateral_contact)
        self.assertEqual(absent.contact_duration, 0.0)

    def test_duplicate_read_does_not_bypass_debounce(self) -> None:
        pairs = [
            (self.left_geom, self.object_geom),
            (self.right_geom, self.object_geom),
        ]
        first = self._sample(0.01, pairs)
        duplicate = self.sensor.read()
        self.assertIs(first, duplicate)
        self.assertFalse(duplicate.bilateral_contact)

        second = self._sample(0.02, pairs)
        self.assertTrue(second.bilateral_contact)

    def test_real_mjdata_ignores_table_object_and_visual_geom_pairs(self) -> None:
        real_sensor = ContactSensor(
            self.env.model,
            self.env.data,
            present_debounce_steps=1,
            absent_debounce_steps=1,
        )
        actual = real_sensor.read()
        self.assertGreater(self.env.data.ncon, 0)
        self.assertFalse(actual.left_finger_object_contact)
        self.assertFalse(actual.right_finger_object_contact)

        ignored_visual = self._sample(
            0.01, [(self.left_visual_geom, self.object_geom)]
        )
        self._sample(0.02, [(self.left_visual_geom, self.object_geom)])
        self.assertFalse(ignored_visual.left_finger_object_contact)
        self.assertFalse(self.sensor.read().left_finger_object_contact)

    def test_feedback_is_frozen_and_exposes_no_privileged_details(self) -> None:
        feedback = self._sample(0.01, [])
        self.assertEqual(
            {field.name for field in fields(ContactFeedback)},
            {
                "left_finger_object_contact",
                "right_finger_object_contact",
                "bilateral_contact",
                "contact_duration",
                "timestamp",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            feedback.bilateral_contact = True  # type: ignore[misc]
        for forbidden in ("position", "force", "object_position", "object_velocity"):
            self.assertFalse(hasattr(feedback, forbidden))

    def test_invalid_debounce_and_model_names_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            ContactSensor(self.env.model, self.data, present_debounce_steps=0)
        with self.assertRaisesRegex(RuntimeError, "missing required object"):
            ContactSensor(
                self.env.model,
                self.data,
                left_finger_body_name="missing_left_finger",
            )


if __name__ == "__main__":
    unittest.main()
