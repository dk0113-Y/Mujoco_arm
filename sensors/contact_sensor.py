from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactFeedback:
    """Debounced binary touch feedback, without privileged contact details."""

    left_finger_object_contact: bool
    right_finger_object_contact: bool
    bilateral_contact: bool
    contact_duration: float
    timestamp: float


@dataclass
class _DebounceState:
    stable: bool = False
    contrary_samples: int = 0

    def update(
        self, raw_state: bool, present_steps: int, absent_steps: int
    ) -> bool:
        if raw_state == self.stable:
            self.contrary_samples = 0
            return self.stable

        self.contrary_samples += 1
        required_samples = present_steps if raw_state else absent_steps
        if self.contrary_samples >= required_samples:
            self.stable = raw_state
            self.contrary_samples = 0
        return self.stable


def _required_id(
    model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str
) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing required object: {name}")
    return object_id


def _positive_step_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


class ContactSensor:
    """Convert MuJoCo geom pairs into a finite, debounced tactile proxy.

    Only boolean finger/object contact is exposed.  Exact contact positions,
    object state, and solver forces intentionally remain behind this adapter.
    Repeated reads at the same simulation timestamp count as one sensor sample.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | Any,
        *,
        object_geom_name: str = "pick_object_geom",
        object_geom_id: int | None = None,
        left_finger_body_name: str = "left_finger",
        right_finger_body_name: str = "right_finger",
        present_debounce_steps: int = 2,
        absent_debounce_steps: int = 2,
    ) -> None:
        self.model = model
        self.data = data
        self.present_debounce_steps = _positive_step_count(
            present_debounce_steps, "present_debounce_steps"
        )
        self.absent_debounce_steps = _positive_step_count(
            absent_debounce_steps, "absent_debounce_steps"
        )

        if object_geom_id is None:
            self.object_geom_id = _required_id(
                model, mujoco.mjtObj.mjOBJ_GEOM, object_geom_name
            )
        else:
            self.object_geom_id = int(object_geom_id)
            if not 0 <= self.object_geom_id < model.ngeom:
                raise ValueError("object_geom_id is outside the model geom range")

        left_body_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, left_finger_body_name
        )
        right_body_id = _required_id(
            model, mujoco.mjtObj.mjOBJ_BODY, right_finger_body_name
        )
        if left_body_id == right_body_id:
            raise RuntimeError("Left and right finger bodies must be distinct")

        self._left_finger_geom_ids = self._collidable_body_geoms(left_body_id)
        self._right_finger_geom_ids = self._collidable_body_geoms(right_body_id)
        if not self._left_finger_geom_ids:
            raise RuntimeError("Left finger body has no collidable geoms")
        if not self._right_finger_geom_ids:
            raise RuntimeError("Right finger body has no collidable geoms")
        if self._left_finger_geom_ids & self._right_finger_geom_ids:
            raise RuntimeError("Left and right finger geom sets overlap")
        if self.object_geom_id in (
            self._left_finger_geom_ids | self._right_finger_geom_ids
        ):
            raise RuntimeError("Object geom cannot also be a finger geom")

        self._left_state = _DebounceState()
        self._right_state = _DebounceState()
        self._last_timestamp: float | None = None
        self._bilateral_since: float | None = None
        self._last_feedback: ContactFeedback | None = None

    def _collidable_body_geoms(self, body_id: int) -> frozenset[int]:
        geom_ids = np.flatnonzero(
            (self.model.geom_bodyid == body_id) & (self.model.geom_contype != 0)
        )
        return frozenset(int(geom_id) for geom_id in geom_ids)

    def reset(self) -> None:
        """Clear debounce and duration state, normally after an environment reset."""

        self._left_state = _DebounceState()
        self._right_state = _DebounceState()
        self._last_timestamp = None
        self._bilateral_since = None
        self._last_feedback = None

    def _raw_contact_state(self) -> tuple[bool, bool]:
        left_contact = False
        right_contact = False
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.object_geom_id:
                other_geom = geom2
            elif geom2 == self.object_geom_id:
                other_geom = geom1
            else:
                continue

            if other_geom in self._left_finger_geom_ids:
                left_contact = True
            elif other_geom in self._right_finger_geom_ids:
                right_contact = True
            if left_contact and right_contact:
                break
        return left_contact, right_contact

    def read(self) -> ContactFeedback:
        timestamp = float(self.data.time)
        if not math.isfinite(timestamp):
            raise RuntimeError("Contact feedback timestamp is NaN or Inf")

        if self._last_timestamp is not None:
            if timestamp < self._last_timestamp:
                self.reset()
            elif timestamp == self._last_timestamp:
                if self._last_feedback is None:
                    raise RuntimeError("Contact sensor cache is inconsistent")
                return self._last_feedback

        raw_left, raw_right = self._raw_contact_state()
        stable_left = self._left_state.update(
            raw_left,
            self.present_debounce_steps,
            self.absent_debounce_steps,
        )
        stable_right = self._right_state.update(
            raw_right,
            self.present_debounce_steps,
            self.absent_debounce_steps,
        )
        bilateral = stable_left and stable_right
        if bilateral:
            if self._bilateral_since is None:
                self._bilateral_since = timestamp
            contact_duration = max(0.0, timestamp - self._bilateral_since)
        else:
            self._bilateral_since = None
            contact_duration = 0.0

        feedback = ContactFeedback(
            left_finger_object_contact=stable_left,
            right_finger_object_contact=stable_right,
            bilateral_contact=bilateral,
            contact_duration=float(contact_duration),
            timestamp=timestamp,
        )
        self._last_timestamp = timestamp
        self._last_feedback = feedback
        return feedback
