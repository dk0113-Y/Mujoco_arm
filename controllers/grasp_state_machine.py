from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sensors import ContactFeedback, GripperFeedback


class GraspState(str, Enum):
    GRIPPER_OPEN = "gripper_open"
    CLOSING = "closing"
    GRASP_CANDIDATE = "grasp_candidate"
    GRASP_CONFIRMED = "grasp_confirmed"
    GRASP_LOST = "grasp_lost"
    RELEASED = "released"


@dataclass(frozen=True)
class GraspUpdate:
    state: GraspState
    empty_closure: bool = False
    bilateral_missing: bool = False
    sudden_further_closure: bool = False
    contact_loss_event: bool = False


@dataclass
class GraspStateMachine:
    """Pure, debounced grasp inference over gripper and binary-contact feedback."""

    empty_aperture_threshold: float
    minimum_grasp_aperture: float
    candidate_hold_steps: int
    confirmation_hold_steps: int
    contact_loss_hold_steps: int
    aperture_drop_threshold: float
    state: GraspState = GraspState.GRIPPER_OPEN
    candidate_aperture: float | None = None
    candidate_steps: int = 0
    confirmation_steps: int = 0
    contact_loss_steps: int = 0
    contact_loss_event_count: int = 0
    _last_bilateral: bool = False

    def begin_closing(self) -> None:
        self.state = GraspState.CLOSING
        self.candidate_aperture = None
        self.candidate_steps = 0
        self.confirmation_steps = 0
        self.contact_loss_steps = 0
        self._last_bilateral = False

    def update_candidate(
        self,
        gripper: GripperFeedback,
        contact: ContactFeedback,
        *,
        robot_table_collision: bool = False,
    ) -> GraspUpdate:
        if self.state not in (GraspState.CLOSING, GraspState.GRASP_CANDIDATE):
            raise RuntimeError(f"Candidate evidence is invalid in state {self.state.value}")
        if gripper.aperture <= self.empty_aperture_threshold:
            self.candidate_steps = 0
            return GraspUpdate(self.state, empty_closure=True)
        bilateral_missing = not contact.bilateral_contact
        valid = bool(
            gripper.commanded_state == "closing"
            and gripper.aperture > self.minimum_grasp_aperture
            and contact.bilateral_contact
            and not robot_table_collision
        )
        self.candidate_steps = self.candidate_steps + 1 if valid else 0
        if self.candidate_steps >= self.candidate_hold_steps:
            self.state = GraspState.GRASP_CANDIDATE
            self.candidate_aperture = float(gripper.aperture)
            self._last_bilateral = True
        return GraspUpdate(self.state, bilateral_missing=bilateral_missing)

    def update_confirmation(
        self,
        gripper: GripperFeedback,
        contact: ContactFeedback,
        *,
        trial_lift_completed: bool,
        robot_table_collision: bool = False,
    ) -> GraspUpdate:
        if self.state not in (GraspState.GRASP_CANDIDATE, GraspState.GRASP_CONFIRMED):
            raise RuntimeError(
                f"Confirmation evidence is invalid in state {self.state.value}"
            )
        if self.candidate_aperture is None:
            raise RuntimeError("Candidate aperture was not recorded")
        aperture_drop = self.candidate_aperture - float(gripper.aperture)
        sudden_closure = aperture_drop >= self.aperture_drop_threshold
        valid = bool(
            trial_lift_completed
            and contact.bilateral_contact
            and gripper.aperture > self.minimum_grasp_aperture
            and not sudden_closure
            and not robot_table_collision
        )
        self.confirmation_steps = self.confirmation_steps + 1 if valid else 0
        if self.confirmation_steps >= self.confirmation_hold_steps:
            self.state = GraspState.GRASP_CONFIRMED
            self._last_bilateral = True
            self.contact_loss_steps = 0
        return GraspUpdate(
            self.state,
            bilateral_missing=not contact.bilateral_contact,
            sudden_further_closure=sudden_closure,
        )

    def update_transport(
        self,
        gripper: GripperFeedback,
        contact: ContactFeedback,
    ) -> GraspUpdate:
        if self.state not in (
            GraspState.GRASP_CANDIDATE,
            GraspState.GRASP_CONFIRMED,
            GraspState.GRASP_LOST,
        ):
            raise RuntimeError(
                f"Transport evidence is invalid in state {self.state.value}"
            )
        contact_loss_event = self._last_bilateral and not contact.bilateral_contact
        if contact_loss_event:
            self.contact_loss_event_count += 1
        self._last_bilateral = contact.bilateral_contact
        self.contact_loss_steps = (
            0 if contact.bilateral_contact else self.contact_loss_steps + 1
        )
        if self.candidate_aperture is None:
            raise RuntimeError("Candidate aperture was not recorded")
        aperture_drop = self.candidate_aperture - float(gripper.aperture)
        sudden_closure = aperture_drop >= self.aperture_drop_threshold
        if self.contact_loss_steps >= self.contact_loss_hold_steps and sudden_closure:
            self.state = GraspState.GRASP_LOST
        return GraspUpdate(
            self.state,
            bilateral_missing=not contact.bilateral_contact,
            sudden_further_closure=sudden_closure,
            contact_loss_event=contact_loss_event,
        )

    def mark_released(self) -> None:
        self.state = GraspState.RELEASED
        self.contact_loss_steps = 0
        self._last_bilateral = False
