from __future__ import annotations

import unittest

from controllers.grasp_state_machine import GraspState, GraspStateMachine
from sensors import ContactFeedback, GripperFeedback


def gripper(aperture: float, *, command: str = "closing") -> GripperFeedback:
    return GripperFeedback(
        left_finger_position=0.5 * aperture,
        right_finger_position=0.5 * aperture,
        aperture=aperture,
        aperture_velocity=0.0,
        commanded_state=command,
        timestamp=1.0,
    )


def contact(left: bool, right: bool, *, duration: float = 0.0) -> ContactFeedback:
    return ContactFeedback(
        left_finger_object_contact=left,
        right_finger_object_contact=right,
        bilateral_contact=left and right,
        contact_duration=duration,
        timestamp=1.0,
    )


def monitor() -> GraspStateMachine:
    return GraspStateMachine(
        empty_aperture_threshold=0.004,
        minimum_grasp_aperture=0.008,
        candidate_hold_steps=3,
        confirmation_hold_steps=2,
        contact_loss_hold_steps=3,
        aperture_drop_threshold=0.003,
    )


class GraspStateMachineTests(unittest.TestCase):
    def test_normal_candidate_requires_consecutive_bilateral_contact(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for _ in range(2):
            update = state_machine.update_candidate(gripper(0.05), contact(True, True))
            self.assertEqual(update.state, GraspState.CLOSING)
        update = state_machine.update_candidate(gripper(0.05), contact(True, True))
        self.assertEqual(update.state, GraspState.GRASP_CANDIDATE)
        self.assertAlmostEqual(state_machine.candidate_aperture, 0.05)

    def test_empty_closure_and_unilateral_contact_are_not_candidates(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        unilateral = state_machine.update_candidate(
            gripper(0.03), contact(True, False)
        )
        self.assertTrue(unilateral.bilateral_missing)
        self.assertEqual(unilateral.state, GraspState.CLOSING)
        empty = state_machine.update_candidate(gripper(0.003), contact(False, False))
        self.assertTrue(empty.empty_closure)
        self.assertNotEqual(empty.state, GraspState.GRASP_CANDIDATE)

    def test_single_step_contact_jitter_resets_candidate_hold(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for bilateral in (True, True, False, True, True):
            state_machine.update_candidate(
                gripper(0.05), contact(bilateral, bilateral)
            )
        self.assertEqual(state_machine.state, GraspState.CLOSING)
        state_machine.update_candidate(gripper(0.05), contact(True, True))
        self.assertEqual(state_machine.state, GraspState.GRASP_CANDIDATE)

    def test_trial_lift_confirmation_requires_stable_feedback(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for _ in range(3):
            state_machine.update_candidate(gripper(0.05), contact(True, True))
        update = state_machine.update_confirmation(
            gripper(0.05),
            contact(True, True),
            trial_lift_completed=False,
        )
        self.assertEqual(update.state, GraspState.GRASP_CANDIDATE)
        self.assertFalse(update.lift_predicate)
        self.assertTrue(update.contact_predicate)
        self.assertTrue(update.minimum_aperture_predicate)
        self.assertTrue(update.aperture_retention_predicate)
        self.assertTrue(update.collision_free_predicate)
        self.assertFalse(update.combined_predicate)
        self.assertEqual(update.hold_steps, 0)
        for _ in range(2):
            update = state_machine.update_confirmation(
                gripper(0.05),
                contact(True, True),
                trial_lift_completed=True,
            )
        self.assertEqual(update.state, GraspState.GRASP_CONFIRMED)
        self.assertTrue(update.combined_predicate)
        self.assertEqual(update.hold_steps, 2)

    def test_confirmation_snapshot_reports_exact_threshold_and_collision_failures(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for _ in range(3):
            candidate = state_machine.update_candidate(
                gripper(0.05), contact(True, True)
            )
        self.assertTrue(candidate.commanded_closing_predicate)
        self.assertTrue(candidate.minimum_aperture_predicate)
        self.assertTrue(candidate.contact_predicate)
        self.assertTrue(candidate.collision_free_predicate)
        self.assertTrue(candidate.combined_predicate)

        threshold = state_machine.update_confirmation(
            gripper(0.047),
            contact(True, True),
            trial_lift_completed=True,
        )
        self.assertAlmostEqual(threshold.aperture_drop, 0.003)
        self.assertFalse(threshold.aperture_retention_predicate)
        self.assertFalse(threshold.combined_predicate)
        self.assertEqual(threshold.hold_steps, 0)

        collision = state_machine.update_confirmation(
            gripper(0.05),
            contact(True, True),
            trial_lift_completed=True,
            robot_table_collision=True,
        )
        self.assertFalse(collision.collision_free_predicate)
        self.assertFalse(collision.combined_predicate)
        self.assertEqual(collision.hold_steps, 0)

    def test_confirmation_rejects_aperture_below_configured_minimum(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for _ in range(3):
            state_machine.update_candidate(gripper(0.009), contact(True, True))
        for _ in range(3):
            update = state_machine.update_confirmation(
                gripper(0.007),
                contact(True, True),
                trial_lift_completed=True,
            )
        self.assertEqual(update.state, GraspState.GRASP_CANDIDATE)
        self.assertEqual(state_machine.confirmation_steps, 0)

    def test_sustained_transfer_contact_loss_and_closure_marks_grasp_lost(self) -> None:
        state_machine = monitor()
        state_machine.begin_closing()
        for _ in range(3):
            state_machine.update_candidate(gripper(0.05), contact(True, True))
        for _ in range(2):
            state_machine.update_confirmation(
                gripper(0.05),
                contact(True, True),
                trial_lift_completed=True,
            )
        self.assertEqual(state_machine.state, GraspState.GRASP_CONFIRMED)
        for _ in range(2):
            update = state_machine.update_transport(
                gripper(0.046), contact(False, False)
            )
            self.assertNotEqual(update.state, GraspState.GRASP_LOST)
        update = state_machine.update_transport(
            gripper(0.046), contact(False, False)
        )
        self.assertEqual(update.state, GraspState.GRASP_LOST)
        self.assertEqual(state_machine.contact_loss_event_count, 1)


if __name__ == "__main__":
    unittest.main()
