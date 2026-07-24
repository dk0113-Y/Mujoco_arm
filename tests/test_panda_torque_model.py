from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORQUE_MODEL = PROJECT_ROOT / "models" / "panda_torque" / "panda_torque.xml"
TORQUE_SCENE = PROJECT_ROOT / "models" / "panda_torque" / "scene_torque.xml"
ORIGINAL_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)
ORIGINAL_SHA256 = "c5a92e6ff47e7282ea303ffe13530ffe150248a22bb4b349a9369881e52facf0"


def object_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    result = int(mujoco.mj_name2id(model, object_type, name))
    if result < 0:
        raise AssertionError(f"Missing object {name}")
    return result


class PandaTorqueModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(TORQUE_SCENE))
        cls.arm_actuators = np.asarray(
            [
                object_id(cls.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}")
                for i in range(1, 8)
            ],
            dtype=int,
        )
        cls.arm_joints = np.asarray(
            [
                object_id(cls.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ],
            dtype=int,
        )

    def test_isolated_torque_scene_loads(self) -> None:
        self.assertEqual(self.model.nu, 8)
        self.assertEqual(self.model.nq, 9)
        self.assertEqual(self.model.nv, 9)
        self.assertAlmostEqual(float(self.model.opt.timestep), 0.002)

    def test_first_seven_actuators_are_unit_direct_motors(self) -> None:
        np.testing.assert_array_equal(self.arm_actuators, np.arange(7))
        np.testing.assert_allclose(
            self.model.actuator_gainprm[self.arm_actuators, 0], 1.0
        )
        np.testing.assert_allclose(
            self.model.actuator_biasprm[self.arm_actuators, :3], 0.0
        )
        np.testing.assert_allclose(
            self.model.actuator_gear[self.arm_actuators, 0], 1.0
        )
        np.testing.assert_array_equal(
            self.model.actuator_dyntype[self.arm_actuators],
            int(mujoco.mjtDyn.mjDYN_NONE),
        )

    def test_actuator_joint_order_and_transmission_are_fixed(self) -> None:
        np.testing.assert_array_equal(
            self.model.actuator_trntype[self.arm_actuators],
            int(mujoco.mjtTrn.mjTRN_JOINT),
        )
        np.testing.assert_array_equal(
            self.model.actuator_trnid[self.arm_actuators, 0], self.arm_joints
        )

    def test_positive_control_increases_same_joint_acceleration(self) -> None:
        home_id = object_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        baseline = mujoco.MjData(self.model)
        baseline.qpos[:] = self.model.key_qpos[home_id]
        baseline.ctrl[:] = self.model.key_ctrl[home_id]
        mujoco.mj_forward(self.model, baseline)
        base_acceleration = baseline.qacc.copy()
        for index, joint_id in enumerate(self.arm_joints):
            data = mujoco.MjData(self.model)
            data.qpos[:] = self.model.key_qpos[home_id]
            data.ctrl[:] = self.model.key_ctrl[home_id]
            data.ctrl[index] = 0.1
            mujoco.mj_forward(self.model, data)
            dof = int(self.model.jnt_dofadr[joint_id])
            self.assertGreater(
                float(data.qacc[dof] - base_acceleration[dof]),
                0.0,
                f"positive actuator{index + 1} did not accelerate joint{index + 1} positively",
            )

    def test_actuator_force_limits_are_physical_and_effective(self) -> None:
        expected = np.asarray([87, 87, 87, 87, 12, 12, 12], dtype=float)
        np.testing.assert_allclose(
            self.model.actuator_ctrlrange[self.arm_actuators],
            np.column_stack((-expected, expected)),
        )
        np.testing.assert_allclose(
            self.model.actuator_forcerange[self.arm_actuators],
            np.column_stack((-expected, expected)),
        )
        data = mujoco.MjData(self.model)
        data.ctrl[:7] = 1000.0
        mujoco.mj_forward(self.model, data)
        np.testing.assert_allclose(data.actuator_force[:7], expected)
        np.testing.assert_allclose(data.qfrc_actuator[:7], expected)

    def test_gripper_actuator_is_independent_tendon_interface(self) -> None:
        gripper = object_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.assertEqual(gripper, 7)
        self.assertEqual(
            int(self.model.actuator_trntype[gripper]),
            int(mujoco.mjtTrn.mjTRN_TENDON),
        )
        np.testing.assert_allclose(
            self.model.actuator_ctrlrange[gripper], [0.0, 255.0]
        )
        self.assertNotIn(gripper, self.arm_actuators)

    def test_non_actuator_robot_properties_match_pinned_model(self) -> None:
        original = mujoco.MjModel.from_xml_path(str(ORIGINAL_MODEL))
        torque = mujoco.MjModel.from_xml_path(str(TORQUE_MODEL))
        self.assertEqual((torque.nbody, torque.njnt, torque.ngeom), (
            original.nbody,
            original.njnt,
            original.ngeom,
        ))
        np.testing.assert_allclose(torque.body_mass, original.body_mass)
        np.testing.assert_allclose(torque.body_inertia, original.body_inertia)
        np.testing.assert_allclose(torque.jnt_range, original.jnt_range)
        np.testing.assert_allclose(torque.dof_damping, original.dof_damping)
        np.testing.assert_allclose(torque.geom_friction, original.geom_friction)

    def test_original_b0_b1_model_fingerprint_is_unchanged(self) -> None:
        digest = hashlib.sha256(ORIGINAL_MODEL.read_bytes()).hexdigest()
        self.assertEqual(digest, ORIGINAL_SHA256)


if __name__ == "__main__":
    unittest.main()
