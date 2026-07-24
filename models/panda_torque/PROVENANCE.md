# Panda torque model provenance

`panda_torque.xml` is a simulation-specific derivative of the Franka Panda
model pinned in this repository at:

```text
models/mujoco_menagerie/franka_emika_panda/panda.xml
MuJoCo Menagerie commit 71f066ad0be9cd271f7ed58c030243ef157af9f4
```

The upstream model is licensed under Apache-2.0; its license and README remain
in the pinned source directory. Geometry, inertial data, collision geometry,
joint ranges, equality constraints, tendon, and gripper actuator are retained.
No mesh asset is copied.

The local MuJoCo adaptation changes only the arm actuation contract and the
home keyframe arm control values:

- `actuator1` through `actuator7` are `<motor>` direct drives;
- each motor has joint transmission, `gear=1`, fixed gain 1, zero bias, and no
  activation dynamics;
- explicit control and force ranges are
  `±[87, 87, 87, 87, 12, 12, 12] N·m`;
- `actuator8` remains the independent tendon position actuator for the gripper;
- the torque home keyframe commands zero arm torque and an open gripper.

This derivative is not Franka official code and does not reproduce FCI hardware
safety, internal gravity/friction compensation, torque sensing, or 1 kHz timing.
