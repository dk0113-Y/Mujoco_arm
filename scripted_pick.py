from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from model_loader import load_panda_with_scene, reset_panda_home
from move_above_cube import get_object_id, smoothstep, solve_pose_ik


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_ROOT / "scenes" / "panda_pick_scene.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))

SETTLE_TIME = 1.0
MAX_SIMULATION_TIME = 35.0
MOTION_HOLD_TIME = 0.6

ABOVE_CUBE_OFFSET = 0.18
GRASP_Z_OFFSET = 0.005
LIFT_OFFSET = 0.20
CUBE_HALF_SIZE = 0.025

WAYPOINT_TOLERANCE = 0.012
MIN_LIFT_HEIGHT = 0.08
PLACE_XY_TOLERANCE = 0.06
PLACE_Z_TOLERANCE = 0.03

GRIPPER_OPEN_CTRL = 255.0
GRIPPER_CLOSE_CTRL = 0.0


@dataclass
class Action:
    kind: str
    name: str
    duration: float
    target_position: np.ndarray | None = None
    gripper_ctrl: float | None = None


@dataclass
class MotionPlan:
    start_time: float
    duration: float
    start_ctrl: np.ndarray
    target_ctrl: np.ndarray
    target_position: np.ndarray


def make_motion_plan(
    action: Action,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    tcp_site_id: int,
    arm_actuator_ids: np.ndarray,
    arm_qpos_addresses: np.ndarray,
    arm_dof_addresses: np.ndarray,
    arm_joint_ranges: np.ndarray,
    arm_ctrl_ranges: np.ndarray,
    target_rotation: np.ndarray,
) -> MotionPlan:
    if action.target_position is None:
        raise ValueError("motion 动作缺少 target_position")

    joint_target, position_error, rotation_error, iterations = (
        solve_pose_ik(
            model=model,
            initial_qpos=data.qpos.copy(),
            tcp_site_id=tcp_site_id,
            arm_qpos_addresses=arm_qpos_addresses,
            arm_dof_addresses=arm_dof_addresses,
            arm_joint_ranges=arm_joint_ranges,
            target_position=action.target_position,
            target_rotation=target_rotation,
        )
    )

    joint_target = np.clip(
        joint_target,
        arm_ctrl_ranges[:, 0],
        arm_ctrl_ranges[:, 1],
    )

    print(
        f"{action.name}：IK 收敛，"
        f"迭代={iterations}，"
        f"位置误差={position_error:.6f} m，"
        f"姿态误差={rotation_error:.6f} rad"
    )
    print(
        f"{action.name}："
        f"目标 TCP={action.target_position}"
    )

    return MotionPlan(
        start_time=float(data.time),
        duration=action.duration,
        start_ctrl=data.ctrl[arm_actuator_ids].copy(),
        target_ctrl=joint_target,
        target_position=action.target_position.copy(),
    )


def apply_motion(
    plan: MotionPlan,
    data: mujoco.MjData,
    arm_actuator_ids: np.ndarray,
) -> float:
    elapsed = float(
        data.time - plan.start_time
    )

    interpolation = smoothstep(
        elapsed / plan.duration
    )

    data.ctrl[arm_actuator_ids] = (
        plan.start_ctrl
        + interpolation
        * (
            plan.target_ctrl
            - plan.start_ctrl
        )
    )

    return elapsed


def main() -> None:
    model = load_panda_with_scene(
        SCENE_PATH
    )

    data = mujoco.MjData(
        model
    )

    tcp_site_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripper_tcp",
    )

    place_target_site_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "place_target",
    )

    cube_body_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cube",
    )

    gripper_actuator_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "actuator8",
    )

    arm_joint_ids = np.array(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    arm_actuator_ids = np.array(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in ARM_ACTUATOR_NAMES
        ],
        dtype=int,
    )

    finger_joint_ids = np.array(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "finger_joint1",
            ),
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                "finger_joint2",
            ),
        ],
        dtype=int,
    )

    arm_qpos_addresses = np.array(
        [
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
            for joint_id in arm_joint_ids
        ],
        dtype=int,
    )

    arm_dof_addresses = np.array(
        [
            int(
                model.jnt_dofadr[
                    joint_id
                ]
            )
            for joint_id in arm_joint_ids
        ],
        dtype=int,
    )

    finger_qpos_addresses = np.array(
        [
            int(
                model.jnt_qposadr[
                    joint_id
                ]
            )
            for joint_id in finger_joint_ids
        ],
        dtype=int,
    )

    arm_joint_ranges = np.asarray(
        model.jnt_range[
            arm_joint_ids
        ],
        dtype=float,
    )

    arm_ctrl_ranges = np.asarray(
        model.actuator_ctrlrange[
            arm_actuator_ids
        ],
        dtype=float,
    )

    reset_panda_home(
        model,
        data,
    )

    data.ctrl[
        gripper_actuator_id
    ] = GRIPPER_OPEN_CTRL

    actions: list[Action] | None = None
    action_index = 0
    action_start_time = 0.0

    motion_plan: MotionPlan | None = None

    target_rotation: np.ndarray | None = None
    cube_initial_position: np.ndarray | None = None
    place_target_position: np.ndarray | None = None

    last_report_time = -1.0

    print(
        "阶段 1：等待立方体稳定在桌面。"
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        while (
            viewer.is_running()
            and data.time
            < MAX_SIMULATION_TIME
        ):
            step_start = time.perf_counter()

            if (
                actions is None
                and data.time >= SETTLE_TIME
            ):
                cube_initial_position = (
                    data.xpos[
                        cube_body_id
                    ].copy()
                )

                place_target_position = (
                    data.site_xpos[
                        place_target_site_id
                    ].copy()
                )

                target_rotation = (
                    data.site_xmat[
                        tcp_site_id
                    ]
                    .reshape(3, 3)
                    .copy()
                )

                above_cube_position = (
                    cube_initial_position
                    + np.array(
                        [
                            0.0,
                            0.0,
                            ABOVE_CUBE_OFFSET,
                        ]
                    )
                )

                grasp_position = (
                    cube_initial_position
                    + np.array(
                        [
                            0.0,
                            0.0,
                            GRASP_Z_OFFSET,
                        ]
                    )
                )

                lift_position = (
                    cube_initial_position
                    + np.array(
                        [
                            0.0,
                            0.0,
                            LIFT_OFFSET,
                        ]
                    )
                )

                above_target_position = np.array(
                    [
                        place_target_position[0],
                        place_target_position[1],
                        lift_position[2],
                    ],
                    dtype=float,
                )

                place_position = np.array(
                    [
                        place_target_position[0],
                        place_target_position[1],
                        place_target_position[2]
                        + CUBE_HALF_SIZE,
                    ],
                    dtype=float,
                )

                actions = [
                    Action(
                        kind="motion",
                        name="阶段 2：移动到立方体上方",
                        duration=4.0,
                        target_position=(
                            above_cube_position
                        ),
                    ),
                    Action(
                        kind="motion",
                        name="阶段 3：下降到抓取位置",
                        duration=3.0,
                        target_position=(
                            grasp_position
                        ),
                    ),
                    Action(
                        kind="gripper",
                        name="阶段 4：闭合夹爪",
                        duration=1.5,
                        gripper_ctrl=(
                            GRIPPER_CLOSE_CTRL
                        ),
                    ),
                    Action(
                        kind="motion",
                        name="阶段 5：抬升立方体",
                        duration=3.0,
                        target_position=(
                            lift_position
                        ),
                    ),
                    Action(
                        kind="motion",
                        name="阶段 6：移动到目标区域上方",
                        duration=4.0,
                        target_position=(
                            above_target_position
                        ),
                    ),
                    Action(
                        kind="motion",
                        name="阶段 7：下降到放置位置",
                        duration=3.0,
                        target_position=(
                            place_position
                        ),
                    ),
                    Action(
                        kind="gripper",
                        name="阶段 8：打开夹爪",
                        duration=1.5,
                        gripper_ctrl=(
                            GRIPPER_OPEN_CTRL
                        ),
                    ),
                    Action(
                        kind="motion",
                        name="阶段 9：从目标区域撤离",
                        duration=3.0,
                        target_position=(
                            above_target_position
                        ),
                    ),
                ]

                print(
                    f"立方体稳定位置："
                    f"{cube_initial_position}"
                )

                print(
                    f"放置目标位置："
                    f"{place_target_position}"
                )

                action_start_time = float(
                    data.time
                )

            if (
                actions is not None
                and action_index
                < len(actions)
            ):
                action = actions[
                    action_index
                ]

                if action.kind == "motion":
                    if motion_plan is None:
                        if target_rotation is None:
                            raise RuntimeError(
                                "目标姿态尚未初始化。"
                            )

                        motion_plan = make_motion_plan(
                            action,
                            model=model,
                            data=data,
                            tcp_site_id=tcp_site_id,
                            arm_actuator_ids=(
                                arm_actuator_ids
                            ),
                            arm_qpos_addresses=(
                                arm_qpos_addresses
                            ),
                            arm_dof_addresses=(
                                arm_dof_addresses
                            ),
                            arm_joint_ranges=(
                                arm_joint_ranges
                            ),
                            arm_ctrl_ranges=(
                                arm_ctrl_ranges
                            ),
                            target_rotation=(
                                target_rotation
                            ),
                        )

                    elapsed = apply_motion(
                        motion_plan,
                        data,
                        arm_actuator_ids,
                    )

                    if (
                        elapsed
                        >= action.duration
                        + MOTION_HOLD_TIME
                    ):
                        waypoint_error = float(
                            np.linalg.norm(
                                motion_plan.target_position
                                - data.site_xpos[
                                    tcp_site_id
                                ]
                            )
                        )

                        print(
                            f"{action.name}完成，"
                            f"实际 TCP="
                            f"{data.site_xpos[tcp_site_id].copy()}，"
                            f"位置误差="
                            f"{waypoint_error:.4f} m"
                        )

                        if (
                            waypoint_error
                            > WAYPOINT_TOLERANCE
                        ):
                            raise RuntimeError(
                                f"{action.name}"
                                "未达到误差要求。"
                            )

                        if "抬升" in action.name:
                            if (
                                cube_initial_position
                                is None
                            ):
                                raise RuntimeError(
                                    "缺少立方体初始位置。"
                                )

                            height_gain = float(
                                data.xpos[
                                    cube_body_id
                                ][2]
                                - cube_initial_position[2]
                            )

                            print(
                                "抬升后立方体高度增量："
                                f"{height_gain:.4f} m"
                            )

                            if (
                                height_gain
                                < MIN_LIFT_HEIGHT
                            ):
                                raise RuntimeError(
                                    "抓取失败："
                                    "立方体未随夹爪抬升。"
                                )

                        motion_plan = None
                        action_index += 1
                        action_start_time = float(
                            data.time
                        )

                elif action.kind == "gripper":
                    if (
                        action.gripper_ctrl
                        is None
                    ):
                        raise ValueError(
                            "gripper 动作缺少 "
                            "gripper_ctrl"
                        )

                    data.ctrl[
                        gripper_actuator_id
                    ] = action.gripper_ctrl

                    if (
                        data.time
                        - action_start_time
                        >= action.duration
                    ):
                        print(
                            f"{action.name}完成，"
                            f"夹爪关节位置="
                            f"{data.qpos[finger_qpos_addresses].copy()}"
                        )

                        action_index += 1
                        action_start_time = float(
                            data.time
                        )

            if (
                data.time
                - last_report_time
                >= 0.5
            ):
                if actions is None:
                    stage_name = "等待初始化"
                elif action_index >= len(actions):
                    stage_name = "完成"
                else:
                    stage_name = actions[
                        action_index
                    ].name

                print(
                    f"t={data.time:.2f} s，"
                    f"状态={stage_name}，"
                    f"TCP="
                    f"{data.site_xpos[tcp_site_id].copy()}，"
                    f"cube="
                    f"{data.xpos[cube_body_id].copy()}，"
                    f"finger="
                    f"{data.qpos[finger_qpos_addresses].copy()}，"
                    f"gripper_ctrl="
                    f"{data.ctrl[gripper_actuator_id]:.1f}"
                )

                last_report_time = float(
                    data.time
                )

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            if (
                actions is not None
                and action_index
                >= len(actions)
            ):
                print(
                    "阶段 10："
                    "全部动作执行完成。"
                )
                break

            elapsed_step = (
                time.perf_counter()
                - step_start
            )

            remaining = (
                model.opt.timestep
                - elapsed_step
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

    if (
        actions is None
        or action_index < len(actions)
    ):
        raise RuntimeError(
            "程序未完成全部抓取与放置流程。"
        )

    if (
        cube_initial_position is None
        or place_target_position is None
    ):
        raise RuntimeError(
            "没有成功初始化抓取场景。"
        )

    final_cube_position = (
        data.xpos[
            cube_body_id
        ].copy()
    )

    final_tcp_position = (
        data.site_xpos[
            tcp_site_id
        ].copy()
    )

    final_finger_positions = (
        data.qpos[
            finger_qpos_addresses
        ].copy()
    )

    target_xy_error = float(
        np.linalg.norm(
            final_cube_position[:2]
            - place_target_position[:2]
        )
    )

    final_height_error = abs(
        float(
            final_cube_position[2]
            - cube_initial_position[2]
        )
    )

    print(
        f"最终 TCP 位置："
        f"{final_tcp_position}"
    )

    print(
        f"最终立方体位置："
        f"{final_cube_position}"
    )

    print(
        f"最终夹爪关节位置："
        f"{final_finger_positions}"
    )

    print(
        f"目标区域 XY 误差："
        f"{target_xy_error:.4f} m"
    )

    print(
        f"立方体最终高度误差："
        f"{final_height_error:.4f} m"
    )

    if (
        target_xy_error
        > PLACE_XY_TOLERANCE
    ):
        raise RuntimeError(
            "放置失败："
            "立方体未进入目标区域。"
        )

    if (
        final_height_error
        > PLACE_Z_TOLERANCE
    ):
        raise RuntimeError(
            "放置失败："
            "立方体未稳定落回桌面。"
        )

    print(
        "验收通过："
        "立方体已被抓取、搬运"
        "并放置到目标区域。"
    )


if __name__ == "__main__":
    main()