from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from model_loader import load_panda_with_scene, reset_panda_home


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_ROOT / "scenes" / "panda_pick_scene.xml"

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATOR_NAMES = tuple(f"actuator{i}" for i in range(1, 8))

# 等待方块落到桌面。
SETTLE_TIME = 1.0

# 机械臂从 home 姿态移动到目标姿态所用时间。
MOVE_DURATION = 4.0

# 到达目标后保持时间。
HOLD_DURATION = 2.0

# 整个程序最大仿真时间。
MAX_SIMULATION_TIME = 10.0

# TCP 位于方块中心上方 18 cm。
TARGET_HEIGHT_OFFSET = 0.18

# 最终允许的 TCP 位置误差。
POSITION_TOLERANCE = 0.01

# IK 中允许的姿态误差。
ORIENTATION_TOLERANCE = 0.03

# IK 参数。
IK_MAX_ITERATIONS = 500
IK_DAMPING = 0.05
IK_STEP_GAIN = 0.7
IK_MAX_JOINT_STEP = 0.08

# 姿态误差相对于位置误差的权重。
ORIENTATION_WEIGHT = 0.30

# Panda 夹爪完全打开的控制输入。
GRIPPER_OPEN_CTRL = 255.0


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """根据名称查找 MuJoCo 对象 ID。"""
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(
            f"模型中没有找到对象：{name}"
        )

    return object_id


def limit_vector_norm(
    vector: np.ndarray,
    max_norm: float,
) -> np.ndarray:
    """限制向量二范数，同时保持方向不变。"""
    norm = float(np.linalg.norm(vector))

    if norm == 0.0 or norm <= max_norm:
        return vector

    return vector * (max_norm / norm)


def rotation_error_world(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    """
    计算当前姿态到目标姿态的小角度误差。

    current_rotation 和 target_rotation 均为 3×3 旋转矩阵。
    返回值表达在世界坐标系中。
    """
    error = np.zeros(3, dtype=float)

    for axis_index in range(3):
        error += np.cross(
            current_rotation[:, axis_index],
            target_rotation[:, axis_index],
        )

    return 0.5 * error


def solve_pose_ik(
    model: mujoco.MjModel,
    initial_qpos: np.ndarray,
    tcp_site_id: int,
    arm_qpos_addresses: np.ndarray,
    arm_dof_addresses: np.ndarray,
    arm_joint_ranges: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
) -> tuple[np.ndarray, float, float, int]:
    """
    在独立 mjData 中求解 TCP 目标位姿。

    该函数只进行运动学迭代：
    - 不推进真实仿真时间；
    - 不修改真实仿真状态；
    - 同时约束 TCP 位置和姿态。
    """
    ik_data = mujoco.MjData(model)

    ik_data.qpos[:] = initial_qpos
    ik_data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        ik_data,
    )

    jacobian_position = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    jacobian_rotation = np.zeros(
        (3, model.nv),
        dtype=float,
    )

    position_error_norm = float("inf")
    orientation_error_norm = float("inf")

    for iteration in range(
        1,
        IK_MAX_ITERATIONS + 1,
    ):
        mujoco.mj_forward(
            model,
            ik_data,
        )

        current_position = (
            ik_data.site_xpos[
                tcp_site_id
            ].copy()
        )

        current_rotation = (
            ik_data.site_xmat[
                tcp_site_id
            ]
            .reshape(3, 3)
            .copy()
        )

        position_error = (
            target_position
            - current_position
        )

        orientation_error = (
            rotation_error_world(
                current_rotation,
                target_rotation,
            )
        )

        position_error_norm = float(
            np.linalg.norm(
                position_error
            )
        )

        orientation_error_norm = float(
            np.linalg.norm(
                orientation_error
            )
        )

        if (
            position_error_norm <= 0.002
            and orientation_error_norm
            <= ORIENTATION_TOLERANCE
        ):
            solved_joint_position = (
                ik_data.qpos[
                    arm_qpos_addresses
                ].copy()
            )

            return (
                solved_joint_position,
                position_error_norm,
                orientation_error_norm,
                iteration,
            )

        mujoco.mj_jacSite(
            model,
            ik_data,
            jacobian_position,
            jacobian_rotation,
            tcp_site_id,
        )

        # 只选择 Panda 七个机械臂关节对应的 Jacobian 列。
        position_jacobian = (
            jacobian_position[
                :,
                arm_dof_addresses,
            ]
        )

        rotation_jacobian = (
            jacobian_rotation[
                :,
                arm_dof_addresses,
            ]
        )

        # 组合平移和旋转任务。
        task_jacobian = np.vstack(
            (
                position_jacobian,
                ORIENTATION_WEIGHT
                * rotation_jacobian,
            )
        )

        task_error = np.concatenate(
            (
                position_error,
                ORIENTATION_WEIGHT
                * orientation_error,
            )
        )

        # 阻尼最小二乘：
        #
        # Δq = Jᵀ (J Jᵀ + λ²I)⁻¹ e
        regularized_matrix = (
            task_jacobian
            @ task_jacobian.T
            + (IK_DAMPING**2)
            * np.eye(6)
        )

        joint_step = (
            task_jacobian.T
            @ np.linalg.solve(
                regularized_matrix,
                task_error,
            )
        )

        joint_step = (
            IK_STEP_GAIN
            * joint_step
        )

        joint_step = limit_vector_norm(
            joint_step,
            IK_MAX_JOINT_STEP,
        )

        next_joint_position = (
            ik_data.qpos[
                arm_qpos_addresses
            ]
            + joint_step
        )

        next_joint_position = np.clip(
            next_joint_position,
            arm_joint_ranges[:, 0],
            arm_joint_ranges[:, 1],
        )

        ik_data.qpos[
            arm_qpos_addresses
        ] = next_joint_position

    raise RuntimeError(
        "逆运动学未收敛："
        f"位置误差="
        f"{position_error_norm:.4f} m，"
        f"姿态误差="
        f"{orientation_error_norm:.4f} rad"
    )


def smoothstep(alpha: float) -> float:
    """
    三次平滑插值函数。

    起点和终点速度均为零，避免机械臂突然启动或停止。
    """
    alpha = float(
        np.clip(
            alpha,
            0.0,
            1.0,
        )
    )

    return (
        alpha
        * alpha
        * (3.0 - 2.0 * alpha)
    )


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

    cube_body_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "cube",
    )

    arm_joint_ids = np.array(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            for joint_name
            in ARM_JOINT_NAMES
        ],
        dtype=int,
    )

    arm_actuator_ids = np.array(
        [
            get_object_id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            for actuator_name
            in ARM_ACTUATOR_NAMES
        ],
        dtype=int,
    )

    gripper_actuator_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "actuator8",
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
            for joint_id
            in arm_joint_ids
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
            for joint_id
            in arm_joint_ids
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
            for joint_id
            in finger_joint_ids
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

    start_arm_ctrl: (
        np.ndarray | None
    ) = None

    target_arm_ctrl: (
        np.ndarray | None
    ) = None

    target_position: (
        np.ndarray | None
    ) = None

    move_start_time: (
        float | None
    ) = None

    last_report_time = -1.0
    completed = False

    print(
        "阶段 1：等待立方体落到桌面。"
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
            step_start = (
                time.perf_counter()
            )

            # 每一个物理步都明确保持夹爪打开。
            data.ctrl[
                gripper_actuator_id
            ] = GRIPPER_OPEN_CTRL

            if (
                target_arm_ctrl is None
                and data.time
                >= SETTLE_TIME
            ):
                cube_position = (
                    data.xpos[
                        cube_body_id
                    ].copy()
                )

                current_tcp_position = (
                    data.site_xpos[
                        tcp_site_id
                    ].copy()
                )

                target_position = (
                    cube_position
                    + np.array(
                        [
                            0.0,
                            0.0,
                            TARGET_HEIGHT_OFFSET,
                        ],
                        dtype=float,
                    )
                )

                # 保持 home 姿态下的夹爪方向。
                # 机械臂移动时不允许腕部随意翻转。
                target_rotation = (
                    data.site_xmat[
                        tcp_site_id
                    ]
                    .reshape(3, 3)
                    .copy()
                )

                print(
                    f"立方体稳定位置："
                    f"{cube_position}"
                )

                print(
                    f"当前 TCP 位置："
                    f"{current_tcp_position}"
                )

                print(
                    f"目标 TCP 位置："
                    f"{target_position}"
                )

                print(
                    "阶段 2：求解同时约束"
                    "位置和姿态的 IK。"
                )

                (
                    solved_joint_target,
                    ik_position_error,
                    ik_orientation_error,
                    ik_iterations,
                ) = solve_pose_ik(
                    model=model,
                    initial_qpos=(
                        data.qpos.copy()
                    ),
                    tcp_site_id=(
                        tcp_site_id
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
                    target_position=(
                        target_position
                    ),
                    target_rotation=(
                        target_rotation
                    ),
                )

                start_arm_ctrl = (
                    data.ctrl[
                        arm_actuator_ids
                    ].copy()
                )

                target_arm_ctrl = (
                    np.clip(
                        solved_joint_target,
                        arm_ctrl_ranges[:, 0],
                        arm_ctrl_ranges[:, 1],
                    )
                )

                move_start_time = (
                    data.time
                )

                print(
                    "IK 已收敛："
                    f"迭代={ik_iterations}，"
                    f"位置误差="
                    f"{ik_position_error:.6f} m，"
                    f"姿态误差="
                    f"{ik_orientation_error:.6f} rad"
                )

                print(
                    "阶段 3：平滑移动到"
                    "固定关节目标。"
                )

            if (
                target_arm_ctrl
                is not None
                and start_arm_ctrl
                is not None
                and move_start_time
                is not None
            ):
                elapsed_move_time = (
                    data.time
                    - move_start_time
                )

                interpolation = smoothstep(
                    elapsed_move_time
                    / MOVE_DURATION
                )

                data.ctrl[
                    arm_actuator_ids
                ] = (
                    start_arm_ctrl
                    + interpolation
                    * (
                        target_arm_ctrl
                        - start_arm_ctrl
                    )
                )

                if (
                    target_position
                    is not None
                    and data.time
                    - last_report_time
                    >= 0.5
                ):
                    current_tcp_position = (
                        data.site_xpos[
                            tcp_site_id
                        ].copy()
                    )

                    current_error = float(
                        np.linalg.norm(
                            target_position
                            - current_tcp_position
                        )
                    )

                    print(
                        f"t={data.time:.2f} s，"
                        f"TCP="
                        f"{current_tcp_position}，"
                        f"位置误差="
                        f"{current_error:.4f} m，"
                        f"夹爪 ctrl="
                        f"{data.ctrl[gripper_actuator_id]:.1f}"
                    )

                    last_report_time = (
                        data.time
                    )

                if (
                    elapsed_move_time
                    >= MOVE_DURATION
                    + HOLD_DURATION
                ):
                    completed = True
                    break

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            elapsed = (
                time.perf_counter()
                - step_start
            )

            remaining = (
                model.opt.timestep
                - elapsed
            )

            if remaining > 0:
                time.sleep(
                    remaining
                )

    if target_position is None:
        raise RuntimeError(
            "仿真在生成目标位置前结束。"
        )

    final_tcp_position = (
        data.site_xpos[
            tcp_site_id
        ].copy()
    )

    final_cube_position = (
        data.xpos[
            cube_body_id
        ].copy()
    )

    final_finger_positions = (
        data.qpos[
            finger_qpos_addresses
        ].copy()
    )

    final_error = float(
        np.linalg.norm(
            target_position
            - final_tcp_position
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
        f"目标 TCP 位置："
        f"{target_position}"
    )

    print(
        f"最终位置误差："
        f"{final_error:.4f} m"
    )

    print(
        f"最终夹爪关节位置："
        f"{final_finger_positions}"
    )

    print(
        f"最终夹爪控制量："
        f"{data.ctrl[gripper_actuator_id]:.1f}"
    )

    if not completed:
        raise RuntimeError(
            "程序未完成预定移动和保持阶段。"
        )

    if (
        final_error
        > POSITION_TOLERANCE
    ):
        raise RuntimeError(
            "TCP 未在允许误差内"
            "到达目标位置。"
        )

    print(
        "验收通过：TCP 已稳定到达"
        "立方体正上方。"
    )


if __name__ == "__main__":
    main()