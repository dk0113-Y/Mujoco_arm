from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from model_loader import (
    load_panda_with_scene,
    reset_panda_home,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SCENE_PATH = PROJECT_ROOT / "scenes" / "panda_pick_scene.xml"


def get_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """按名称获取 MuJoCo 对象 ID，不存在时直接报错。"""
    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    if object_id < 0:
        raise RuntimeError(f"模型中没有找到对象：{name}")

    return object_id


def main() -> None:
    model = load_panda_with_scene(SCENE_PATH)
    data = mujoco.MjData(model)

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

    cube_joint_id = get_object_id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "cube_free_joint",
    )

    # 查找立方体自由关节在 qpos 和 qvel 中的起始位置。
    cube_qpos_address = int(
        model.jnt_qposadr[cube_joint_id]
    )

    cube_qvel_address = int(
        model.jnt_dofadr[cube_joint_id]
    )

    # 恢复完整场景默认状态，并仅将 Panda 设置为 home 姿态。
    # 这样不会覆盖立方体在 XML 中设置的初始位置。
    reset_panda_home(model, data)

    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")

    print(
        "立方体默认 qpos:",
        model.qpos0[
            cube_qpos_address:cube_qpos_address + 7
        ],
    )

    print(
        "立方体当前 qpos:",
        data.qpos[
            cube_qpos_address:cube_qpos_address + 7
        ],
    )

    print(
        "立方体当前 qvel:",
        data.qvel[
            cube_qvel_address:cube_qvel_address + 6
        ],
    )

    print(
        "TCP 初始位置:",
        data.site_xpos[tcp_site_id].copy(),
    )

    print(
        "立方体初始位置:",
        data.xpos[cube_body_id].copy(),
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            remaining = model.opt.timestep - elapsed

            if remaining > 0:
                time.sleep(remaining)

    final_tcp_position = (
        data.site_xpos[tcp_site_id].copy()
    )

    final_cube_position = (
        data.xpos[cube_body_id].copy()
    )

    print(f"TCP 最终位置: {final_tcp_position}")
    print(f"立方体最终位置: {final_cube_position}")
    print(
        f"立方体最终高度: "
        f"{final_cube_position[2]:.4f} m"
    )

    tcp_cube_distance = np.linalg.norm(
        final_tcp_position - final_cube_position
    )

    print(
        f"TCP 与立方体距离: "
        f"{tcp_cube_distance:.4f} m"
    )


if __name__ == "__main__":
    main()