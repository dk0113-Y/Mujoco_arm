from pathlib import Path

import mujoco


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print(f"模型路径: {MODEL_PATH}")
    print(f"nq  位置变量数量: {model.nq}")
    print(f"nv  速度变量数量: {model.nv}")
    print(f"nu  控制输入数量: {model.nu}")
    print(f"nbody 刚体数量: {model.nbody}")
    print(f"njnt  关节数量: {model.njnt}")
    print(f"ngeom 几何体数量: {model.ngeom}")
    print()

    print("关节列表:")
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )
        print(f"  joint[{joint_id}]: {name}")

    print()
    print("执行器列表:")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
        print(
            f"  actuator[{actuator_id}]: {name}, "
            f"ctrlrange=[{ctrl_min:.4f}, {ctrl_max:.4f}]"
        )

    mujoco.mj_forward(model, data)

    print()
    print("初始 qpos:")
    print(data.qpos)

    print()
    print("初始 ctrl:")
    print(data.ctrl)


if __name__ == "__main__":
    main()