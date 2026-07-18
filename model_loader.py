from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parent

PANDA_XML_PATH = (
    PROJECT_ROOT
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)

PANDA_ASSET_DIR = PANDA_XML_PATH.parent / "assets"


def load_panda_with_scene(scene_xml_path: Path) -> mujoco.MjModel:
    """加载官方 Panda 模型，并合并项目自定义场景。"""

    scene_xml_path = scene_xml_path.resolve()

    if not PANDA_XML_PATH.exists():
        raise FileNotFoundError(
            "找不到 Panda 模型。请检查 Git submodule 是否已经初始化：\n"
            "git submodule update --init --recursive\n"
            f"期望路径：{PANDA_XML_PATH}"
        )

    if not scene_xml_path.exists():
        raise FileNotFoundError(f"找不到场景文件：{scene_xml_path}")

    panda_tree = ET.parse(PANDA_XML_PATH)
    panda_root = panda_tree.getroot()

    # panda.xml 中的 meshdir 原本是相对路径 assets。
    # 因为后续从 XML 字符串编译，所以这里改成绝对路径。
    compiler = panda_root.find("compiler")
    if compiler is None:
        raise RuntimeError("Panda XML 中没有找到 compiler 节点")

    compiler.set("meshdir", PANDA_ASSET_DIR.resolve().as_posix())

    panda_worldbody = panda_root.find("worldbody")
    if panda_worldbody is None:
        raise RuntimeError("Panda XML 中没有找到 worldbody 节点")

    scene_tree = ET.parse(scene_xml_path)
    scene_root = scene_tree.getroot()
    scene_worldbody = scene_root.find("worldbody")

    if scene_worldbody is None:
        raise RuntimeError("自定义场景中没有找到 worldbody 节点")

    # 将桌面、立方体、灯光和目标区域加入 Panda 的 worldbody。
    for element in list(scene_worldbody):
        panda_worldbody.append(element)

    # 在两个指尖之间定义一个近似工具中心点 TCP。
    hand_body = panda_root.find(".//body[@name='hand']")
    if hand_body is None:
        raise RuntimeError("Panda XML 中没有找到 hand 刚体")

    ET.SubElement(
        hand_body,
        "site",
        {
            "name": "gripper_tcp",
            "type": "sphere",
            "pos": "0 0 0.103",
            "size": "0.008",
            "rgba": "1 0 0 1",
        },
    )

    model_xml = ET.tostring(
        panda_root,
        encoding="unicode",
    )

    return mujoco.MjModel.from_xml_string(model_xml)
PANDA_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)


def reset_panda_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """
    恢复完整场景的默认状态，并仅将 Panda 关节设置为官方 home 姿态。

    这样可以保留立方体等自由物体在 MJCF 中定义的初始位置。
    """

    # 恢复 model.qpos0，包括自由物体在 XML 中定义的位置和姿态。
    mujoco.mj_resetData(model, data)

    home_key_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )

    if home_key_id < 0:
        raise RuntimeError("模型中没有找到 home keyframe")

    # 只复制 Panda 自身的 9 个单自由度关节。
    for joint_name in PANDA_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            raise RuntimeError(f"模型中没有找到 Panda 关节：{joint_name}")

        qpos_address = int(model.jnt_qposadr[joint_id])

        data.qpos[qpos_address] = model.key_qpos[
            home_key_id,
            qpos_address,
        ]

    # 使用 home keyframe 中的 8 个控制输入。
    data.ctrl[:] = model.key_ctrl[home_key_id]

    # 清空所有关节和自由物体的初始速度。
    data.qvel[:] = 0.0

    # 根据新的 qpos 更新刚体、site 和 geom 的世界坐标。
    mujoco.mj_forward(model, data)