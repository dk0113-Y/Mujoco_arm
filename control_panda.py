import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    home_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_KEY,
        "home",
    )
    if home_id < 0:
        raise RuntimeError("没有找到 home keyframe")

    # 使用官方定义的 home 姿态，而不是全零状态
    mujoco.mj_resetDataKeyframe(model, data, home_id)

    home_ctrl = data.ctrl.copy()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 16.0:
            step_start = time.perf_counter()

            # 每一步先恢复基础目标，避免控制量累积
            data.ctrl[:] = home_ctrl

            # actuator1 在 home 附近小范围往复运动
            data.ctrl[0] = home_ctrl[0] + 0.35 * np.sin(
                2.0 * np.pi * 0.15 * data.time
            )

            # 每 4 秒切换一次夹爪状态
            phase = int(data.time // 4.0) % 2
            data.ctrl[7] = 255.0 if phase == 0 else 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()