# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Isaac Sim Franka pick-and-place example in headless mode."
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Physics simulation device. Default: cpu",
    )

    parser.add_argument(
        "--ik-method",
        type=str,
        choices=[
            "singular-value-decomposition",
            "pseudoinverse",
            "transpose",
            "damped-least-squares",
        ],
        default="damped-least-squares",
        help="Differential inverse kinematics method.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=30000,
        help="Maximum simulation update steps before timeout. Default: 30000",
    )

    args, _ = parser.parse_known_args()

    if args.max_steps <= 0:
        parser.error("--max-steps must be greater than 0.")

    return args


args = parse_args()

# SimulationApp必须在大多数Isaac Sim和Omniverse模块之前创建。
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        # 不创建Isaac Sim图形界面。
        "headless": True,

        # Headless模式下关闭不必要的视口更新。
        "disable_viewport_updates": True,

        # 当前CPU为8核16线程，先限制为6个工作线程，
        # 为Windows系统保留一定响应能力。
        "limit_cpu_threads": 6,

        # 当前只有一块GPU，不启用多GPU工作流。
        "multi_gpu": False,
    }
)

# 以下模块必须在SimulationApp创建后导入。
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace


def main() -> int:
    """Run one Franka pick-and-place episode.

    Returns:
        0: Task completed successfully.
        2: Task timed out or simulation stopped unexpectedly.
    """
    print("=" * 72, flush=True)
    print("Starting Simple Franka Pick-and-Place Demo", flush=True)
    print(f"Physics device: {args.device}", flush=True)
    print(f"IK method: {args.ik_method}", flush=True)
    print(f"Maximum steps: {args.max_steps}", flush=True)
    print("GUI enabled: False", flush=True)
    print("=" * 72, flush=True)

    # 设置PhysX仿真设备。
    SimulationManager.set_physics_sim_device(args.device)

    # 让SimulationApp完成一次初始化更新。
    simulation_app.update()

    # 创建官方Franka抓取放置示例。
    pick_place = FrankaPickPlace()
    pick_place.setup_scene()

    timeline = omni.timeline.get_timeline_interface()

    # 启动物理仿真。
    timeline.play()
    simulation_app.update()

    reset_needed = True
    step_count = 0

    print("Starting pick-and-place execution", flush=True)

    while simulation_app.is_running():
        simulation_app.update()
        step_count += 1

        if SimulationManager.is_simulating():
            if reset_needed:
                print("Resetting pick-and-place system...", flush=True)
                pick_place.reset()
                reset_needed = False

            # 执行一个抓取放置控制步骤。
            pick_place.forward(args.ik_method)

            # 任务完成后立即退出主循环，避免继续占用资源。
            if pick_place.is_done():
                print("done picking and placing", flush=True)
                print(f"Completed simulation steps: {step_count}", flush=True)

                timeline.stop()
                simulation_app.update()

                return 0

        # 防止控制器异常时无限运行。
        if step_count >= args.max_steps:
            print(
                f"ERROR: Pick-and-place task timed out after "
                f"{step_count} simulation steps.",
                flush=True,
            )

            timeline.stop()
            simulation_app.update()

            return 2

    print(
        "ERROR: SimulationApp stopped before the pick-and-place task completed.",
        flush=True,
    )

    timeline.stop()
    return 2


if __name__ == "__main__":
    exit_code = 1

    try:
        exit_code = main()

    except KeyboardInterrupt:
        print("Execution interrupted by user.", flush=True)
        exit_code = 130

    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )
        exit_code = 1

    finally:
        print("Closing Isaac Sim...", flush=True)
        simulation_app.close()
        print(f"Isaac Sim closed. Exit code: {exit_code}", flush=True)

    raise SystemExit(exit_code)