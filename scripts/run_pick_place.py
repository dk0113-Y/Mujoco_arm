from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controllers import FixedDLSPickPlaceController
from environments import PandaUTableEnv, load_config
from evaluation import FailureReason
from perception import (
    ColorDepthDetector,
    OverheadRGBDCamera,
    PrivilegedStateProvider,
    RGBDPerceptionProvider,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one scripted Fixed-DLS Panda U-table pick/place episode."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "u_table.toml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pick-mode", choices=("fixed", "random"), default=None)
    parser.add_argument("--place-mode", choices=("fixed", "random"), default=None)
    parser.add_argument("--physics-mode", choices=("fixed", "random"), default=None)
    parser.add_argument(
        "--observation-source",
        choices=("privileged", "perception"),
        default=None,
    )
    viewer_group = parser.add_mutually_exclusive_group()
    viewer_group.add_argument("--viewer", action="store_true", dest="viewer")
    viewer_group.add_argument("--headless", action="store_false", dest="viewer")
    parser.set_defaults(viewer=None)
    return parser.parse_args()


def print_summary(result: object) -> None:
    data = result.to_dict()
    print("Panda U-table pick/place episode")
    print(f"  seed: {data['seed']}")
    print(
        "  modes: "
        f"pick={data['pick_mode']}, place={data['place_mode']}, "
        f"physics={data['physics_mode']}"
    )
    print(
        f"  regions: pick={data['pick_region']}, place={data['place_region']}"
    )
    print(f"  final stage: {data['final_stage']}")
    print(f"  simulation time: {data['simulation_time']:.3f} s")
    print(f"  success: {data['success']}")
    print(f"  failure_reason: {data['failure_reason']}")
    print(f"  final XY error: {data['final_xy_error']}")
    print(f"  final height error: {data['final_height_error']}")
    print(f"  collision count: {data['collision_count']}")
    print(f"  observation source: {data['observation_source']}")
    if data["observation_source"] == "perception":
        print(f"  perception success: {data['perception_success']}")
        print(f"  perception failure: {data['perception_failure_reason']}")
        print(f"  object estimate error: {data['object_position_error']}")
        print(f"  target estimate error: {data['target_position_error']}")
        print(f"  perception latency: {data['perception_latency_ms']} ms")
    if data["exception_message"]:
        print(f"  detail: {data['exception_message']}")
    print("Structured JSON result:")
    print(result.to_json())


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        effective_seed = config.seed if args.seed is None else args.seed
        config = config.with_modes(
            seed=effective_seed,
            pick_mode=args.pick_mode,
            place_mode=args.place_mode,
            physics_mode=args.physics_mode,
            viewer=args.viewer,
            observation_source=args.observation_source,
        )
        env = PandaUTableEnv(config)
        controller = FixedDLSPickPlaceController(config.controller)
        if config.observation.source == "perception":
            state_provider = RGBDPerceptionProvider(
                OverheadRGBDCamera(env.model, config.camera),
                env.data,
                ColorDepthDetector(config.perception),
            )
        else:
            state_provider = PrivilegedStateProvider(env)
        use_viewer = config.simulation.viewer
        try:
            if use_viewer:
                import mujoco.viewer

                last_wall_time = time.perf_counter()
                with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

                    def sync_viewer(current_env: PandaUTableEnv) -> bool:
                        nonlocal last_wall_time
                        viewer.sync()
                        target_period = (
                            current_env.model.opt.timestep
                            * current_env.config.simulation.frame_skip
                        )
                        elapsed = time.perf_counter() - last_wall_time
                        if elapsed < target_period:
                            time.sleep(target_period - elapsed)
                        last_wall_time = time.perf_counter()
                        return viewer.is_running()

                    result = controller.run_episode(
                        env,
                        seed=effective_seed,
                        state_provider=state_provider,
                        step_callback=sync_viewer,
                    )
            else:
                result = controller.run_episode(
                    env, seed=effective_seed, state_provider=state_provider
                )
        finally:
            state_provider.close()
            env.close()
    except Exception as exc:
        print(f"Program error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print_summary(result)
    if result.success:
        return 0
    if result.failure_reason == FailureReason.UNEXPECTED_EXCEPTION.value:
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
