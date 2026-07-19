from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments import PandaUTableEnv, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a reset Panda U-table scene without running the controller."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "u_table.toml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Validate/reset and print the sample without opening a GUI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    config = config.with_modes(seed=seed, viewer=not args.headless)
    env = PandaUTableEnv(config)
    _, info = env.reset(seed=seed)
    print("Reset sample:")
    print(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False))
    print("Table geometry:")
    for name, region in env.workspace.regions.items():
        full_size = tuple(2.0 * value for value in region.half_extents_xy)
        print(
            f"  {name}: center_xy={region.center_xy}, "
            f"size_xy={full_size}, top_z={region.top_z}"
        )

    if not args.headless:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
