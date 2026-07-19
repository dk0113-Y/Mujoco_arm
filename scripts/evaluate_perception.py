from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments import PandaUTableEnv, load_config
from evaluation import evaluate_task_state
from perception import ColorDepthDetector, OverheadRGBDCamera, RGBDPerceptionProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RGB-D task-state estimates against privileged labels."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--pick-mode", choices=("fixed", "random"), default="random")
    parser.add_argument("--place-mode", choices=("fixed", "random"), default="random")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config).with_modes(
        observation_source="perception",
        pick_mode=args.pick_mode,
        place_mode=args.place_mode,
    )
    env = PandaUTableEnv(config)
    provider = RGBDPerceptionProvider(
        OverheadRGBDCamera(env.model, config.camera),
        env.data,
        ColorDepthDetector(config.perception),
    )
    records: list[dict[str, object]] = []
    try:
        for seed in args.seeds:
            env.reset(seed=seed)
            estimate = provider.estimate()
            metrics = evaluate_task_state(env, estimate)
            episode = env.current_episode
            records.append(
                {
                    "seed": seed,
                    "pick_region": None if episode is None else episode.pick_region,
                    "place_region": None if episode is None else episode.place_region,
                    "detection_success": estimate.valid,
                    "confidence": estimate.confidence,
                    "failure_reason": estimate.failure_reason,
                    "inference_time_ms": estimate.latency_ms,
                    **asdict(metrics),
                }
            )
    finally:
        provider.close()
        env.close()

    successful = [record for record in records if record["detection_success"]]
    object_errors = [
        float(record["object_3d_error"])
        for record in successful
        if record["object_3d_error"] is not None
    ]
    target_errors = [
        float(record["target_3d_error"])
        for record in successful
        if record["target_3d_error"] is not None
    ]
    summary = {
        "camera_name": config.camera.name,
        "resolution": [config.camera.width, config.camera.height],
        "sample_modes": {"pick": args.pick_mode, "place": args.place_mode},
        "episodes": len(records),
        "successful_detections": len(successful),
        "detection_rate": len(successful) / len(records),
        "mean_object_3d_error": (
            None if not object_errors else statistics.fmean(object_errors)
        ),
        "mean_target_3d_error": (
            None if not target_errors else statistics.fmean(target_errors)
        ),
        "mean_inference_time_ms": statistics.fmean(
            float(record["inference_time_ms"]) for record in records
        ),
        "records": records,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
