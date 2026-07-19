from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments import PandaUTableEnv, load_config
from evaluation import evaluate_task_state
from perception import ColorDepthDetector, OverheadRGBDCamera, RGBDPerceptionProvider
from perception.image_io import (
    save_depth_millimeters_png,
    save_depth_preview_png,
    save_mask_png,
    save_rgb_png,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one overhead RGB-D frame.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def detection_dict(detection: object) -> dict[str, object]:
    return {
        "id": detection.detection_id,
        "success": detection.success,
        "pixel_count": detection.pixel_count,
        "center_pixel": detection.center_pixel,
        "position": detection.position,
        "confidence": detection.confidence,
        "failure_reason": detection.failure_reason,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config).with_modes(observation_source="perception")
    seed = config.seed if args.seed is None else args.seed
    env = PandaUTableEnv(config)
    provider = RGBDPerceptionProvider(
        OverheadRGBDCamera(env.model, config.camera),
        env.data,
        ColorDepthDetector(config.perception),
    )
    try:
        env.reset(seed=seed)
        estimate = provider.estimate()
        metrics = evaluate_task_state(env, estimate)
        frame = provider.last_frame
        object_detection = provider.last_object_detection
        target_detection = provider.last_target_detection
        if frame is None or object_detection is None or target_detection is None:
            raise RuntimeError("Perception provider did not retain capture diagnostics")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_rgb_png(args.output_dir / "rgb.png", frame.rgb)
        save_depth_millimeters_png(args.output_dir / "depth_mm.png", frame.depth)
        save_depth_preview_png(args.output_dir / "depth_preview.png", frame.depth)
        save_mask_png(args.output_dir / "object_mask.png", object_detection.mask)
        save_mask_png(args.output_dir / "target_mask.png", target_detection.mask)
        np.save(args.output_dir / "rgb.npy", frame.rgb)
        np.save(args.output_dir / "depth_m.npy", frame.depth)
        metadata = {
            "seed": seed,
            "camera_name": frame.camera_name,
            "resolution": [frame.width, frame.height],
            "simulation_time": frame.simulation_time,
            "depth_semantics": frame.depth_semantics,
            "intrinsics": asdict(frame.intrinsics),
            "camera_to_world": frame.extrinsics.camera_to_world.tolist(),
            "world_to_camera": frame.extrinsics.world_to_camera.tolist(),
            "estimate": asdict(estimate),
            "object_detection": detection_dict(object_detection),
            "target_detection": detection_dict(target_detection),
            "metrics": asdict(metrics),
            "files": {
                "rgb_png": "rgb.png",
                "depth_png": "depth_mm.png",
                "depth_preview_png": "depth_preview.png",
                "object_mask_png": "object_mask.png",
                "target_mask_png": "target_mask.png",
                "rgb_npy": "rgb.npy",
                "depth_m_npy": "depth_m.npy",
            },
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        provider.close()
        env.close()
    return 0 if estimate.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
