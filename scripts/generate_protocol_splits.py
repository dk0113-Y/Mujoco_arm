from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.seed_io import load_seeds
from evaluation.protocol import load_protocol
from evaluation.split_analysis import (
    SPLIT_ORDER,
    generate_split_plan,
    seed_file_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or reproduce-check Evaluation Protocol seed splits."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol, validate_splits=False)
    seeds_by_split, manifest = generate_split_plan(protocol)
    if args.write:
        for name in SPLIT_ORDER:
            path = protocol.splits[name].path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(seed_file_text(seeds_by_split[name]))
        protocol.split_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        protocol.split_manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        for name in SPLIT_ORDER:
            actual = load_seeds(protocol.splits[name].path)
            if actual != seeds_by_split[name]:
                raise ValueError(f"{name} split differs from deterministic generation")
        actual_manifest = json.loads(
            protocol.split_manifest_path.read_text(encoding="utf-8")
        )
        if actual_manifest != manifest:
            raise ValueError("split_manifest.json differs from deterministic generation")
    load_protocol(args.protocol, validate_splits=True)
    print(
        "Evaluation Protocol splits valid: "
        + ", ".join(f"{name}={len(seeds_by_split[name])}" for name in SPLIT_ORDER)
        + f", manifest_sha256={manifest['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
