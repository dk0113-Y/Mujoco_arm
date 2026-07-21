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
    collect_task_samples,
    distribution_summary,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect protocol task distributions using environment reset metadata only."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=(*SPLIT_ORDER, "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    names = SPLIT_ORDER if args.split == "all" else (args.split,)
    report: dict[str, object] = {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "split_id": protocol.split_id,
        "renderer_created": False,
        "controller_outcomes_used": False,
        "splits": {},
    }
    combined = []
    for name in names:
        samples = collect_task_samples(protocol, load_seeds(protocol.splits[name].path))
        combined.extend(samples)
        report["splits"][name] = distribution_summary(samples)  # type: ignore[index]
    report["combined"] = distribution_summary(combined)
    text = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    illegal = report["combined"]["illegal_sample_count"]  # type: ignore[index]
    return 0 if illegal == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
