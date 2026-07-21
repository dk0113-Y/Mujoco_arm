from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.calibration_diagnostics import finalize_structured_review


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize a Round 0.5 report from an explicit structured evidence "
            "review. This does not run episodes or modify any B1 parameter."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = finalize_structured_review(args.output_dir, args.review_file)
    except Exception as exc:
        print(f"Structured review finalization error: {exc}", file=sys.stderr)
        return 1
    print(
        "Structured review finalized: "
        f"decision={summary['manual_assessment']['round_1_decision']}, "
        "b1_parameters_modified=false, round_1_executed=false, b1_frozen=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
