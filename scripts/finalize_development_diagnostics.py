from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.development_diagnostics import finalize_mechanism_review


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize the human-reviewed D0.5 M-* mechanism conclusion."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = finalize_mechanism_review(args.output_dir, args.review_file)
    except Exception as exc:
        print(f"Development D0.5 finalization error: {exc}", file=sys.stderr)
        return 1
    print(
        "Development D0.5 structured review finalized: "
        f"decision={evidence['decision']}, output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
