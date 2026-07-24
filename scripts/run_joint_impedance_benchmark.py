from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from control_benchmarks.outputs import strict_json_value
from control_benchmarks.runner import SUPPORTED_EXPERIMENTS, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated Panda JI-Baseline v1 benchmark"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=SUPPORTED_EXPERIMENTS,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only known benchmark files in the output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    summary = run_benchmark(
        arguments.config,
        experiment=arguments.experiment,
        output=arguments.output,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            strict_json_value(summary),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
