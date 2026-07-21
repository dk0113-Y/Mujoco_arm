from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import BenchmarkRunError, FORMAL_METHOD_IDS, run_benchmark
from evaluation.protocol import load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Benchmark-0 episodes with B0-Oracle and B1-Vision. "
            "Both methods use the same sensor_event_b1 controller."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "u_table.toml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Optional Evaluation Protocol config that adds versioned metrics.",
    )
    parser.add_argument(
        "--split-name",
        choices=("calibration", "development", "held_out_test", "calibration_smoke"),
        help="Required with --protocol; seed path must match this registered split.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=FORMAL_METHOD_IDS,
        default=list(FORMAL_METHOD_IDS),
    )
    parser.add_argument(
        "--seeds-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly clear an existing non-empty output directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later pairs after invalid pairs or program errors.",
    )
    parser.add_argument(
        "--require-clean-git",
        action="store_true",
        help="Refuse to start when the repository has tracked or untracked changes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.protocol is None) != (args.split_name is None):
        print("--protocol and --split-name must be supplied together", file=sys.stderr)
        return 2
    protocol = None if args.protocol is None else load_protocol(args.protocol)
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    command = [sys.executable, str(Path(__file__).resolve()), *command_arguments]
    try:
        result = run_benchmark(
            config_path=args.config,
            method_ids=args.methods,
            seeds_file=args.seeds_file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
            require_clean_git=args.require_clean_git,
            command=command,
            protocol=protocol,
            split_name=args.split_name,
        )
    except Exception as exc:
        print(f"Benchmark-0 error: {exc}", file=sys.stderr)
        return 1
    print(
        "Benchmark-0 finished: "
        f"completed_pairs={result.completed_pairs}/{result.requested_pairs}, "
        f"invalid_pairs={result.invalid_pairs}, "
        f"program_errors={result.program_errors}, "
        f"output={result.output_dir}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
