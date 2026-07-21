from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import FORMAL_METHOD_IDS, run_benchmark
from evaluation.protocol import load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a recorded B1 calibration episode set. The paired Oracle run remains "
            "diagnostic; this tool never searches or rewrites parameters and never freezes B1."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--seeds-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        calibration_raw = protocol.raw["calibration"]
        baseline_path = args.baseline_config
        if baseline_path is None:
            baseline_path = PROJECT_ROOT / str(
                calibration_raw["baseline_template_path"]
            )
        seeds_path = (
            protocol.splits["calibration"].path
            if args.seeds_file is None
            else args.seeds_file.expanduser().resolve()
        )
        allowed_splits = {
            protocol.splits["calibration"].path.resolve(): "calibration",
            protocol.splits["calibration_smoke"].path.resolve(): "calibration_smoke",
        }
        split_name = allowed_splits.get(seeds_path.resolve())
        if split_name is None:
            raise ValueError(
                "Calibration runner accepts only the registered calibration or "
                "calibration_smoke split; Development and Held-out Test are forbidden"
            )
        command_arguments = list(sys.argv[1:] if argv is None else argv)
        result = run_benchmark(
            config_path=baseline_path,
            method_ids=FORMAL_METHOD_IDS,
            seeds_file=seeds_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            continue_on_error=args.continue_on_error,
            require_clean_git=args.require_clean_git,
            command=[sys.executable, str(Path(__file__).resolve()), *command_arguments],
            protocol=protocol,
            split_name=split_name,
            calibration_run=True,
            baseline_frozen=False,
        )
    except Exception as exc:
        print(f"Calibration run error: {exc}", file=sys.stderr)
        return 1
    print(
        "Calibration recording finished: "
        f"completed_pairs={result.completed_pairs}/{result.requested_pairs}, "
        "calibration_run=true, baseline_frozen=false, "
        f"output={result.output_dir}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
