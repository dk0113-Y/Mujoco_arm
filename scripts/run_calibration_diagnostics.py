from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.calibration_diagnostics import (
    REQUIRED_METHODS,
    run_calibration_diagnostics,
)
from evaluation.protocol import load_protocol


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed four-seed B1 Round 0.5 passive diagnostic replay. "
            "This is not a formal split, never writes production metrics, never "
            "changes parameters, and never freezes B1."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--diagnostic-seeds-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-zero-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Must be exactly: b0_oracle b1_vision",
    )
    parser.add_argument("--diagnostics-enabled", action="store_true")
    parser.add_argument("--visualization-artifacts-enabled", action="store_true")
    parser.add_argument(
        "--require-traceable-source",
        action="store_true",
        help=(
            "Require a content-addressed source snapshot. This is the traceable "
            "equivalent used when reviewed, uncommitted diagnostic code is present."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command_arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if tuple(args.methods) != REQUIRED_METHODS:
            raise ValueError(
                f"--methods must be exactly {' '.join(REQUIRED_METHODS)} in that order"
            )
        protocol = load_protocol(args.protocol)
        result = run_calibration_diagnostics(
            protocol=protocol,
            config_path=args.baseline_config,
            seeds_file=args.diagnostic_seeds_file,
            output_dir=args.output_dir,
            round_zero_dir=args.round_zero_dir,
            method_ids=args.methods,
            diagnostics_enabled=args.diagnostics_enabled,
            visualization_enabled=args.visualization_artifacts_enabled,
            require_traceable_source=args.require_traceable_source,
            command=[sys.executable, str(Path(__file__).resolve()), *command_arguments],
        )
    except Exception as exc:
        print(f"Round 0.5 diagnostic error: {exc}", file=sys.stderr)
        return 1
    print(
        "Round 0.5 diagnostic replay finished: "
        f"completed_pairs={result.completed_pairs}/{result.requested_pairs}, "
        f"invalid_pairs={result.invalid_pairs}, program_errors={result.program_errors}, "
        "diagnostic_only=true, production_metrics=false, "
        f"output={result.output_dir}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
