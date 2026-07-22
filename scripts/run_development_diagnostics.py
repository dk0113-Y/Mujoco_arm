from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.development_diagnostics import REQUIRED_METHODS, run_development_diagnostics
from evaluation.protocol import load_protocol


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the fixed ten Development D0.5 mechanism-diagnostic seeds. "
            "This diagnostic-only run never writes production metrics, changes B1, "
            "or reads Held-out Test."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--diagnostics-enabled", action="store_true")
    parser.add_argument("--visualization-artifacts-enabled", action="store_true")
    parser.add_argument("--require-traceable-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if tuple(args.methods) != REQUIRED_METHODS:
            raise ValueError("--methods must be exactly b0_oracle b1_vision in that order")
        protocol = load_protocol(args.protocol, validate_splits=False)
        result = run_development_diagnostics(
            protocol=protocol,
            config_path=args.frozen_config,
            development_run_dir=args.development_run_dir,
            candidate_file=args.candidate_file,
            output_dir=args.output_dir,
            method_ids=args.methods,
            diagnostics_enabled=args.diagnostics_enabled,
            visualization_enabled=args.visualization_artifacts_enabled,
            require_traceable_source=args.require_traceable_source,
            command=[sys.executable, str(Path(__file__).resolve()), *arguments],
        )
    except Exception as exc:
        print(f"Development D0.5 diagnostic error: {exc}", file=sys.stderr)
        return 1
    print(
        "Development D0.5 diagnostic replay finished: "
        f"completed_pairs={result.completed_pairs}/{result.requested_pairs}, "
        f"invalid_pairs={result.invalid_pairs}, program_errors={result.program_errors}, "
        "diagnostic_only=true, production_metrics=false, "
        f"output={result.output_dir}"
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
