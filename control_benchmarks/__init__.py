"""Isolated joint-torque control benchmark utilities."""

from .config import ControlBenchmarkConfig, load_control_config


def run_benchmark(*args, **kwargs):
    from .runner import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)


__all__ = ["ControlBenchmarkConfig", "load_control_config", "run_benchmark"]
