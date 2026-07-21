from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import traceback
from typing import Any, Sequence

import numpy as np

from environments import EnvConfig, PandaUTableEnv, load_config
from evaluation import EpisodeResult, FailureReason, evaluate_task_state
from evaluation.production_metrics import (
    build_production_metrics,
    derive_episode_protocol_fields,
)
from evaluation.protocol import ProtocolConfig, validate_baseline_compatibility
from perception.types import PerceptionMetrics

from .manifest import repository_metadata, runtime_metadata, sha256_file
from .methods import (
    BENCHMARK_NAME,
    BENCHMARK_SCHEMA_VERSION,
    MethodSpec,
    assert_static_fairness,
    resolve_methods,
)
from .pairing import (
    DEFAULT_FINGERPRINT_ATOL,
    EpisodeFingerprint,
    PairMismatchError,
    classify_outcome,
    validate_pair,
)
from .schemas import (
    FAILURE_COUNT_FIELDS,
    PAIRED_RESULT_FIELDS,
    episode_fieldnames,
    write_csv,
    write_json,
)
from .seed_io import load_seeds
from .summary import build_summary, failure_counts_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkRunError(RuntimeError):
    """Raised after traceable outputs are written for an invalid benchmark run."""


@dataclass(frozen=True)
class BenchmarkRunResult:
    output_dir: Path
    requested_pairs: int
    completed_pairs: int
    invalid_pairs: int
    program_errors: int
    exit_code: int


@dataclass
class _EpisodeExecution:
    pair_id: str
    seed: int
    method: MethodSpec
    execution_index: int
    result: EpisodeResult | None = None
    fingerprint: EpisodeFingerprint | None = None
    initial_robot_state: tuple[float, ...] | None = None
    external_state_metrics: PerceptionMetrics | None = None
    pair_valid: bool = False
    program_error: str | None = None
    diagnostic_recording: Any | None = None


class _RecordingProvider:
    """Record robot-only reset state while returning provider estimates unchanged."""

    def __init__(self, provider: Any, env: PandaUTableEnv) -> None:
        self.provider = provider
        self.env = env
        self.source = provider.source
        self.initial_robot_state: tuple[float, ...] | None = None
        self.initial_metrics: PerceptionMetrics | None = None

    def estimate(self):
        if self.initial_robot_state is None:
            observation = self.env.observation()
            values = np.concatenate(
                (
                    np.asarray(observation["arm_joint_positions"], dtype=float),
                    np.asarray(observation["arm_joint_velocities"], dtype=float),
                    np.asarray(observation["finger_positions"], dtype=float),
                    np.asarray(observation["tcp_position"], dtype=float),
                    np.asarray(observation["tcp_orientation"], dtype=float).reshape(-1),
                )
            )
            if not np.all(np.isfinite(values)):
                raise RuntimeError("Initial robot state contains NaN or Inf")
            self.initial_robot_state = tuple(float(value) for value in values)
        estimate = self.provider.estimate()
        if self.initial_metrics is None:
            # This independent label calculation is recorded only by the
            # benchmark wrapper; the estimate returned to control is unchanged.
            self.initial_metrics = evaluate_task_state(self.env, estimate)
        return estimate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    resolved = output_dir.resolve()
    repository_outputs = (PROJECT_ROOT / "outputs").resolve()
    if resolved == PROJECT_ROOT or resolved in PROJECT_ROOT.parents:
        raise ValueError(f"Unsafe benchmark output directory: {resolved}")
    if resolved == repository_outputs:
        raise ValueError(
            f"Use a dedicated subdirectory instead of the outputs root: {resolved}"
        )
    if PROJECT_ROOT in resolved.parents and repository_outputs not in resolved.parents:
        raise ValueError(
            "Benchmark outputs inside the repository must stay under "
            f"{repository_outputs}: {resolved}"
        )
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Benchmark output path is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {resolved}; pass --overwrite explicitly"
            )
        for child in resolved.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    resolved.mkdir(parents=True, exist_ok=True)


def _logger_for(output_dir: Path) -> tuple[logging.Logger, logging.Handler]:
    logger = logging.getLogger(f"benchmark0.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(
        output_dir / "run.log", mode="w", encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _effective_config(config_path: Path) -> tuple[EnvConfig, dict[str, Any]]:
    base = load_config(config_path)
    config = base.with_modes(
        controller_type="sensor_event_b1",
        observation_source="perception",
        viewer=False,
    )
    overrides: dict[str, Any] = {}
    if base.controller.type != config.controller.type:
        overrides["controller.type"] = config.controller.type
    if base.observation.source != config.observation.source:
        overrides["observation.source"] = config.observation.source
    if base.simulation.viewer != config.simulation.viewer:
        overrides["simulation.viewer"] = config.simulation.viewer
    return config, overrides


def _result_program_error(result: EpisodeResult) -> str | None:
    if result.failure_reason == FailureReason.UNEXPECTED_EXCEPTION.value:
        return result.exception_message or "unexpected_exception"
    if result.exception_message and result.exception_message.startswith(
        "External-state provider failed:"
    ):
        return result.exception_message
    return None


def _execute_episode(
    method: MethodSpec,
    config: EnvConfig,
    seed: int,
    pair_id: str,
    execution_index: int,
    logger: logging.Logger,
    diagnostic_factory: Any | None = None,
) -> _EpisodeExecution:
    execution = _EpisodeExecution(
        pair_id=pair_id,
        seed=seed,
        method=method,
        execution_index=execution_index,
    )
    env: PandaUTableEnv | None = None
    raw_provider: Any | None = None
    recording_provider: _RecordingProvider | None = None
    cleanup_errors: list[str] = []
    try:
        env = PandaUTableEnv(config)
        controller = method.controller_class(config.controller, config.b1)
        if type(controller) is not method.controller_class:
            raise RuntimeError("Method constructed an unexpected controller class")
        if controller.controller_config != config.controller:
            raise RuntimeError("Method changed ControllerConfig")
        if controller.config != config.b1:
            raise RuntimeError("Method changed B1Config")
        raw_provider = method.provider_factory(env, config)
        if type(raw_provider) is not method.provider_type:
            raise RuntimeError(
                f"{method.method_id} constructed {type(raw_provider).__name__}, "
                f"expected {method.provider_type.__name__}"
            )
        expected_provider_source = (
            "oracle" if method.external_state_source == "oracle" else "perception"
        )
        if raw_provider.source != expected_provider_source:
            raise RuntimeError(
                f"{method.method_id} provider source is {raw_provider.source!r}, "
                f"expected {expected_provider_source!r}"
            )
        if method.external_state_source == "oracle" and hasattr(
            raw_provider, "camera"
        ):
            raise RuntimeError("Oracle provider must not construct a camera or Renderer")
        recording_provider = _RecordingProvider(raw_provider, env)
        if diagnostic_factory is not None:
            execution.diagnostic_recording = diagnostic_factory.start_episode(
                env=env,
                method=method,
                seed=seed,
                pair_id=pair_id,
                execution_index=execution_index,
            )
        logger.info(
            "episode_start pair=%s seed=%s method=%s execution_index=%s",
            pair_id,
            seed,
            method.method_id,
            execution_index,
        )
        run_arguments: dict[str, Any] = {
            "seed": seed,
            "state_provider": recording_provider,
        }
        if execution.diagnostic_recording is not None:
            run_arguments["diagnostic_observer"] = (
                execution.diagnostic_recording.observe
            )
        execution.result = controller.run_episode(env, **run_arguments)
        execution.initial_robot_state = recording_provider.initial_robot_state
        execution.external_state_metrics = recording_provider.initial_metrics
        execution.fingerprint = EpisodeFingerprint.from_episode_result(
            execution.result
        )
        execution.program_error = _result_program_error(execution.result)
        if execution.diagnostic_recording is not None:
            execution.diagnostic_recording.finish(
                result=execution.result,
                fingerprint=execution.fingerprint,
                initial_robot_state=execution.initial_robot_state,
                external_state_metrics=execution.external_state_metrics,
            )
        logger.info(
            "episode_end pair=%s method=%s stage=%s controller_success=%s "
            "ground_truth_success=%s failure=%s",
            pair_id,
            method.method_id,
            execution.result.final_stage,
            execution.result.controller_reported_success,
            execution.result.privileged_ground_truth_success,
            execution.result.failure_reason,
        )
    except Exception:
        execution.program_error = traceback.format_exc()
        logger.exception(
            "episode_program_error pair=%s seed=%s method=%s",
            pair_id,
            seed,
            method.method_id,
        )
    finally:
        if execution.diagnostic_recording is not None:
            try:
                execution.diagnostic_recording.close()
            except Exception:
                cleanup_errors.append(
                    "diagnostic recorder close failed:\n" + traceback.format_exc()
                )
        if raw_provider is not None:
            try:
                raw_provider.close()
            except Exception:
                cleanup_errors.append(
                    "provider close failed:\n" + traceback.format_exc()
                )
        if env is not None:
            try:
                env.close()
            except Exception:
                cleanup_errors.append("environment close failed:\n" + traceback.format_exc())
        if cleanup_errors:
            cleanup_detail = "\n".join(cleanup_errors)
            execution.program_error = "\n".join(
                value
                for value in (execution.program_error, cleanup_detail)
                if value
            )
            logger.error(
                "episode_cleanup_error pair=%s method=%s detail=%s",
                pair_id,
                method.method_id,
                cleanup_detail,
            )
    return execution


def _robot_states_match(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
    *,
    atol: float,
) -> bool:
    if left is None or right is None:
        return False
    return len(left) == len(right) and bool(
        np.allclose(left, right, rtol=0.0, atol=atol)
    )


def _validate_execution_pair(
    oracle: _EpisodeExecution,
    vision: _EpisodeExecution,
    *,
    atol: float,
) -> str | None:
    if oracle.program_error or vision.program_error:
        return "One or more paired episodes ended with a program error"
    if oracle.seed != vision.seed:
        return f"Pair seed mismatch: oracle={oracle.seed}, vision={vision.seed}"
    if oracle.fingerprint is None or vision.fingerprint is None:
        return "One or more paired episodes did not produce a fingerprint"
    try:
        validate_pair(oracle.fingerprint, vision.fingerprint, atol=atol)
    except PairMismatchError as exc:
        return str(exc)
    if not _robot_states_match(
        oracle.initial_robot_state, vision.initial_robot_state, atol=atol
    ):
        return "Paired methods did not start from the same robot state"
    return None


def _episode_row(
    execution: _EpisodeExecution,
    *,
    protocol: ProtocolConfig | None = None,
    split_name: str | None = None,
    config_sha256: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "benchmark_name": BENCHMARK_NAME,
        "pair_id": execution.pair_id,
        "method_id": execution.method.method_id,
        "external_state_source": execution.method.external_state_source,
        "execution_index": execution.execution_index,
        "episode_fingerprint": (
            None if execution.fingerprint is None else execution.fingerprint.digest
        ),
        "pair_valid": execution.pair_valid,
        "program_error": execution.program_error,
    }
    if execution.result is not None:
        result_data = execution.result.to_dict()
        key_errors = result_data.pop("key_errors") or {}
        stage_durations = result_data.pop("stage_durations") or {}
        row.update(result_data)
        row.update(
            {f"key_error.{name}": value for name, value in key_errors.items()}
        )
        row.update(
            {
                f"stage_duration.{name}": value
                for name, value in stage_durations.items()
            }
        )
        if execution.external_state_metrics is not None:
            row["object_position_error"] = (
                execution.external_state_metrics.object_3d_error
            )
            row["target_position_error"] = (
                execution.external_state_metrics.target_3d_error
            )
    if protocol is not None:
        row.update(
            {
                "split_name": split_name,
                "config_sha256": config_sha256,
                "code_commit": code_commit,
            }
        )
        row.update(derive_episode_protocol_fields(row, protocol))
    return row


def _paired_row(
    pair_id: str,
    seed: int,
    oracle: _EpisodeExecution,
    vision: _EpisodeExecution,
    pair_error: str | None,
) -> dict[str, Any]:
    pair_valid = pair_error is None
    program_error = bool(oracle.program_error or vision.program_error)
    oracle_result = oracle.result
    vision_result = vision.result
    outcome = classify_outcome(
        None
        if oracle_result is None
        else oracle_result.privileged_ground_truth_success,
        None
        if vision_result is None
        else vision_result.privileged_ground_truth_success,
        pair_valid=pair_valid,
        program_error=program_error,
    )
    return {
        "pair_id": pair_id,
        "seed": seed,
        "pair_valid": pair_valid,
        "pair_error": pair_error,
        "fingerprint": (
            None if oracle.fingerprint is None else oracle.fingerprint.digest
        ),
        "oracle_ground_truth_success": (
            None
            if oracle_result is None
            else oracle_result.privileged_ground_truth_success
        ),
        "vision_ground_truth_success": (
            None
            if vision_result is None
            else vision_result.privileged_ground_truth_success
        ),
        "oracle_controller_reported_success": (
            None
            if oracle_result is None
            else oracle_result.controller_reported_success
        ),
        "vision_controller_reported_success": (
            None
            if vision_result is None
            else vision_result.controller_reported_success
        ),
        "oracle_failure_reason": (
            None if oracle_result is None else oracle_result.failure_reason
        ),
        "vision_failure_reason": (
            None if vision_result is None else vision_result.failure_reason
        ),
        "oracle_final_stage": (
            None if oracle_result is None else oracle_result.final_stage
        ),
        "vision_final_stage": (
            None if vision_result is None else vision_result.final_stage
        ),
        "oracle_simulation_time": (
            None if oracle_result is None else oracle_result.simulation_time
        ),
        "vision_simulation_time": (
            None if vision_result is None else vision_result.simulation_time
        ),
        "oracle_collision_count": (
            None if oracle_result is None else oracle_result.collision_count
        ),
        "vision_collision_count": (
            None if vision_result is None else vision_result.collision_count
        ),
        "vision_object_position_error": (
            vision_result.object_position_error
            if vision.external_state_metrics is None and vision_result is not None
            else (
                None
                if vision.external_state_metrics is None
                else vision.external_state_metrics.object_3d_error
            )
        ),
        "vision_target_position_error": (
            vision_result.target_position_error
            if vision.external_state_metrics is None and vision_result is not None
            else (
                None
                if vision.external_state_metrics is None
                else vision.external_state_metrics.target_3d_error
            )
        ),
        "outcome_category": outcome,
    }


def run_benchmark(
    *,
    config_path: str | Path,
    method_ids: Sequence[str],
    seeds_file: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    continue_on_error: bool = False,
    require_clean_git: bool = False,
    command: Sequence[str] | None = None,
    fingerprint_atol: float = DEFAULT_FINGERPRINT_ATOL,
    protocol: ProtocolConfig | None = None,
    split_name: str | None = None,
    calibration_run: bool = False,
    baseline_frozen: bool = False,
) -> BenchmarkRunResult:
    config_path = Path(config_path).expanduser().resolve()
    seeds_path = Path(seeds_file).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    methods = resolve_methods(list(method_ids))
    seeds = load_seeds(seeds_path)
    config, overrides = _effective_config(config_path)
    if protocol is not None:
        if split_name not in protocol.splits:
            raise ValueError(f"Unknown protocol split name: {split_name!r}")
        expected_seed_path = protocol.splits[str(split_name)].path.resolve()
        if seeds_path != expected_seed_path:
            raise ValueError(
                f"Seed file does not match protocol split {split_name}: {seeds_path}"
            )
        validate_baseline_compatibility(protocol, config)
        if baseline_frozen:
            raise ValueError(
                "Evaluation Protocol v1 calibration/benchmark tooling does not "
                "declare a baseline frozen"
            )
    assert_static_fairness(methods, config)
    repository = repository_metadata(PROJECT_ROOT)
    if require_clean_git and repository["git_dirty"]:
        raise BenchmarkRunError(
            "Repository is dirty and --require-clean-git was requested"
        )
    _prepare_output_dir(output_path, overwrite)
    shutil.copyfile(config_path, output_path / "config_snapshot.toml")
    if protocol is not None:
        shutil.copyfile(protocol.path, output_path / "protocol_snapshot.toml")
    logger, log_handler = _logger_for(output_path)
    start_time = _utc_now()
    config_digest = sha256_file(config_path)
    manifest: dict[str, Any] = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "start_time": start_time,
        "end_time": None,
        "command": list(command or []),
        **repository,
        **runtime_metadata(),
        "config_path": str(config_path),
        "config_sha256": config_digest,
        "config_snapshot_path": str(output_path / "config_snapshot.toml"),
        "effective_overrides": overrides,
        "seed_file_path": str(seeds_path),
        "seed_file_sha256": sha256_file(seeds_path),
        "methods": [method.method_id for method in methods],
        "method_execution_order": [method.method_id for method in methods],
        "total_requested_pairs": len(seeds),
        "completed_pairs": 0,
        "invalid_pairs": 0,
        "unhandled_errors": 0,
        "unhandled_error_details": [],
        "pilot": protocol is None,
    }
    if protocol is not None:
        manifest.update(
            {
                "protocol_id": protocol.protocol_id,
                "protocol_version": protocol.protocol_version,
                "metrics_schema_version": protocol.metrics_schema_version,
                "protocol_config_path": str(protocol.path),
                "protocol_config_sha256": protocol.sha256,
                "protocol_snapshot_path": str(
                    output_path / "protocol_snapshot.toml"
                ),
                "split_id": protocol.split_id,
                "split_name": split_name,
                "calibration_run": calibration_run,
                "baseline_frozen": baseline_frozen,
                "automatic_parameter_search": False,
            }
        )
    executions: list[_EpisodeExecution] = []
    pair_rows: list[dict[str, Any]] = []
    fatal_error: str | None = None
    execution_index = 0
    try:
        write_json(
            output_path / "seeds.json",
            {
                "seeds": seeds,
                "seed_count": len(seeds),
                "duplicates_present": False,
                "pilot": protocol is None,
            },
        )
        logger.info(
            "benchmark_start pairs=%s methods=%s config=%s seeds=%s",
            len(seeds),
            [method.method_id for method in methods],
            config_path,
            seeds_path,
        )
        for pair_index, seed in enumerate(seeds):
            pair_id = f"pair_{pair_index:04d}_seed_{seed}"
            pair_executions: dict[str, _EpisodeExecution] = {}
            for method in methods:
                execution = _execute_episode(
                    method,
                    config,
                    seed,
                    pair_id,
                    execution_index,
                    logger,
                )
                execution_index += 1
                executions.append(execution)
                pair_executions[method.method_id] = execution

            oracle = pair_executions["b0_oracle"]
            vision = pair_executions["b1_vision"]
            pair_error = _validate_execution_pair(
                oracle, vision, atol=fingerprint_atol
            )
            pair_valid = pair_error is None
            oracle.pair_valid = pair_valid
            vision.pair_valid = pair_valid
            row = _paired_row(pair_id, seed, oracle, vision, pair_error)
            pair_rows.append(row)
            if pair_valid:
                manifest["completed_pairs"] += 1
            elif row["outcome_category"] == "invalid_pair":
                manifest["invalid_pairs"] += 1
            if oracle.program_error or vision.program_error:
                details = [
                    {
                        "pair_id": pair_id,
                        "method_id": execution.method.method_id,
                        "error": execution.program_error,
                    }
                    for execution in (oracle, vision)
                    if execution.program_error
                ]
                manifest["unhandled_errors"] += len(details)
                manifest["unhandled_error_details"].extend(details)
            if pair_error is not None:
                logger.error("pair_rejected pair=%s detail=%s", pair_id, pair_error)
                if not continue_on_error:
                    fatal_error = pair_error
                    break
    except Exception:
        fatal_error = traceback.format_exc()
        manifest["unhandled_errors"] += 1
        manifest["unhandled_error_details"].append(
            {"pair_id": None, "method_id": None, "error": fatal_error}
        )
        logger.exception("benchmark_program_error")
    finally:
        try:
            episode_rows = [
                _episode_row(
                    execution,
                    protocol=protocol,
                    split_name=split_name,
                    config_sha256=config_digest,
                    code_commit=str(repository.get("git_commit") or ""),
                )
                for execution in executions
            ]
            write_csv(
                output_path / "episodes.csv",
                episode_rows,
                episode_fieldnames(episode_rows),
            )
            write_csv(
                output_path / "paired_results.csv",
                pair_rows,
                PAIRED_RESULT_FIELDS,
            )
            failure_rows = failure_counts_rows(
                episode_rows, [method.method_id for method in methods]
            )
            write_csv(
                output_path / "failure_counts.csv",
                failure_rows,
                FAILURE_COUNT_FIELDS,
            )
            summary = build_summary(
                episode_rows,
                pair_rows,
                [method.method_id for method in methods],
                requested_episode_count=len(seeds),
            )
            write_json(output_path / "summary.json", summary)
            if protocol is not None:
                production_metrics = build_production_metrics(
                    episode_rows,
                    pair_rows,
                    protocol=protocol,
                )
                production_metrics["methods"] = {
                    method.method_id: build_production_metrics(
                        [
                            row
                            for row in episode_rows
                            if row.get("method_id") == method.method_id
                        ],
                        protocol=protocol,
                    )
                    for method in methods
                }
                write_json(
                    output_path / "production_metrics.json",
                    production_metrics,
                )
        except Exception:
            output_error = traceback.format_exc()
            manifest["unhandled_errors"] += 1
            manifest["unhandled_error_details"].append(
                {
                    "pair_id": None,
                    "method_id": None,
                    "error": "Output finalization failed:\n" + output_error,
                }
            )
            fatal_error = fatal_error or (
                "Benchmark output finalization failed:\n" + output_error
            )
            logger.exception("benchmark_output_error")
        finally:
            manifest["end_time"] = _utc_now()
            try:
                write_json(output_path / "run_manifest.json", manifest)
            except Exception:
                manifest_error = traceback.format_exc()
                fatal_error = fatal_error or (
                    "Run manifest finalization failed:\n" + manifest_error
                )
                logger.exception("benchmark_manifest_error")
            try:
                logger.info(
                    "benchmark_end completed_pairs=%s invalid_pairs=%s errors=%s",
                    manifest["completed_pairs"],
                    manifest["invalid_pairs"],
                    manifest["unhandled_errors"],
                )
            finally:
                logger.removeHandler(log_handler)
                log_handler.close()

    exit_code = (
        1
        if fatal_error or manifest["unhandled_errors"] or manifest["invalid_pairs"]
        else 0
    )
    result = BenchmarkRunResult(
        output_dir=output_path,
        requested_pairs=len(seeds),
        completed_pairs=int(manifest["completed_pairs"]),
        invalid_pairs=int(manifest["invalid_pairs"]),
        program_errors=int(manifest["unhandled_errors"]),
        exit_code=exit_code,
    )
    if fatal_error:
        raise BenchmarkRunError(
            f"Benchmark-0 stopped after writing traceable outputs: {fatal_error}"
        )
    return result
