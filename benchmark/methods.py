from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from controllers import SensorEventPickPlaceController
from environments import EnvConfig, PandaUTableEnv
from evaluation.perception_evaluator import build_episode_result
from perception import (
    ColorDepthDetector,
    OracleExternalStateProvider,
    OverheadRGBDCamera,
    RGBDPerceptionProvider,
    TaskStateProvider,
)


BENCHMARK_NAME = "benchmark0_oracle_paired_eval"
BENCHMARK_SCHEMA_VERSION = "1.0.0"


ProviderFactory = Callable[[PandaUTableEnv, EnvConfig], TaskStateProvider]


def _build_oracle_provider(
    env: PandaUTableEnv, _config: EnvConfig
) -> OracleExternalStateProvider:
    return OracleExternalStateProvider(env)


def _build_vision_provider(
    env: PandaUTableEnv, config: EnvConfig
) -> RGBDPerceptionProvider:
    camera = OverheadRGBDCamera(env.model, config.camera)
    try:
        return RGBDPerceptionProvider(
            camera,
            env.data,
            ColorDepthDetector(config.perception),
        )
    except Exception:
        camera.close()
        raise


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    external_state_source: str
    provider_type: type
    provider_factory: ProviderFactory
    controller_type: str = "sensor_event_b1"
    controller_class: type = SensorEventPickPlaceController
    ground_truth_evaluator: Callable[..., object] = build_episode_result


METHOD_SPECS: dict[str, MethodSpec] = {
    "b0_oracle": MethodSpec(
        method_id="b0_oracle",
        external_state_source="oracle",
        provider_type=OracleExternalStateProvider,
        provider_factory=_build_oracle_provider,
    ),
    "b1_vision": MethodSpec(
        method_id="b1_vision",
        external_state_source="vision",
        provider_type=RGBDPerceptionProvider,
        provider_factory=_build_vision_provider,
    ),
}

FORMAL_METHOD_IDS = tuple(METHOD_SPECS)


def resolve_methods(method_ids: list[str] | tuple[str, ...]) -> tuple[MethodSpec, ...]:
    if not method_ids:
        raise ValueError("At least one Benchmark-0 method is required")
    duplicates = sorted(
        method_id for method_id in set(method_ids) if method_ids.count(method_id) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate Benchmark-0 methods: {duplicates}")
    unknown = sorted(set(method_ids) - set(FORMAL_METHOD_IDS))
    if unknown:
        raise ValueError(
            f"Unknown Benchmark-0 methods {unknown}; allowed: {list(FORMAL_METHOD_IDS)}"
        )
    missing = sorted(set(FORMAL_METHOD_IDS) - set(method_ids))
    if missing:
        raise ValueError(
            "Benchmark-0 is a paired evaluation and requires both formal methods; "
            f"missing: {missing}"
        )
    return tuple(METHOD_SPECS[method_id] for method_id in method_ids)


def assert_static_fairness(
    methods: tuple[MethodSpec, ...], config: EnvConfig
) -> None:
    if config.controller.type != "sensor_event_b1":
        raise ValueError("Benchmark-0 requires controller.type='sensor_event_b1'")
    if config.observation.source != "perception":
        raise ValueError(
            "Benchmark-0 keeps the environment truth-isolated with "
            "observation.source='perception'"
        )
    if {method.method_id for method in methods} != set(FORMAL_METHOD_IDS):
        raise ValueError("Benchmark-0 fairness checks require the complete method pair")
    if any(
        method.controller_class is not SensorEventPickPlaceController
        for method in methods
    ):
        raise ValueError("Both methods must use SensorEventPickPlaceController")
    if any(method.controller_type != "sensor_event_b1" for method in methods):
        raise ValueError("Both methods must record controller_type='sensor_event_b1'")
    if any(
        method.ground_truth_evaluator is not build_episode_result
        for method in methods
    ):
        raise ValueError("Both methods must use the same ground-truth evaluator")
    if METHOD_SPECS["b0_oracle"].provider_type is not OracleExternalStateProvider:
        raise ValueError("b0_oracle must use OracleExternalStateProvider")
    if METHOD_SPECS["b1_vision"].provider_type is not RGBDPerceptionProvider:
        raise ValueError("b1_vision must use RGBDPerceptionProvider")
