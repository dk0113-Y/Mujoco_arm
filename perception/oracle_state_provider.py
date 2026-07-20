from __future__ import annotations

import time

from environments.panda_u_table_env import PandaUTableEnv

from .types import TaskStateEstimate


class OracleExternalStateProvider:
    """Benchmark-only zero-error external-state provider.

    The provider exposes only the current object position, current target
    position, and simulation timestamp through the same estimate contract used
    by vision.  It deliberately creates no renderer or synthetic image data.
    """

    source = "oracle"

    def __init__(self, env: PandaUTableEnv) -> None:
        self._env = env

    def estimate(self) -> TaskStateEstimate:
        start = time.perf_counter()
        object_position = tuple(
            float(value) for value in self._env.data.xpos[self._env.object_body_id]
        )
        target_position = tuple(
            float(value)
            for value in self._env.data.site_xpos[self._env.place_target_site_id]
        )
        return TaskStateEstimate(
            object_id="pick_object_0",
            target_id="place_target_0",
            object_position=object_position,
            target_position=target_position,
            timestamp=float(self._env.data.time),
            source=self.source,
            valid=True,
            confidence=1.0,
            failure_reason=None,
            object_pixel_count=0,
            target_pixel_count=0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            camera_name=None,
            image_resolution=None,
            object_valid=True,
            target_valid=True,
            object_confidence=1.0,
            target_confidence=1.0,
            object_failure_reason=None,
            target_failure_reason=None,
        )

    def close(self) -> None:
        return None
