from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from environments import PandaUTableEnv
from perception import OverheadRGBDCamera


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"


class RGBDRenderingTests(unittest.TestCase):
    def test_capture_shape_determinism_and_close(self) -> None:
        env = PandaUTableEnv(CONFIG_PATH)
        env.reset(seed=42)
        camera = OverheadRGBDCamera(env.model, env.config.camera)
        first = camera.capture(env.data)
        second = camera.capture(env.data)
        self.assertEqual(first.rgb.shape, (512, 512, 3))
        self.assertEqual(first.rgb.dtype, np.uint8)
        self.assertEqual(first.depth.shape, (512, 512))
        self.assertTrue(np.issubdtype(first.depth.dtype, np.floating))
        self.assertGreater(np.unique(first.rgb.reshape(-1, 3), axis=0).shape[0], 10)
        self.assertTrue(np.any(np.isfinite(first.depth) & (first.depth > 0.0)))
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(first.depth, second.depth)
        camera.close()
        camera.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            camera.capture(env.data)
        env.close()

    def test_renderer_depth_is_axial_camera_depth(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <visual><global offwidth="64" offheight="64"/></visual>
              <worldbody>
                <camera name="top" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="90"/>
                <geom type="plane" size="5 5 0.1"/>
              </worldbody>
            </mujoco>
            """
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=64, width=64) as renderer:
            renderer.update_scene(data, camera="top")
            renderer.enable_depth_rendering()
            depth = renderer.render().copy()
        # A z=0 plane has axial depth 1 m everywhere. Corner ray range is >1 m.
        np.testing.assert_allclose(depth, 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
