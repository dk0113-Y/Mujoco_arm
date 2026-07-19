from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np

from .config import EnvConfig, load_config
from .randomization import EpisodeParameters, sample_episode_parameters
from .workspace import TABLE_GEOM_NAMES, Workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANDA_XML_PATH = (
    PROJECT_ROOT
    / "models"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)
PANDA_ASSET_DIR = PANDA_XML_PATH.parent / "assets"
DEFAULT_SCENE_PATH = PROJECT_ROOT / "scenes" / "panda_u_table_scene.xml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "u_table.toml"

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FINGER_JOINT_NAMES = ("finger_joint1", "finger_joint2")
PANDA_JOINT_NAMES = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
ARM_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))
GRIPPER_ACTUATOR_NAME = "actuator8"


class InvalidResetError(RuntimeError):
    """Raised when a configured or sampled initial state is geometrically invalid."""


def _required_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing required object: {name}")
    return int(object_id)


def load_u_table_model(
    scene_path: str | Path = DEFAULT_SCENE_PATH,
) -> mujoco.MjModel:
    """Merge the read-only Menagerie Panda XML with the project U-table scene."""
    scene_path = Path(scene_path).expanduser().resolve()
    if not PANDA_XML_PATH.is_file():
        raise FileNotFoundError(
            "Panda submodule model is missing. Run: git submodule update --init --recursive\n"
            f"Expected: {PANDA_XML_PATH}"
        )
    if not scene_path.is_file():
        raise FileNotFoundError(f"U-table scene does not exist: {scene_path}")

    panda_root = ET.parse(PANDA_XML_PATH).getroot()
    compiler = panda_root.find("compiler")
    worldbody = panda_root.find("worldbody")
    if compiler is None or worldbody is None:
        raise RuntimeError("Panda XML is missing compiler or worldbody")
    compiler.set("meshdir", PANDA_ASSET_DIR.resolve().as_posix())

    scene_root = ET.parse(scene_path).getroot()
    scene_worldbody = scene_root.find("worldbody")
    if scene_worldbody is None:
        raise RuntimeError("U-table scene XML is missing worldbody")
    for element in list(scene_worldbody):
        worldbody.append(element)

    scene_visual = scene_root.find("visual")
    if scene_visual is not None:
        panda_visual = panda_root.find("visual")
        if panda_visual is None:
            panda_root.append(scene_visual)
        else:
            for element in list(scene_visual):
                panda_visual.append(element)

    hand = panda_root.find(".//body[@name='hand']")
    if hand is None:
        raise RuntimeError("Panda XML is missing the hand body")
    ET.SubElement(
        hand,
        "site",
        {
            "name": "gripper_tcp",
            "type": "sphere",
            "pos": "0 0 0.103",
            "size": "0.008",
            "rgba": "1 0 0 1",
        },
    )
    return mujoco.MjModel.from_xml_string(
        ET.tostring(panda_root, encoding="unicode")
    )


class PandaUTableEnv:
    """Configurable Panda U-table simulation with privileged task observations.

    ``step`` uses a deliberately simple reward: 1.0 only when the placement
    success check passes, otherwise 0.0. Task 1 does not define a learning reward.
    """

    def __init__(
        self,
        config: EnvConfig | str | Path = DEFAULT_CONFIG_PATH,
        *,
        scene_path: str | Path = DEFAULT_SCENE_PATH,
    ) -> None:
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        self.model = load_u_table_model(scene_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.config.seed)
        self.current_seed: int | None = self.config.seed
        self.current_episode: EpisodeParameters | None = None
        self.collision_count = 0
        self._active_robot_table_pairs: set[tuple[int, int]] = set()
        self._closed = False

        self.tcp_site_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp"
        )
        self.place_target_site_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "place_target"
        )
        self.object_body_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_object"
        )
        self.object_geom_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "pick_object_geom"
        )
        self.object_joint_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_object_free_joint"
        )
        self.arm_joint_ids = np.asarray(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        self.finger_joint_ids = np.asarray(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in FINGER_JOINT_NAMES
            ],
            dtype=int,
        )
        self.arm_actuator_ids = np.asarray(
            [
                _required_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ARM_ACTUATOR_NAMES
            ],
            dtype=int,
        )
        self.gripper_actuator_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME
        )
        self.overhead_camera_id = _required_id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera.name
        )
        self._configure_overhead_camera()
        self.arm_qpos_addresses = self.model.jnt_qposadr[self.arm_joint_ids].astype(int)
        self.arm_dof_addresses = self.model.jnt_dofadr[self.arm_joint_ids].astype(int)
        self.finger_qpos_addresses = self.model.jnt_qposadr[
            self.finger_joint_ids
        ].astype(int)
        self.finger_dof_addresses = self.model.jnt_dofadr[
            self.finger_joint_ids
        ].astype(int)
        self.object_qpos_address = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_address = int(self.model.jnt_dofadr[self.object_joint_id])
        self.arm_joint_ranges = np.asarray(
            self.model.jnt_range[self.arm_joint_ids], dtype=float
        ).copy()
        self.arm_ctrl_ranges = np.asarray(
            self.model.actuator_ctrlrange[self.arm_actuator_ids], dtype=float
        ).copy()

        self.table_geom_ids = frozenset(
            _required_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in TABLE_GEOM_NAMES.values()
        )
        link0_id = _required_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        self.robot_geom_ids = frozenset(
            geom_id
            for geom_id in range(self.model.ngeom)
            if self._is_descendant_body(int(self.model.geom_bodyid[geom_id]), link0_id)
        )

        self.workspace = Workspace.from_model(
            self.model,
            self.data,
            base_clearance_radius=self.config.workspace.base_clearance_radius,
            object_half_size=self.config.workspace.object_half_size,
            spawn_clearance=self.config.workspace.spawn_clearance,
            target_site_offset=self.config.workspace.target_site_offset,
        )
        self._base_object_mass = float(self.model.body_mass[self.object_body_id])
        self._base_object_inertia = np.asarray(
            self.model.body_inertia[self.object_body_id], dtype=float
        ).copy()
        self._base_object_friction = np.asarray(
            self.model.geom_friction[self.object_geom_id], dtype=float
        ).copy()

    def _configure_overhead_camera(self) -> None:
        if int(self.model.cam_bodyid[self.overhead_camera_id]) != 0:
            raise RuntimeError("overhead_rgbd must be attached directly to worldbody")
        x_axis = np.asarray(self.config.camera.x_axis_world, dtype=float)
        y_axis = np.asarray(self.config.camera.y_axis_world, dtype=float)
        z_axis = np.cross(x_axis, y_axis)
        rotation_world_from_camera = np.column_stack((x_axis, y_axis, z_axis))
        quaternion = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quaternion, rotation_world_from_camera.ravel())
        self.model.cam_pos[self.overhead_camera_id] = self.config.camera.position
        self.model.cam_quat[self.overhead_camera_id] = quaternion
        self.model.cam_fovy[self.overhead_camera_id] = self.config.camera.fovy
        self.model.vis.global_.offwidth = max(
            int(self.model.vis.global_.offwidth), self.config.camera.width
        )
        self.model.vis.global_.offheight = max(
            int(self.model.vis.global_.offheight), self.config.camera.height
        )
        mujoco.mj_forward(self.model, self.data)

    def _is_descendant_body(self, body_id: int, ancestor_id: int) -> bool:
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _reset_panda_home(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        home_id = _required_id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        for joint_name in PANDA_JOINT_NAMES:
            joint_id = _required_id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_address] = self.model.key_qpos[home_id, qpos_address]
        self.data.ctrl[:] = self.model.key_ctrl[home_id]
        self.data.qvel[:] = 0.0

    def _apply_physics(self, episode: EpisodeParameters) -> None:
        self.model.body_mass[self.object_body_id] = self._base_object_mass
        self.model.body_inertia[self.object_body_id] = self._base_object_inertia
        self.model.geom_friction[self.object_geom_id] = self._base_object_friction
        mass_scale = episode.mass / self._base_object_mass
        self.model.body_mass[self.object_body_id] = episode.mass
        self.model.body_inertia[self.object_body_id] = (
            self._base_object_inertia * mass_scale
        )
        self.model.geom_friction[self.object_geom_id] = episode.friction
        # MuJoCo 3.10 exposes these model arrays as mutable. mj_setConst updates
        # derived subtree constants after changing mass/inertia at runtime.
        mujoco.mj_setConst(self.model, self.data)

    def _robot_table_contact_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self.table_geom_ids and geom2 in self.robot_geom_ids:
                pairs.add((geom2, geom1))
            elif geom2 in self.table_geom_ids and geom1 in self.robot_geom_ids:
                pairs.add((geom1, geom2))
        return pairs

    def robot_table_collision(self) -> bool:
        return bool(self._robot_table_contact_pairs())

    def _update_collision_events(self) -> None:
        current = self._robot_table_contact_pairs()
        self.collision_count += len(current - self._active_robot_table_pairs)
        self._active_robot_table_pairs = current

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed PandaUTableEnv")
        if options:
            unknown = sorted(options)
            raise ValueError(f"Unsupported reset options: {unknown}")
        self.current_episode = None
        self.collision_count = 0
        self._active_robot_table_pairs.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.current_seed = int(seed)

        try:
            episode = sample_episode_parameters(
                self.rng,
                self.config,
                self.workspace,
                seed=self.current_seed,
            )
            self.current_episode = episode
            self._apply_physics(episode)
            self._reset_panda_home()
            self.data.qpos[self.object_qpos_address : self.object_qpos_address + 7] = (
                *episode.pick_position,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            self.data.qvel[self.object_dof_address : self.object_dof_address + 6] = 0.0
            self.model.site_pos[self.place_target_site_id] = episode.place_position
            self.data.ctrl[self.gripper_actuator_id] = (
                self.config.controller.gripper_open_control
            )
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

            pick_region = self.workspace.region(episode.pick_region)
            object_bottom = float(
                self.data.xpos[self.object_body_id, 2]
                - self.config.workspace.object_half_size
            )
            if object_bottom < pick_region.top_z - 1e-9:
                raise InvalidResetError(
                    f"Object initially penetrates table by {pick_region.top_z - object_bottom:.6g} m"
                )
            if self.robot_table_collision():
                raise InvalidResetError("Panda home pose contacts a U-table geom")

            settle_until = self.config.simulation.settle_time
            while self.data.time + 1e-12 < settle_until:
                mujoco.mj_step(self.model, self.data)
                self._update_collision_events()
                if self.robot_table_collision():
                    raise InvalidResetError(
                        "Panda contacted a U-table geom while settling the reset state"
                    )

            arrays = (self.data.qpos, self.data.qvel, self.data.xpos)
            if not all(np.all(np.isfinite(array)) for array in arrays):
                raise InvalidResetError("Reset produced NaN or Inf in MuJoCo state")
            actual_object = self.data.xpos[self.object_body_id].copy()
            if not pick_region.contains_xy(actual_object[:2], self.config.pick.edge_margin):
                raise InvalidResetError("Object left the configured pick region while settling")
        except InvalidResetError:
            raise
        except (RuntimeError, ValueError) as exc:
            raise InvalidResetError(str(exc)) from exc

        if self.config.observation.source == "privileged":
            info = episode.as_dict()
            info["actual_object_position"] = actual_object.tolist()
        else:
            info = {"seed": self.current_seed, "observation_source": "perception"}
        info["simulation_time"] = float(self.data.time)
        return self.observation(), info

    def observation(self) -> dict[str, Any]:
        """Return robot state; external truth is exposed only in privileged mode."""
        observation = {
            "arm_joint_positions": self.data.qpos[self.arm_qpos_addresses].copy(),
            "arm_joint_velocities": self.data.qvel[self.arm_dof_addresses].copy(),
            "finger_positions": self.data.qpos[self.finger_qpos_addresses].copy(),
            "tcp_position": self.data.site_xpos[self.tcp_site_id].copy(),
            "tcp_orientation": self.data.site_xmat[self.tcp_site_id]
            .reshape(3, 3)
            .copy(),
            "simulation_time": float(self.data.time),
        }
        if self.config.observation.source == "privileged":
            observation.update(
                {
                    "privileged_object_position": self.data.xpos[
                        self.object_body_id
                    ].copy(),
                    "privileged_place_target_position": self.data.site_xpos[
                        self.place_target_site_id
                    ].copy(),
                }
            )
        return observation

    def placement_errors(self) -> tuple[float, float]:
        if self.current_episode is None:
            return float("inf"), float("inf")
        object_position = self.data.xpos[self.object_body_id]
        target_position = self.data.site_xpos[self.place_target_site_id]
        xy_error = float(np.linalg.norm(object_position[:2] - target_position[:2]))
        desired_center_z = (
            self.workspace.region(self.current_episode.place_region).top_z
            + self.config.workspace.object_half_size
        )
        height_error = abs(float(object_position[2] - desired_center_z))
        return xy_error, height_error

    def success(self) -> bool:
        xy_error, height_error = self.placement_errors()
        return bool(
            xy_error <= self.config.controller.place_xy_tolerance
            and height_error <= self.config.controller.place_height_tolerance
        )

    def failure_reason(
        self,
        *,
        stage: str,
        initial_object_height: float | None = None,
        lift_was_confirmed: bool = False,
    ) -> str | None:
        if self.robot_table_collision():
            return "robot_table_collision"
        if self.data.time >= self.config.simulation.episode_timeout:
            return "timeout"
        return None

    def step(
        self, control: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.current_episode is None:
            raise RuntimeError("reset() must be called before step()")
        control_array = np.asarray(control, dtype=float)
        if control_array.shape != (self.model.nu,):
            raise ValueError(
                f"control must have shape ({self.model.nu},), got {control_array.shape}"
            )
        if not np.all(np.isfinite(control_array)):
            raise ValueError("control contains NaN or Inf")
        self.data.ctrl[:] = np.clip(
            control_array,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        for _ in range(self.config.simulation.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._update_collision_events()

        privileged = self.config.observation.source == "privileged"
        success = self.success() if privileged else False
        collision = self.robot_table_collision()
        timeout = self.data.time >= self.config.simulation.episode_timeout
        terminated = success or collision
        truncated = timeout and not terminated
        failure_reason = "robot_table_collision" if collision else (
            "timeout" if truncated else None
        )
        xy_error, height_error = (
            self.placement_errors() if privileged else (None, None)
        )
        info = {
            "success": success,
            "failure_reason": failure_reason,
            "collision_count": self.collision_count,
            "place_xy_error": xy_error,
            "place_height_error": height_error,
        }
        return self.observation(), 1.0 if success else 0.0, terminated, truncated, info

    def close(self) -> None:
        self._closed = True
