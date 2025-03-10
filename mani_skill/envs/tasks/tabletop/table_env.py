from mani_skill.envs.sapien_env import BaseEnv
from collections.abc import MutableMapping
from mani_skill.utils.scene_builder.table import TableSceneBuilder
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.envs.utils.observations import (
    sensor_data_to_pointcloud,
    sensor_data_to_rgbd,
)
from transforms3d.euler import euler2quat


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "_"
) -> MutableMapping:
    """
    Recursively flattens a nested dictionary.

    :param d: The dictionary to flatten.
    :param parent_key: The base key string for the current level of keys.
    :param sep: The separator to use between keys.
    :return: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class WoodTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.662,
                    0.212,
                    0.086,
                    -2.685,
                    -0.115,
                    2.898,
                    1.673,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def build(self):
        super().build()
        # cheap way to un-texture table
        for part in self.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(np.array([180, 103, 50, 255]) / 255.0)
                triangle.material.set_base_color_texture(None)
                # triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)


class TableEnv(BaseEnv):
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.5, -0.7, 0.7], [-0.2, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 384, 384, 1, 0.01, 100)

    @property
    def _default_sensor_configs(self):

        # pose1 = sapien_utils.look_at(eye=[-0.4, -.4, 0.3], target=[0.0, 0.1, 0.1])
        pose1 = sapien_utils.look_at(eye=[-0.315, 0.3, 0.2], target=[0.0, 0.1, 0.1])
        pose2 = sapien_utils.look_at(eye=[-0.315, -0.3, 0.2], target=[0.0, 0.1, 0.1])

        return [
            CameraConfig("right_camera", pose1, 256, 256, np.pi / 2, 0.01, 100),
            CameraConfig("left_camera", pose2, 256, 256, np.pi / 2, 0.01, 100),
        ]

    def build_walls(self):
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=(1e-3, 2.418 / 2, 2.0),
            pose=sapien.Pose(
                p=[1.209 / 2 - 0.12, 0.0, 1.0], q=[1, 0, 0, 0]
            ),  # q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.02, 0.02, 0.02, 1])
            ),
        )
        builder.add_box_visual(
            half_size=(1e-3, 0.75, 2.0),
            pose=sapien.Pose(p=[0.0, 2.418 / 2, 1.0], q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.02, 0.02, 0.02, 1])
            ),
        )
        builder.add_box_visual(
            half_size=(1e-3, 0.75, 2.0),
            pose=sapien.Pose(p=[0.0, -2.418 / 2, 1.0], q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.02, 0.02, 0.02, 1])
            ),
        )
        return builder.build_static(name="walls")

    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_spot_light(
            [-1.2, 0, 0.1],
            [1.0, 0.0, -0.0],
            color=[1.0, 1.0, 1.0],
            inner_fov=1,
            outer_fov=10,
            shadow=shadow,
        )  # , shadow_scale=5, shadow_map_size=2048)
        self.scene.add_spot_light(
            [0.0, 0, 1.0],
            [0.0, 0.0, -1.0],
            color=[1.0, 1.0, 1.0],
            inner_fov=1,
            outer_fov=10,
            shadow=shadow,
        )  # , shadow_scale=5, shadow_map_size=2048)
        self.scene.add_directional_light(
            direction=[0, 0, -1], position=[0, 0, 1], color=[0.5, 0.5, 0.5]
        )

    def _load_scene(self, options: dict):
        self.table_scene = WoodTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.walls = self.build_walls()

    def get_obs(self, info: Optional[Dict] = None):
        """
        Return the current observation of the environment. User may call this directly to get the current observation
        as opposed to taking a step with actions in the environment.

        Note that some tasks use info of the current environment state to populate the observations to avoid having to
        compute slow operations twice. For example a state based observation may wish to include a boolean indicating
        if a robot is grasping an object. Computing this boolean correctly is slow, so it is preferable to generate that
        data in the info object by overriding the `self.evaluate` function.

        Args:
            info (Dict): The info object of the environment. Generally should always be the result of `self.get_info()`.
                If this is None (the default), this function will call `self.get_info()` itself
        """
        if info is None:
            info = self.get_info()
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return dict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict(info)
            obs = common.flatten_state_dict(
                state_dict, use_torch=True, device=self.device
            )
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict(info)
            obs = flatten_dict(obs)
        elif self._obs_mode in ["sensor_data", "rgbd", "rgb", "pointcloud"]:
            obs = self._get_obs_with_sensor_data(info)
            if self._obs_mode == "rgbd":
                obs = sensor_data_to_rgbd(
                    obs, self._sensors, rgb=True, depth=True, segmentation=True
                )
            elif self._obs_mode == "rgb":
                # NOTE (stao): this obs mode is merely a convenience, it does not make simulation run noticebally faster
                obs = sensor_data_to_rgbd(
                    obs, self._sensors, rgb=True, depth=False, segmentation=False
                )
            elif self.obs_mode == "pointcloud":
                obs = sensor_data_to_pointcloud(obs, self._sensors)
            ## changes to get all possible features and get a flatten dictionary.
            obs = flatten_dict(obs)
            obs.update(flatten_dict(self._get_obs_state_dict(info)))

        else:
            raise NotImplementedError(self._obs_mode)
        return obs
