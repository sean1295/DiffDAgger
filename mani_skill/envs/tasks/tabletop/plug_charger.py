from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("PlugCharger-v1", max_episode_steps=200)
class PlugChargerEnv(BaseEnv):
    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )

        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)

        return builder.build_kinematic(name="receptacle")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize agent
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            xy = randomization.uniform(
                [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
            )
            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = 0.1
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
            )
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            success=success,
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                charger_pose=self.charger.pose.raw_pose,
                receptacle_pose=self.receptacle.pose.raw_pose,
                goal_pose=self.goal_pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


from mani_skill.envs.tasks.tabletop.table_env import TableEnv


@register_env("PlugCharger-v2", max_episode_steps=200)
class PlugChargerEnv2(TableEnv):
    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.01, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_configs = super()._default_sensor_configs
        pose1 = sapien_utils.look_at([0.05, -0.05, -0.05], [0, 0, 0])
        pose2 = sapien_utils.look_at([0.04, -0.08, 0.02], [0.03, 0, 0])
        return camera_configs + [
            CameraConfig(
                "closeup_camera",
                pose=pose1,
                width=256,
                height=256,
                fov=1,
                mount=self.receptacle,
            ),
            CameraConfig(
                "closeup_camera1",
                pose=pose2,
                width=256,
                height=256,
                fov=1,
                mount=self.receptacle,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.03, 0.3, 0.3], [0.05, 0.1, 0.0])
        return CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)

    # @property
    # def _default_human_render_camera_configs(self):
    #     pose = sapien_utils.look_at([0.07, 0.2, 0.08], [0.03, 0.02, 0])
    #     return [
    #         CameraConfig(
    #             "render_camera",
    #             pose=pose,
    #             width=2048,
    #             height=2048,
    #             fov=1,
    #             mount=self.receptacle,
    #         )
    #     ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat1 = sapien.render.RenderMaterial()
        mat1.set_base_color([0.2, 0.2, 0.2, 1])
        mat1.metallic = 1.0
        mat1.roughness = 0.0
        mat1.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat1
        )
        mat2 = sapien.render.RenderMaterial()
        mat2.metallic = 1.0
        mat2.roughness = 0.0
        mat2.specular = 1.0
        mat2.set_base_color([0.35, 0.35, 0.35, 1])
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat2
        )

        # base
        collision_mat = sapien.pysapien.physx.PhysxMaterial(
            static_friction=500.0,
            dynamic_friction=500.0,
            restitution=0,
        )

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.4, 0.8, 0.4, 1])  # option 0
        # mat.set_base_color([0.6, 1.0, 0.6, 1]) #option 1
        # mat.set_base_color([0.8, 0.4, 0.8, 1]) #option 2
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=collision_mat
        )
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )

        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1.0, 1.0, 0.2, 1])  # option 0
        # mat.set_base_color([0.6, 0.8, 1.0, 1])  #option 1
        # mat.set_base_color([1.0, 1.0, 1.0, 1]) #option 2
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        # mat.set_base_color(sapien_utils.hex2rgba("#6D0000"))
        # mat.set_base_color([0.0, 0.0, 0.2, 1])
        mat.set_base_color([0.0, 0.0, 0.0, 1])  # option 3
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0] * 1.001, peg_size[1] * 1.001, peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)

        return builder.build_kinematic(name="receptacle")

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize agent
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize charger
            xy = randomization.uniform([-0.1, -0.05], [0.0, 0.05], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._base_size[2]
            ori = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 4, torch.pi / 4)
            )
            self.charger.set_pose(Pose.create_from_pq(pos, ori))

            # Initialize receptacle
            xy = randomization.uniform([0.1, 0.1], [0.1, 0.1], size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self._receptacle_size[2]
            ori = randomization.random_quaternions(
                n=b,
                lock_x=True,
                lock_y=True,
                bounds=(torch.pi, torch.pi),
            )
            # ori = torch.tensor([[1,0,0,0]]).repeat(b, 1)
            self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

            self.goal_pose = self.receptacle.pose * (
                sapien.Pose(q=euler2quat(0, 0, np.pi))
            )

    @property
    def charger_base_pose(self):
        return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

    def _compute_distance(self):
        obj_pose = self.charger.pose
        obj_to_goal_pos = self.goal_pose.p - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

        obj_to_goal_quat = rotation_conversions.quaternion_multiply(
            rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
        )
        obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
            obj_to_goal_quat
        )
        obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
        obj_to_goal_angle = torch.min(
            obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
        )

        return obj_to_goal_dist, obj_to_goal_angle

    def evaluate(self):
        obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
        success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
        return dict(
            obj_to_goal_dist=obj_to_goal_dist,
            obj_to_goal_angle=obj_to_goal_angle,
            is_charger_grasped=self.agent.is_grasping(self.charger),
            success=success,
        )

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        obs.update(
            charger_pose=self.charger.pose.raw_pose,
            receptacle_pose=self.receptacle.pose.raw_pose,
            goal_pose=self.goal_pose.raw_pose,
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
