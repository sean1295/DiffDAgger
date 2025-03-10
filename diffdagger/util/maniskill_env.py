import numpy as np
import torch
from gymnasium import spaces
from gymnasium import ObservationWrapper, Wrapper
import roma
import fast_kinematics
import pytorch_kinematics as pk
from os import devnull
from collections.abc import MutableMapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_multiply,
    quaternion_invert,
)
from util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d


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


class FlattenObservationKeyWrapper(ObservationWrapper):
    """
    Flattens the observations into a single vector
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.base_env.update_obs_space(flatten_dict(self.base_env._init_raw_obs))

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        return flatten_dict(observation)


class AdditionalInfoWrapper(Wrapper):
    def reset(self, *args, seed=None, options=dict(), **kwargs):
        return super().reset(*args, seed=seed, options=options, **kwargs)

    def step(self, action_dict):
        if not isinstance(action_dict, dict):
            return super().step(action_dict)
        action = action_dict.pop("action")
        obs, rew, done, trunc, info = super().step(action)
        info.update(action_dict)
        return obs, rew, done, trunc, info


class VariousActionSpaceWrapper(Wrapper):

    def __init__(self, env, action_space="joint_pos", internal_step=1):
        super().__init__(env)
        assert self.control_mode == "pd_joint_pos"
        # self.fast_kinematics_model = fast_kinematics.FastKinematics(self.agent.urdf_path, self.num_envs, self.agent.ee_link_name)
        with open(self.agent.urdf_path, "r") as f:
            urdf_str = f.read()

        @contextmanager
        def suppress_stdout_stderr():
            """A context manager that redirects stdout and stderr to devnull"""
            with open(devnull, "w") as fnull:
                with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
                    yield (err, out)

        with suppress_stdout_stderr():
            self.pk_chain = pk.build_serial_chain_from_urdf(
                urdf_str,
                end_link_name=self.agent.ee_link_name,
            ).to(device=self.device)
        self.use_gripper = (
            "gripper" in next(iter(self.agent.controllers.values())).configs
        )
        self.internal_step = internal_step
        self.num_joints = len(self.agent.arm_joint_names)
        self.set_action_space(action_space)

    def transform_obs_pose_as_rotmat(self, obs_dict):
        _obs_dict = obs_dict.copy()
        for key in _obs_dict.keys():
            if "pose" in key:
                quat = _obs_dict[key][..., [4, 5, 6, 3]]
                rotmat = roma.unitquat_to_rotmat(quat).flatten(start_dim=-2)
                _obs_dict[key] = torch.cat((_obs_dict[key][..., :3], rotmat), dim=-1)

        return _obs_dict

    def compute_ee_delta_pose(self, prev_pose):

        pose_matrix = self.pk_chain.forward_kinematics(
            self.agent.robot.qpos[:, : self.num_joints]
        ).get_matrix()
        pose_p = pose_matrix[:, :3, 3]
        pose_q = roma.rotmat_to_unitquat(pose_matrix[:, :3, :3])[:, [3, 0, 1, 2]]
        delta_pos = pose_p + self.agent.robot.root_pose.p - prev_pose.p
        delta_quat = quaternion_multiply(quaternion_invert(pose_q), prev_pose.q)
        rotvec = roma.unitquat_to_rotvec(delta_quat[:, [1, 2, 3, 0]])

        ee_delta_pose = torch.cat([delta_pos, rotvec], dim=-1)
        quat = roma.rotvec_to_unitquat(ee_delta_pose[:, 3:6])[:, [3, 0, 1, 2]]
        ee_delta_pose = torch.cat([ee_delta_pose[:, :3], quat], dim=-1)

        return ee_delta_pose

    def compute_ee_delta_pose_from_qpos(self, prev_pose, qpos):

        # pose = Pose(self.fast_kinematics_model.forward_kinematics_pytorch(qpos.flatten()).view(qpos.shape[0], -1))# * self.agent.robot.root_pose
        pose_matrix = self.pk_chain.forward_kinematics(qpos).get_matrix()
        pose_p = pose_matrix[:, :3, 3]
        pose_q = roma.rotmat_to_unitquat(pose_matrix[:, :3, :3])[:, [3, 0, 1, 2]]
        delta_pos = pose_p + self.agent.robot.root_pose.p - prev_pose.p
        delta_quat = quaternion_multiply(quaternion_invert(pose_q), prev_pose.q)
        rotvec = roma.unitquat_to_rotvec(delta_quat[:, [1, 2, 3, 0]])

        ee_delta_pose = torch.cat([delta_pos, rotvec], dim=-1)
        quat = roma.rotvec_to_unitquat(ee_delta_pose[:, 3:6])[:, [3, 0, 1, 2]]
        ee_delta_pose = torch.cat([ee_delta_pose[:, :3], quat], dim=-1)

        return ee_delta_pose

    def compute_ee_pose(self):
        pose_matrix = self.pk_chain.forward_kinematics(
            self.agent.robot.qpos[:, : self.num_joints]
        ).get_matrix()
        rotmat = pose_matrix[:, :3, :3].swapaxes(-1, -2)
        pos = pose_matrix[:, :3, 3] + self.agent.robot.root_pose.p
        ori = roma.rotmat_to_unitquat(rotmat)[:, [3, 0, 1, 2]]

        ee_delta_pose = torch.cat([pos, ori], dim=-1)

        return ee_delta_pose

    def compute_ee_pose_from_qpos(self, qpos):

        # pose = Pose(self.fast_kinematics_model.forward_kinematics_pytorch(qpos.flatten()).view(qpos.shape[0], -1))# * self.agent.robot.root_pose
        pose_matrix = self.pk_chain.forward_kinematics(qpos).get_matrix()
        rotmat = pose_matrix[:, :3, :3].swapaxes(-1, -2)
        pos = pose_matrix[:, :3, 3] + self.agent.robot.root_pose.p
        ori = roma.rotmat_to_unitquat(rotmat)[:, [3, 0, 1, 2]]

        ee_delta_pose = torch.cat([pos, ori], dim=-1)

        return ee_delta_pose

    def compute_ik(self, delta_pose):
        jacobian = self.pk_chain.jacobian(self.agent.robot.qpos[:, : self.num_joints])
        delta_joint_pos = torch.linalg.pinv(jacobian) @ delta_pose.float().unsqueeze(
            -1
        ).to(self.device)

        return self.agent.robot.qpos[:, : self.num_joints] + delta_joint_pos.squeeze(-1)

    def set_action_space(self, action_space: str = "joint_pos"):
        assert action_space in [
            "joint_pos",
            "del_joint_pos",
            "rel_joint_pos",
            "ee_pose_6d",
            "del_rel_ee_pose_6d",
            "rel_ee_pose_6d",
        ]
        self.eff_action_space = action_space

    def post_process_action(self, action_seq):

        _action_seq = action_seq.clone()

        if "joint" in self.eff_action_space:
            if "rel" in self.eff_action_space:
                _action_seq[:, :, : self.num_joints] += self.agent.robot.get_qpos()[
                    :, None, : self.num_joints
                ]
        else:
            action_seq_pos = action_seq[:, :, :3]
            if "6d" in self.eff_action_space:
                action_seq_quat = roma.rotmat_to_unitquat(
                    rotation_6d_to_matrix(_action_seq[:, :, 3:9])
                )[:, :, [3, 0, 1, 2]]
            elif "rotvec" in self.eff_action_space:
                action_seq_quat = roma.rotvec_to_unitquat(_action_seq[:, :, 3:6])[
                    :, :, [3, 0, 1, 2]
                ]
            else:
                action_seq_quat = _action_seq[:, :, 3:7]

            if "rel" in self.eff_action_space:
                action_seq_pos[:, :, :3] = (
                    self.agent.tcp.pose.p[:, :3].unsqueeze(1) + action_seq_pos[:, :, :3]
                )
                action_seq_quat[:, :, [1, 2, 3, 0]] = roma.quat_composition(
                    [
                        self.agent.tcp.pose.q[:, [1, 2, 3, 0]]
                        .unsqueeze(1)
                        .repeat(1, _action_seq.shape[1], 1),
                        action_seq_quat[:, :, [1, 2, 3, 0]],
                    ]
                )
            if self.use_gripper:
                action_seq_gripper = _action_seq[:, :, -1:]
                _action_seq = torch.cat(
                    (action_seq_pos, action_seq_quat, action_seq_gripper), dim=-1
                )
            else:
                _action_seq = torch.cat((action_seq_pos, action_seq_quat), dim=-1)

        return _action_seq

    def step(self, action):
        target_qpos = self.transform_action(action)
        return self.env.step(target_qpos.float())

    def transform_action(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        assert isinstance(action, torch.Tensor)
        if action.ndim == 1:
            action = action[None]
        action = action.to(self.device)
        if self.use_gripper:
            non_gripper_action = action[:, :-1].clone()
        else:
            non_gripper_action = action.clone()

        if "ee" in self.eff_action_space:
            delta_pose = non_gripper_action
            delta_pose[:, 3:6] = roma.unitquat_to_rotvec(delta_pose[:, [4, 5, 6, 3]])

            if "delta" not in self.eff_action_space:
                delta_pose[:, :3] -= self.agent.tcp.pose.p
                robot_rotvec = roma.unitquat_to_rotvec(
                    self.agent.tcp.pose.q[:, [1, 2, 3, 0]]
                )
                target_pose_rotvec = delta_pose[:, 3:6]
                delta_pose[:, 3:6] = -roma.rotvec_composition(
                    [robot_rotvec.float(), target_pose_rotvec.float()]
                )

            delta_pose = delta_pose[:, :6]
            target_qpos = self.compute_ik(delta_pose)
        else:
            if "delta" in self.eff_action_space:
                delta_joint_pos = non_gripper_action
                num_joints = len(self.agent.arm_joint_names)
                target_qpos = (
                    delta_joint_pos + self.agent.robot.get_qpos()[:, :num_joints]
                )
            else:
                target_qpos = non_gripper_action

        if self.use_gripper:
            target_qpos = torch.cat([target_qpos, action[:, -1:]], dim=-1)

        return target_qpos.float()


from gymnasium.wrappers import FilterObservation


def wrap_env(env, obs_keys, action_space):
    env = FlattenObservationKeyWrapper(env)
    env = FilterObservation(env, filter_keys=obs_keys)
    env = VariousActionSpaceWrapper(env, action_space=action_space)
    env = AdditionalInfoWrapper(env)
    return env
