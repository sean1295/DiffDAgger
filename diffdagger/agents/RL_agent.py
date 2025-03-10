import numpy as np
import torch
from hydra.utils import instantiate
import random
import copy
from tensordict import TensorDict
from collections import defaultdict


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    action = torch.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


class MultipleExperts:
    def __init__(
        self,
        model,
        ckpts,
        obs_keys,
        action_space,
        max_episode_steps,
        success_mode,
        device,
    ):

        self.experts = [instantiate(model).to(device) for _ in range(len(ckpts))]
        for i in range(len(self.experts)):
            self.experts[i].load_state_dict(torch.load(ckpts[i]))
        self.obs_keys = sorted(obs_keys)
        self.action_space = action_space
        self.max_episode_steps = max_episode_steps
        self.success_mode = (
            success_mode  # Expert will continue demonstration until success
        )

    def reset(self, env):
        self.env = env
        self.base_env = env.unwrapped
        self.env_agent = self.base_env.agent
        self.robot = self.env_agent.robot

    def generate_random_action(self):
        num_joints = self.env.num_joints
        cur_joint_pos = self.robot.get_qpos()[:, :num_joints]
        joint_action = cur_joint_pos + torch.randn_like(cur_joint_pos) * 0.001
        gripper_action = (
            self.robot.get_qpos()[:, -2:].sum(dim=1, keepdim=True) * 20 - 0.6
        )
        action = torch.cat((joint_action, gripper_action), dim=-1)

        return action

    def generate_stationary_action(self):
        num_joints = self.env.num_joints
        action = self.robot.qpos[:, :num_joints]

        return action

    @torch.no_grad()
    def get_action(self, obs, expert_idx=None, *args, **kwargs):
        expert_obs = torch.cat([obs[key] for key in self.obs_keys], dim=-1)

        if expert_idx is None:
            expert = random.choice(self.experts)
        else:
            expert = self.experts[expert_idx]

        expert_action = expert.get_action(expert_obs, *args, **kwargs)
        return expert_action

    def move_to_next_goal(self, ep_info):
        tactile = []
        num_joints = self.env.num_joints
        truncated, success = False, False
        data_dict = defaultdict(list)
        count = 0

        while not truncated and not success:
            count += 1
            obs = self.env.get_obs(self.env.get_info())
            expert_action = self.get_action(
                obs, expert_idx=ep_info["seed"] % len(self.experts), deterministic=True
            )
            td = obs.copy()

            prev_pose = copy.deepcopy(self.env.agent.tcp.pose)
            prev_joint_pos = self.env.agent.robot.get_qpos()[:, :num_joints].clone()
            joint_delta_pos = clip_and_scale_action(
                expert_action,
                self.env.agent._controller_configs["pd_joint_delta_pos"]["arm"].lower,
                self.env.agent._controller_configs["pd_joint_delta_pos"]["arm"].upper,
            )
            joint_delta_pos[:, :num_joints] = (
                joint_delta_pos[:, :num_joints] + torch.pi
            ) % (2 * torch.pi) - torch.pi
            target_joint_pos = prev_joint_pos + joint_delta_pos

            obs, _, done, _, info = self.env.step(target_joint_pos)
            # tactile.append(self.env.agent.robot.get_net_contact_forces(["panda_hand"]).clone())
            action_dict = {}

            action_dict["action_joint_delta_pos"] = joint_delta_pos
            action_dict["action_joint_pos"] = target_joint_pos[:, :num_joints]
            action_dict["action_ee_delta_pose"] = (
                self.env.compute_ee_delta_pose_from_qpos(prev_pose, target_joint_pos)
            )
            action_dict["action_ee_pose"] = self.env.compute_ee_pose_from_qpos(
                target_joint_pos
            )

            td.update(
                {
                    **action_dict,
                    "done": done,
                    "episode": torch.ones(1) * (ep_info["seed"]),
                }
            )
            for k, v in td.items():
                data_dict[k].append(v.cpu())
            success = info["success"]
            truncated = count >= self.max_episode_steps
            if not self.success_mode:
                break

        if success:
            for _ in range(5):
                obs = self.env.get_obs(self.env.get_info())
                expert_action = self.get_action(
                    obs,
                    expert_idx=ep_info["seed"] % len(self.experts),
                    deterministic=True,
                )
                td = obs.copy()

                prev_pose = copy.deepcopy(self.env.agent.tcp.pose)
                prev_joint_pos = self.env.agent.robot.get_qpos()[:, :num_joints].clone()
                joint_delta_pos = clip_and_scale_action(
                    expert_action,
                    self.env.agent._controller_configs["pd_joint_delta_pos"][
                        "arm"
                    ].lower,
                    self.env.agent._controller_configs["pd_joint_delta_pos"][
                        "arm"
                    ].upper,
                )
                joint_delta_pos[:, :num_joints] = (
                    joint_delta_pos[:, :num_joints] + torch.pi
                ) % (2 * torch.pi) - torch.pi
                target_joint_pos = prev_joint_pos + joint_delta_pos

                obs, _, done, _, info = self.env.step(target_joint_pos)
                # tactile.append(self.env.agent.robot.get_net_contact_forces(["panda_hand"]).clone())
                action_dict = {}

                action_dict["action_joint_delta_pos"] = joint_delta_pos
                action_dict["action_joint_pos"] = target_joint_pos[:, :num_joints]
                action_dict["action_ee_delta_pose"] = (
                    self.env.compute_ee_delta_pose_from_qpos(
                        prev_pose, target_joint_pos
                    )
                )
                action_dict["action_ee_pose"] = self.env.compute_ee_pose_from_qpos(
                    target_joint_pos
                )

                td.update(
                    {
                        **action_dict,
                        "done": done,
                        "episode": torch.ones(1) * (ep_info["seed"]),
                    }
                )
                for k, v in td.items():
                    data_dict[k].append(v.cpu())
        return {k: torch.cat(v) for k, v in data_dict.items()}

    def setup_task(self):
        pass
