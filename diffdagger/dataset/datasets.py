import numpy as np
import torch
from torch.utils.data import Dataset
from tensordict import TensorDict, MemoryMappedTensor
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from util.normalization import (
    CDFNormalizer,
    SafeLimitsNormalizer,
    get_vision_normalizer,
)
from typing import Union, List
import roma
from util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        dataset_path: Union[str, List[str]] = None,
        load_count: int = 100,
        obs_keys: list = [],
        action_space: str = "joint_pos",
        device: str = "cpu",
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        validation_ratio=0.2,
    ) -> None:

        self.obs_keys = obs_keys
        self.action_space = action_space
        self.device = device
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.validation_ratio = validation_ratio
        self.num_gpus = torch.cuda.device_count()

        if dataset_path is not None and load_count != 0:
            if isinstance(dataset_path, str):
                dataset_path = [
                    dataset_path
                ]  # Convert to list if a single path is provided
            self.rb = self.load_and_concatenate(dataset_path, load_count)
            assert isinstance(
                self.rb, TensorDictReplayBuffer
            ), "The loaded dataset is not a TensorDictReplayBuffer"
            sample = self.rb.sample(1)
            for key in obs_keys:
                assert (
                    key in sample
                ), f"The replay buffer is missing the required key: {key}"
            self.make_indices()
            self.generate_action_sequence()
            self.set_normalizers()
        else:
            self.rb = TensorDictReplayBuffer(storage=LazyTensorStorage(int(5e4)))

    def load_and_concatenate(
        self, dataset_paths: List[str], load_count: int
    ) -> TensorDictReplayBuffer:
        buffers = []
        for path in dataset_paths:
            rb = torch.load(path)
            partial_rb = self.partial_load(rb, load_count)
            buffers.append(partial_rb)
        # Concatenate all replay buffers
        concatenated_rb = TensorDictReplayBuffer(storage=LazyTensorStorage(int(5e4)))
        for buffer in buffers:
            concatenated_rb.extend(buffer.storage._storage)

        return concatenated_rb

    def partial_load(self, rb, load_count):
        assert load_count != 0, "load count must be either -1 or a positive integer"
        if load_count == -1:
            return rb
        else:
            episodes = rb.storage["episode"].unique()
            selected_episodes = episodes[:load_count]
            mask = torch.isin(rb.storage["episode"], selected_episodes)
            partial_rb = TensorDictReplayBuffer(
                storage=LazyTensorStorage(int(mask.sum()))
            )
            indices = torch.nonzero(mask, as_tuple=False).squeeze().tolist()
            partial_rb.extend(rb.storage._storage[indices])
            del rb
            import gc

            gc.collect()
            return partial_rb

    def set_normalizers(self, normalizers=None):
        if normalizers:
            self.normalizers = normalizers
        else:
            self.normalizers = {}
            for key in self.obs_keys:
                if "rgb" in key:
                    self.normalizers[key] = get_vision_normalizer(self.device)
                else:
                    self.normalizers[key] = SafeLimitsNormalizer(self.rb[key])
                self.normalizers[key].to_device(self.device)
            self.normalizers["action"] = SafeLimitsNormalizer(
                self.action_seq.flatten(end_dim=1)
            )
            self.normalizers["action"].to_device(self.device)

    def generate_action_sequence(self):
        if "joint" in self.action_space:
            if "del" in self.action_space:
                action_seq = self.rb[f"action_joint_delta_pos"][self.action_indices]
            else:
                action_seq = self.rb[f"action_joint_pos"][self.action_indices]
                if "rel" in self.action_space:
                    action_seq[:, :, :7] -= self.rb["agent_qpos"][
                        self.action_indices[:, 0], :7
                    ].unsqueeze(1)
        else:
            if "del" in self.action_space:
                action_seq = self.rb[f"action_ee_delta_pose"][self.action_indices]
            else:
                action_seq = self.rb[f"action_ee_pose"][self.action_indices]

            action_seq_pos = action_seq[..., :3]

            if "rel" in self.action_space and "del" not in self.action_space:
                action_seq_pos = action_seq_pos - self.rb["extra_tcp_pose"][
                    self.action_indices[:, 0]
                ][:, :3].unsqueeze(1)
                action_seq_quat = roma.quat_composition(
                    [
                        roma.quat_conjugation(
                            self.rb["extra_tcp_pose"][self.action_indices[:, 0]][
                                :, [4, 5, 6, 3]
                            ]
                        )
                        .unsqueeze(1)
                        .repeat(1, self.pred_horizon, 1),
                        action_seq[:, :, [4, 5, 6, 3]],
                    ]
                )
            else:
                action_seq_quat = action_seq[:, :, [4, 5, 6, 3]]
            if "rotvec" in self.action_space:
                action_seq_ori = roma.unitquat_to_rotvec(action_seq_quat)
            elif "6d" in self.action_space:
                action_seq_ori = matrix_to_rotation_6d(
                    roma.unitquat_to_rotmat(action_seq_quat)
                )
            else:
                action_seq_ori = action_seq_quat[:, :, [3, 0, 1, 2]]

            if action_seq.shape[-1] == 7:
                action_seq = torch.cat((action_seq_pos, action_seq_ori), dim=-1)
            elif action_seq.shape[-1] == 8:
                action_seq_gripper = action_seq[..., -1:]
                action_seq = torch.cat(
                    (action_seq_pos, action_seq_ori, action_seq_gripper), dim=-1
                )

        self.action_seq = action_seq.to(self.device)

    def set_normalizers_from_path(self, path):
        self.set_normalizers(torch.load(path))
        for key in self.normalizers.keys():
            self.normalizers[key].to_device(self.device)

    def make_indices(self):
        obs_indices, action_indices = [], []
        episode_starts = []
        current_episode = None

        for idx, episode in enumerate(self.rb.storage["episode"]):
            if episode != current_episode:
                episode_starts.append(idx)
                current_episode = episode
        episode_starts.append(
            len(self.rb.storage["episode"])
        )  # add fake last next_episode_start element

        for i, episode_start in enumerate(episode_starts[:-1]):
            next_episode_start = episode_starts[i + 1]
            episode_length = next_episode_start - episode_start

            for start in range(-self.obs_horizon + 1, episode_length):
                obs_indices.append(
                    torch.tensor(
                        [
                            max(
                                episode_start,
                                min(episode_start + j, next_episode_start - 1),
                            )
                            for j in range(start, start + self.obs_horizon)
                        ]
                    )
                )
                action_indices.append(
                    torch.tensor(
                        [
                            max(
                                episode_start,
                                min(episode_start + j, next_episode_start - 1),
                            )
                            for j in range(start, start + self.pred_horizon)
                        ]
                    )
                )

        self.obs_indices = torch.stack(obs_indices)
        self.action_indices = torch.stack(action_indices)
        self.episode_starts = episode_starts[:-1]

    def __len__(self):
        return len(self.obs_indices)

    def __getitem__(self, idx):
        obs_index = self.obs_indices[idx]
        action = self.action_seq[idx]
        batch = {}
        for key in self.obs_keys:
            batch[key] = self.rb[key][obs_index].to(self.device)
        batch["action"] = action.to(self.device)

        return batch
