import os

os.environ["HYDRA_FULL_ERROR"] = "1"
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
from mani_skill.utils.wrappers import RecordEpisode
from collections import defaultdict, deque
from util.maniskill_util import wrap_env


# Define the example_rollout function to perform policy rollouts in the environment
@hydra.main(
    version_base=None,
    config_path="../diffdagger/config/sim",
    config_name="pusht_state.yaml",
)
def main(cfg: DictConfig):
    env = instantiate(cfg.env)
    env = wrap_env(env, cfg.obs_keys, cfg.action_space)
    env = RecordEpisode(
        env,
        output_dir=cfg.save_file_dir,
        save_video=True,
        save_trajectory=False,
        save_on_reset=True,
        info_on_video=True,
        max_steps_per_video=cfg.env.max_episode_steps + 150,
        video_fps=30,
    )

    eval_env = instantiate(cfg.eval_env)
    eval_env = wrap_env(eval_env, cfg.obs_keys, cfg.action_space)
    dataset = instantiate(cfg.dataset)

    eval_kwarg = {
        "num_ep": 100,
        "max_episode_steps": cfg.env.max_episode_steps,
        "extra_steps": 0,
        "action_horizon": cfg.action_horizon,
        "seed_base": 7777,
    }

    result = defaultdict(list)
    num_init_ep = cfg.num_init_ep
    total_episodes = 500
    seed = 0

    policy = instantiate(cfg.policy, _recursive_=False)
    expert = instantiate(cfg.expert, _recursive_=False)


if __name__ == "__main__":
    main()
