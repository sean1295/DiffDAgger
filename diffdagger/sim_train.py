import os

os.environ["HYDRA_FULL_ERROR"] = "1"
import hydra
from hydra import compose, initialize
from hydra.utils import instantiate
from IPython.display import Video
import numpy as np
from tqdm.auto import tqdm
import gymnasium as gym
from util.maniskill_env import wrap_env
from mani_skill.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf, DictConfig
import json
import torch
from util.normalization import SafeLimitsNormalizer
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from torchrl.data import TensorDictReplayBuffer, ReplayBuffer
from tensordict import TensorDict


def conditional_resolver(condition, true_value, false_value):
    return true_value if eval(condition) else false_value


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("if", conditional_resolver, replace=True)


# Define the example_rollout function to perform policy rollouts in the environment
@hydra.main(
    version_base=None,
    config_path="../diffdagger/config/sim",
    config_name="pusht_state.yaml",
)
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

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

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    dataset = instantiate(cfg.dataset)
    dataset.set_normalizers_from_path(cfg.normalizers_path)
    print(dataset, dataset.normalizers)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    plt.tight_layout()
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
    chkpts = [[] for _ in range(cfg.train_freq)]
    chkpt_idx = 0
    policy = instantiate(cfg.policy, _recursive_=False)
    expert = instantiate(cfg.expert, _recursive_=False)
    auto_episodes, trunc_episodes, fail_episodes = [], [], []
    num_joints = len(env.agent.arm_joint_names)
    skip_training = False

    with tqdm(total=total_episodes) as pbar:
        while pbar.n - len(auto_episodes) - len(fail_episodes) < total_episodes:
            obs, info = env.reset(seed=seed)
            ep_info = dict(seed=seed)
            env.set_action_space("joint_pos")
            expert.reset(env)
            policy.reset()
            timestep = 0
            ep_tds = []
            diff_losses = []
            obs_deque = deque([obs] * cfg.pred_horizon, maxlen=cfg.obs_horizon)
            done = False
            obs_seq = {
                key: torch.stack(
                    [obs_deque[i][key] for i in range(cfg.obs_horizon)]
                ).swapaxes(0, 1)
                for key in obs_deque[0].keys()
            }
            intervention_count, conseq_intervention_count = 0, 0

            if pbar.n >= num_init_ep:
                if not skip_training:
                    del policy
                    import gc

                    gc.collect()
                    dataset.make_indices()
                    dataset.generate_action_sequence()
                    dataset.normalizers["action"] = SafeLimitsNormalizer(
                        dataset.action_seq.flatten(end_dim=1)
                    )
                    dataset.normalizers["action"].to_device(dataset.device)
                    policy = (
                        instantiate(cfg.policy, _recursive_=False)
                        .float()
                        .to(cfg.device)
                    )
                    policy.set_normalizers(dataset.normalizers)
                    cfg.train.train_steps = (
                        min(
                            cfg.epoch * len(dataset) // cfg.train.train_bs,
                            cfg.max_train_steps,
                        )
                        + 1
                    )
                    policy.hparams.num_warmup_steps = min(
                        cfg.train.train_steps // 10, 250
                    )
                    chkpt_indices = [
                        int(n * cfg.train.train_steps) - 1
                        for n in np.linspace(0.8, 1.0, cfg.train_freq)
                    ]  # intermediate checkpoints fom 80% to 100%
                    chkpts = policy.train_model(
                        cfg,
                        dataset,
                        chkpt_indices=chkpt_indices,
                        chkpt_dir=cfg.save_file_dir,
                        log_dir=f"{cfg.save_file_dir}/lightning_logs",
                    )
                    chkpt_idx = 0

                policy.load(chkpts[chkpt_idx])
                policy.to(cfg.device)
                action_dict = policy.get_action(obs_seq, dagger=True, return_dict=True)
                action_seq = action_dict.pop("action")
                query = action_dict["query"]
                diff_losses.append(action_dict["diffusion_loss"])

            while (
                timestep < cfg.max_episode_steps + cfg.expert.max_episode_steps
                or not done
            ):
                if pbar.n < num_init_ep or query or timestep >= cfg.max_episode_steps:
                    if conseq_intervention_count == 0:
                        intervention_count += 1
                        #### code for waiting a few timesteps ####
                        env.set_action_space(cfg.expert.action_space)
                        if timestep > 0:
                            for _ in range(cfg.wait_timestep):
                                action = expert.generate_stationary_action()
                                obs, reward, done, _, info = env.step(action)
                                obs_deque.append(obs)
                    expert.setup_task()
                    while True:
                        conseq_intervention_count += 1
                        td = expert.move_to_next_goal(ep_info)
                        td.update(dict(episode=torch.ones(len(td["episode"])) * seed))
                        ep_tds.append(td)
                        info = env.get_info()
                        done = td["done"][-1]
                        timestep += len(td["episode"])
                        for i in range(-min(cfg.obs_horizon, len(td["episode"])), 0, 1):
                            obs = {k: td[k][i] for k in cfg.obs_keys}
                            obs_deque.append(obs)
                        text = f"Expert Demonstration: {pbar.n-len(trunc_episodes)}, Automatic Rollout: {len(auto_episodes)}, Timeout Intervention: {len(trunc_episodes)}, Expert Failure: {len(fail_episodes)} t = {timestep}, \
                        {max(result['Success_rate'][-1]) if len(result['Success_rate']) else 0}"
                        pbar.set_description(text)
                        if (
                            done
                            or not timestep
                            < cfg.max_episode_steps + cfg.expert.max_episode_steps
                        ):  ### if dynamically switching, add logic
                            break

                    if (
                        done
                        or not timestep
                        < cfg.max_episode_steps + cfg.expert.max_episode_steps
                    ):
                        break

                else:
                    env.set_action_space(cfg.action_space)
                    actions = env.post_process_action(action_seq)

                    if conseq_intervention_count > 0:
                        conseq_intervention_count = 0

                    for i in range(cfg.action_horizon):
                        action = actions[:, policy.obs_horizon - 1 + i]
                        action_dict["action"] = action
                        obs, reward, _, _, info = env.step(action_dict)
                        done = info["success"]
                        obs_deque.append(obs)
                        text = f"{pbar.n-len(trunc_episodes)}/{len(auto_episodes)}/{len(trunc_episodes)}/{len(fail_episodes)}, {fail_episodes} t = {timestep}, Nov, \
                        {max(result['Success_rate'][-1]) if len(result['Success_rate']) else 0}"
                        pbar.set_description(text)
                        timestep += 1
                        if done or not timestep < cfg.max_episode_steps:
                            break

                        obs_seq = {
                            key: torch.stack(
                                [obs_deque[i][key] for i in range(policy.obs_horizon)]
                            ).swapaxes(0, 1)
                            for key in obs_deque[0].keys()
                        }
                        action_dict = policy.get_action(
                            obs_seq, dagger=True, return_dict=True
                        )
                        action_seq = action_dict.pop("action")
                        query = action_dict["query"]
                        diff_losses.append(action_dict["diffusion_loss"])
                        if query:
                            break
                if done:
                    break

            result["diff_loss"].append(diff_losses)
            seed += 1
            if (
                not done
                or sum([len(td["episode"]) for td in ep_tds])
                > cfg.expert.max_episode_steps
            ):  # the expert tried to intervene, but the episode failed eventually.
                fail_episodes.append(seed)
                skip_training = False
                continue

            if (
                intervention_count == 0
                or sum([len(td["episode"]) for td in ep_tds]) < 10
            ):  # the episode succeeded w/o expert intervention.
                auto_episodes.append(seed)
                skip_training = True
                continue

            pbar.update(1)

            for td in ep_tds:
                for key in td.keys():
                    if "rgb" in key:
                        td[key] = td[key].permute(0, 3, 1, 2)
                    elif key in list(dataset.normalizers.keys()):
                        dataset.normalizers[key].update(td[key])
                dataset.rb.extend(TensorDict(td, batch_size=len(td["episode"])))
            if (
                timestep > cfg.max_episode_steps
            ):  # the timeout intervention is called and episode is successful.
                trunc_episodes.append(seed)
                skip_training = chkpt_idx < len(chkpts) - 1
            else:  # the regular intervention is called and episode is successful.
                skip_training = not pbar.n == num_init_ep and (
                    len(dataset.rb) == 0 or chkpt_idx < len(chkpts) - 1
                )

            if skip_training:
                chkpt_idx += 1

            if pbar.n <= num_init_ep:
                continue

            result["Supervision_count"].append(len(dataset.rb))
            result["Intervention_count"].append(
                result["Intervention_count"][-1] + intervention_count
                if len(result["Intervention_count"])
                else intervention_count
            )

            if len(result["Success_rate"]) % cfg.train_freq == 0:
                ep_evaluations = []
                ep_evaluations.append(
                    policy.parallel_evaluate(eval_env, **eval_kwarg)[0]
                )
            result["Success_rate"].append(ep_evaluations)
            if max(result["Success_rate"][-1]) == 1.0:
                raise ValueError(f"Episode {pbar.n}, Success Rate 100%")

            ep_counts = [
                pbar.n - len(trunc_episodes),
                len(auto_episodes),
                len(trunc_episodes),
                len(fail_episodes),
            ]
            result["Ep_counts"].append(ep_counts)

            fig, ax = plt.subplots(len(result.keys()), figsize=(5, 9), sharex=True)
            for i, (key, values) in enumerate(result.items()):
                if type(values) == dict:
                    for k, v in values.items():
                        ax[i].plot(v, label=k)
                        ax[i].legend()
                elif type(values) == list:
                    if "Success_rate" in key and len(result["Success_rate"]):
                        ax[i].plot([ele[0] for ele in values], label=key)
                    elif "Ep_counts" in key and len(result["Ep_counts"]):
                        ax[i].plot([ele[0] for ele in values], label="succ")
                        ax[i].plot([ele[1] for ele in values], label="no_sup")
                        ax[i].plot([ele[2] for ele in values], label="trunc")
                        ax[i].plot([ele[3] for ele in values], label="fail")
                    elif "Validation_error" in key:
                        ax[i].plot(np.log(values), label=key)
                    elif "diff_loss" in key:
                        pass
                    else:
                        ax[i].plot(values, label=key)
                    ax[i].legend()
            fig.savefig(cfg.save_file_dir + "/state.png")
            torch.save(result, cfg.save_file_dir + "/result.pth")
            torch.save(dataset.rb, cfg.save_file_dir + "/replay_buffer.pth")
            torch.save(dataset.normalizers, cfg.save_file_dir + "/normalizers.pth")
            plt.close(fig)


if __name__ == "__main__":
    main()
