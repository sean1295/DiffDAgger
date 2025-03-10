import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import datetime
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm
import time
import wandb
import copy
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device
print("DEVICE", device)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(
    version_base=None, config_path="config/test", config_name="bread_toast.yaml"
)
def main(cfg: DictConfig):
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        # set the wandb project where this run will be logged
        project="diff-dagger"
    )
    dataset = hydra.utils.instantiate(cfg.dataset)
    print(dataset.rb)
    train_dataset_td = dataset
    policy = hydra.utils.instantiate(cfg.policy, _recursive_=False)
    policy.normalizers = dataset.normalizers
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=torch.cuda.device_count()
        * cfg.train.max_train_steps
        * cfg.train.train_bs,
    )
    dataloader_train = torch.utils.data.DataLoader(
        train_dataset_td, batch_size=cfg.train.train_bs, sampler=sampler
    )  # , collate_fn=lambda x:x)
    # Optimizer and scheduler setup
    optimizer, scheduler = policy.get_optimizer_scheduler(cfg)
    policy.model["model"], optimizer, dataloader_train, scheduler = accelerator.prepare(
        policy.model["model"], optimizer, dataloader_train, scheduler
    )
    # accelerator = None
    time_init = time.time()

    prefix = (
        cfg.action_space + "_" + cfg.prediction_type + "_" + cfg.policy.vision_model
    )
    if "Transformer" in cfg.policy.model._target_:
        prefix += "_transformer"
    else:
        prefix += "_unet"
    if cfg.dataset.skip_frames:
        prefix += f"_skipframe{cfg.dataset.skip_frames}"
    prefix += f"_diffstep{cfg.policy.noise_scheduler.num_train_timesteps}"
    prefix += f"_AH{cfg.pred_horizon}"
    # Training loop
    with tqdm(dataloader_train, disable=True) as pbar:
        for i, batch in enumerate(pbar):
            train_loss = policy.train_model_step(
                batch, optimizer, scheduler, accelerator
            )
            wandb.log({"Diffusion Loss": train_loss})
            policy.model["ema"].copy_to(policy.model["ema_model"].parameters())

            if i % 5000 == 0 and i > 4999:
                if accelerator.is_main_process:
                    print(i, end=" ")
                    policy_copy = copy.deepcopy(policy)
                    policy_copy.model["model"] = accelerator.unwrap_model(
                        policy_copy.model["model"]
                    )
                    policy_copy.dagger_reset(train_dataset_td)
                    print(f"Policy reset complete...")
                    print(
                        f"Policy max cdf value: {max(policy_copy.diffusion_loss_cdf.data)}"
                    )
                    print(
                        f"Threshold with alpha {policy_copy.alpha}: {policy_copy.diffusion_loss_threshold}"
                    )
                    file_name = f"{cfg.dataset_dir}/{prefix}_{i//1000}k.pth"
                    policy_copy.save(file_name)
                    print(f"Policy saved to {file_name}")
                    wandb.log(
                        {"Max Diffusion Loss": max(policy_copy.diffusion_loss_cdf.data)}
                    )
                    wandb.log(
                        {
                            "Diffusion Loss Threshold": policy_copy.diffusion_loss_threshold
                        }
                    )
                accelerator.wait_for_everyone()

    print(f"Train Frequency: {len(dataloader_train)/(time.time()-time_init):.3f} Hz")


if __name__ == "__main__":
    main()
