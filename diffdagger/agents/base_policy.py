import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from hydra.utils import instantiate
from collections import deque
import copy
from util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d
from util.augmentation import (
    permute_images,
    center_transform,
    crop_transform,
    rotate_crop_transform,
    rotate_crop_color_transform,
)
from util.fusion import ConcatFusion, FiLM
from model.vision.image_encoder import vision_encoders


class BCPolicy(pl.LightningModule):
    def __init__(
        self,
        obs_keys,
        proprio_dim,
        latent_dim,
        action_dim,
        action_space,
        obs_horizon,
        pred_horizon,
        model,
        vision_model,
        frozen_encoder,
        obs_encoder_group_norm,
        spatial_softmax,
        optim,
        scheduler,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.views = [key for key in obs_keys if "image" in key or "rgb" in key]
        self.proprio_dim = proprio_dim
        self.latent_dim = latent_dim
        if latent_dim:
            self.obs_dim = (
                512 * (1 + int(spatial_softmax)) * len(self.views) + latent_dim
            )
        else:
            self.obs_dim = (
                512 * (1 + int(spatial_softmax)) * len(self.views) + proprio_dim
            )
        self.action_dim = action_dim
        self.action_space = action_space
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.frozen_encoder = frozen_encoder

        self.obs_keys = sorted(obs_keys)
        self.image_keys = sorted(
            [key for key in self.obs_keys if "image" in key or "rgb" in key]
        )
        self.non_image_keys = sorted(
            [key for key in self.obs_keys if "image" not in key and "rgb" not in key]
        )
        self.obs_encoder_group_norm = obs_encoder_group_norm
        self.spatial_softmax = spatial_softmax
        self.vision_model = vision_model
        self.train_transform = (
            rotate_crop_transform()
        )  # rotate_crop_color_transform rotate_crop_transform crop_transform
        self.eval_transform = center_transform()

        self.self_cond = False
        self.w = None  # 2.0

        if model is not None:
            self.model = self.get_model(model)

    def set_normalizers_from_path(self, path):
        self.set_normalizers(torch.load(path))

    def set_normalizers(self, normalizers):
        self.normalizers = normalizers
        for key in self.normalizers.keys():
            self.normalizers[key].to_device(self.device)

    def get_model(self, model_config, recursive=True):
        model_config.global_cond_dim = self.obs_dim
        model = dict()
        model["model"] = nn.ModuleDict(
            {
                "image_encoder": vision_encoders(
                    self.vision_model,
                    self.views,
                    self.latent_dim,
                    self.frozen_encoder,
                    self.obs_encoder_group_norm,
                    self.spatial_softmax,
                ),
                "fusion_fn": ConcatFusion(),
                "proprio_encoder": (
                    nn.Linear(self.proprio_dim, self.latent_dim)
                    if self.latent_dim and self.proprio_dim
                    else nn.Identity()
                ),
                "policy_head": instantiate(model_config, _recursive_=recursive),
            }
        )
        if self.self_cond:
            model["model"]["act_proj"] = FiLM(
                self.action_dim, self.action_dim, False, 256
            )

        if self.frozen_encoder:
            model["model"]["image_encoder"].requires_grad_(False)

        model["ema"] = EMAModel(parameters=model["model"].parameters(), power=0.75)
        model["ema_model"] = copy.deepcopy(
            model["model"]
        )  # Note the use of .module to access the wrapped model
        model["ema_model"].eval()
        model["ema_model"].requires_grad_(False)
        return model

    def process_batch(self, batch):
        for key in self.non_image_keys:
            batch[key] = self.normalizers[key].normalize(batch[key].float())
        for key in self.image_keys:
            batch[key] = batch[key].float()
        if "action" in batch:
            batch["action"] = self.normalizers["action"].normalize(
                batch["action"].float()
            )
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        action_batch = batch["action"]
        img_feats, proprio_feats = {}, {}
        if self.image_keys:
            for key in self.image_keys:
                if "rgb" in key or "image" in key:
                    B, T, C, H, W = batch[key].shape
                    img_feats[key] = self.train_transform(
                        batch[key].flatten(end_dim=1)
                    ).reshape(B, T, C, *self.eval_transform.transforms[0].size)
                    img_feats[key] = self.model["model"]["image_encoder"](
                        img_feats[key], key
                    ).to(memory_format=torch.contiguous_format)
            img_feats = torch.cat(
                [img_feats[key] for key in sorted(img_feats.keys())], dim=-1
            )

        for key in self.non_image_keys:
            proprio_feats[key] = batch[key]
        proprio_feats = self.model["model"]["proprio_encoder"](
            torch.cat(
                [proprio_feats[key] for key in sorted(proprio_feats.keys())], dim=-1
            )
        )
        obs_batch = self.model["model"]["fusion_fn"](img_feats, proprio_feats)
        pred_action = self.model["model"]["policy_head"](obs_batch)
        loss = self.compute_loss(pred_action, action_batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def compute_loss(self, pred, actual):
        return nn.functional.mse_loss(pred, actual)

    def configure_optimizers(self):
        # Instantiate Optimizer
        optim_params = {
            "lr": self.hparams.optim.lr,
            "weight_decay": self.hparams.optim.weight_decay,
            "betas": self.hparams.optim.betas,
        }
        optimizer = instantiate(
            self.hparams.optim, params=self.model["model"].parameters()
        )

        # Instantiate Scheduler
        scheduler_params = {
            "optimizer": optimizer,
            "num_warmup_steps": self.hparams.scheduler.num_warmup_steps,
            "num_training_steps": self.trainer.max_steps,
        }

        scheduler = instantiate(self.hparams.scheduler, **scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def to(self, *args, **kwargs):
        """
        Override the to() method to move the model, ema_model, and normalizers.
        """
        super().to(*args, **kwargs)  # Call the parent class's to() method

        device = args[0] if args else kwargs.get("device")  # Extract the device

        if hasattr(self, "model") and self.model is not None:
            if "model" in self.model and self.model["model"] is not None:
                self.model["model"].to(device)
            if "ema_model" in self.model and self.model["ema_model"] is not None:
                self.model["ema_model"].to(device)
            if "ema" in self.model and self.model["ema"] is not None:
                self.model["ema"].to(device)
        if hasattr(self, "normalizers") and self.normalizers is not None:
            for key in self.normalizers.keys():
                self.normalizers[key].to_device(device)

        return self  # Return self to allow chaining

    @torch.no_grad()
    def validate_model(self):
        pass

    def ema_step(self):
        self.model["ema"].step(self.model["model"].parameters())

    def save_to_ema_model(self):
        self.model["ema"].copy_to(self.model["ema_model"].parameters())

    @torch.no_grad()
    def normalize_obs(self, obs):
        vis_model = self.model["ema_model"]["image_encoder"]
        proprio_model = self.model["ema_model"]["proprio_encoder"]
        fusion_fn = self.model["ema_model"]["fusion_fn"]
        img_feats, proprio_feats = {}, {}
        if self.image_keys:
            for key in self.image_keys:
                if key in self.image_keys:
                    if obs[key].shape[-1] == 3:  # channel last
                        img_feats[key] = permute_images(obs[key])
                    else:
                        img_feats[key] = obs[key]
                    shape = img_feats[key].shape[:-3]
                    if len(shape) == 2:  # 5D image
                        img_feats[key] = img_feats[key].flatten(end_dim=1)
                    img_feats[key] = self.eval_transform(img_feats[key].float()).to(
                        memory_format=torch.channels_last
                    )
                    img_feats[key] = vis_model(img_feats[key], key).reshape(*shape, -1)
            img_feats = torch.cat(
                [img_feats[key] for key in sorted(img_feats.keys())], dim=-1
            )
        for key in self.non_image_keys:
            proprio_feats[key] = (
                self.normalizers[key].normalize(obs[key].to(self.device)).float()
            )
        proprio_feats = proprio_model(
            torch.cat(
                [proprio_feats[key] for key in sorted(proprio_feats.keys())], dim=-1
            )
        )
        nobs = fusion_fn(img_feats, proprio_feats)
        return nobs

    def normalize_action(self, action):
        naction = self.normalizers[f"action"].normalize(
            action.to(self.device)
        )  # .cpu().numpy()

        return naction.float()

    def unnormalize_action(self, naction):
        action = self.normalizers[f"action"].unnormalize(
            naction.to(self.device)
        )  # .cpu().numpy()

        return action.float()

    def get_naction(self, nobs, *args, **kwargs):
        naction = self.model["ema_model"]["policy_head"](nobs)
        return naction

    @torch.no_grad()
    def get_action(self, obs, *args, **kwargs):
        nobs = self.normalize_obs(obs)
        is_batched = nobs.ndim == 3
        if not is_batched:
            nobs = nobs.unsqueeze_(0)
        assert nobs.ndim == 3
        naction = self.get_naction(nobs, *args, **kwargs)
        if is_batched:
            action = self.unnormalize_action(naction)
        else:
            action = self.unnormalize_action(naction)[0]

        return action.float()

    def parallel_evaluate(
        self,
        env,
        num_ep,
        action_horizon,
        max_episode_steps,
        latency=0,
        render=False,
        seed_base=777777,
        pbar=None,
        *args,
        **kwargs,
    ):
        num_envs = env.num_envs
        comp_times = []
        ep_per_env = num_ep // num_envs
        if render:
            imgs = []
        for ep_idx in range(ep_per_env):
            obs, infos = env.reset(seed=seed_base + ep_idx)
            obs_deques = deque(
                [obs.copy()] * (self.obs_horizon + latency),
                maxlen=self.obs_horizon + latency,
            )
            dones = torch.zeros(num_envs, dtype=bool)
            t = 0
            while not all(dones) and t < max_episode_steps:
                if pbar:
                    pbar.set_description(
                        f"Evaluation: {len(comp_times)}/{num_envs * (ep_idx + 1)}, time {t}"
                    )
                obs_seq = {
                    key: torch.stack(
                        [obs_deques[j][key] for j in range(self.obs_horizon)]
                    ).swapaxes(0, 1)
                    for key in obs.keys()
                }
                action_seq = self.get_action(obs_seq, *args, **kwargs)
                actions = env.post_process_action(action_seq)
                for i in range(action_horizon):
                    action = actions[:, self.obs_horizon - 1 + i, :]
                    obs, _, done, _, infos = env.step(action.float())
                    if "success" in infos.keys():
                        done = infos["success"]
                    elif "is_success" in infos.keys():
                        done = infos["is_success"]
                    obs_deques.append(obs)
                    t += 1
                    if render:
                        imgs.append(env.render())

                    for j in range(num_envs):
                        if done[j] and not dones[j]:
                            comp_times.append(t)
                            dones[j] = True

                    if not t < max_episode_steps or all(dones):
                        break
        if not render:
            return len(comp_times) / num_ep, comp_times
        imgs.append(env.render())

        return len(comp_times) / num_ep, comp_times, imgs
