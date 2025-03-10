import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from hydra.utils import instantiate
from collections import deque
import copy
from tqdm.auto import tqdm
from itertools import chain
from util.cdf import CDF
from util.rotations import rotation_6d_to_matrix, matrix_to_rotation_6d
from util.augmentation import (
    permute_images,
    center_transform,
    crop_transform,
    rotate_crop_transform,
    rotate_crop_color_transform,
)
from util.pl_utils import (
    IntermediateCheckpointCallback,
    BCTrainingEndCallback,
    DiffDAggerTrainingEndCallback,
    EvaluationCallback,
)
from model.vision.image_encoder import vision_encoders
from agents.base_policy import BCPolicy


class DiffusionPolicy(BCPolicy):
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
        noise_scheduler,
        num_inference_steps,
        optim,
        scheduler,
    ):

        super().__init__(
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
        )
        self.save_hyperparameters()
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = instantiate(noise_scheduler)
        self.noise_scheduler.set_timesteps(num_inference_steps)

    def compute_loss(self, obs_batch, action_batch, timesteps, noise, mask_batch=None):
        noisy_actions = self.noise_scheduler.add_noise(action_batch, noise, timesteps)
        if self.self_cond:
            with torch.no_grad():
                uncon_noise_pred = torch.zeros_like(noisy_actions)
                if self.model["model"].training and torch.rand(1).item() > 0.5:
                    uncon_noise_pred = self.model["model"]["act_proj"](
                        noisy_actions, uncon_noise_pred
                    )
                    uncon_noise_pred = self.model["model"]["policy_head"](
                        uncon_noise_pred, timesteps, obs_batch
                    ).detach()
            noisy_actions = self.model["model"]["act_proj"](
                noisy_actions, uncon_noise_pred
            )
        noise_pred = self.model["model"]["policy_head"](
            noisy_actions, timesteps, obs_batch
        )

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(action_batch, noise, timesteps)
        elif self.noise_scheduler.config.prediction_type == "sample":
            target = action_batch
        else:
            raise TypeError("prediction type not recognized.")

        loss = F.mse_loss(noise_pred, target)

        if mask_batch is not None:
            # Efficient masking and loss calculation
            diff = noise_pred - target  # Difference between predictions and targets
            squared_diff = (
                diff[mask_batch] ** 2
            )  # Apply the mask and compute squared differences
            loss = squared_diff.mean()  # Mean over the masked elements
        else:
            # Standard MSE loss when no mask is provided
            loss = F.mse_loss(noise_pred, target, reduction="mean")
        return loss

    def training_step(self, batch, batch_idx):
        batch = self.process_batch(batch)
        action_batch = batch["action"]
        mask_batch = batch.get("mask", None)
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
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (action_batch.shape[0],),
            device=self.device,
        ).long()
        noise = torch.empty_like(action_batch).normal_(0, 1)
        loss = self.compute_loss(obs_batch, action_batch, timesteps, noise, mask_batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def train_model(
        self,
        cfg,
        dataset,
        eval_env=None,
        eval_freq=None,
        chkpt_indices=None,
        chkpt_dir=None,
        log_dir="./lightning_logs",
    ):
        """
        Trains the model using PyTorch Lightning and periodically evaluates it.
        """
        train_steps = cfg.train.train_steps
        train_bs = cfg.train.train_bs
        sampler = RandomSampler(
            dataset, num_samples=train_steps * train_bs, replacement=True
        )
        dataloader_train = DataLoader(
            dataset, batch_size=train_bs, sampler=sampler, num_workers=0
        )

        callbacks = [
            IntermediateCheckpointCallback(self, chkpt_indices, chkpt_dir),
            BCTrainingEndCallback(self, dataset, chkpt_indices, chkpt_dir),
        ]

        if eval_env is not None and eval_freq is not None:
            eval_kwarg = {
                "num_ep": cfg.eval_env.num_envs,
                "max_episode_steps": cfg.env.max_episode_steps,
                "extra_steps": 0,
                "action_horizon": cfg.action_horizon,
                "seed_base": 7777,
                "render": False,
            }
            callbacks.insert(
                1, EvaluationCallback(self, eval_env, eval_kwarg, eval_freq=eval_freq)
            )
        # self = torch.compile(self)
        trainer = pl.Trainer(
            max_steps=train_steps,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            callbacks=callbacks,
            enable_checkpointing=False,
            default_root_dir=log_dir,
        )

        trainer.fit(self, dataloader_train)

        if chkpt_indices and chkpt_dir:
            return [f"{chkpt_dir}/{i}.pth" for i in range(len(chkpt_indices))]
        else:
            return []

    @torch.no_grad()
    def get_naction(self, nobs, store_output=False, extra_steps=0, initial_noise=None):
        assert nobs.ndim == 3

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        infer_model = self.model["ema_model"]
        naction = (
            initial_noise
            if initial_noise is not None
            else torch.empty(
                (nobs.shape[0], self.pred_horizon, self.action_dim), device=self.device
            ).normal_(0, 1)
        )
        if store_output:
            outputs = [naction.clone()]

        # Initialize previous prediction to None or zeros
        prev_pred = None

        for k in self.noise_scheduler.timesteps:
            if self.self_cond:
                if prev_pred is None:
                    # First step: no self-conditioning
                    prev_pred = torch.zeros_like(naction)

                # Apply self-conditioning using previous prediction
                conditioned_actions = infer_model["act_proj"](naction, prev_pred)
                noise_pred = infer_model["policy_head"](conditioned_actions, k, nobs)

                # Save current prediction for next iteration
                prev_pred = noise_pred
            else:
                # No self-conditioning
                noise_pred = infer_model["policy_head"](naction, k, nobs)

            # Update sample
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction
            )[0]

            if store_output:
                outputs.append(naction.clone())
        for _ in range(extra_steps):
            noise_pred = infer_model["policy_head"](naction, k, nobs)
            variance_noise = torch.empty_like(noise_pred).normal_(0, 1)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
                variance_noise=variance_noise,
            )[0]
            if store_output:
                outputs.append(naction.clone())
        if store_output:
            outputs = torch.cat(outputs)
            return naction, outputs
        return naction

    @torch.no_grad()
    def get_action(self, obs, initial_noise=None, store_output=False, *args, **kwargs):
        nobs = self.normalize_obs(obs)
        assert nobs.ndim == 3
        naction = self.get_naction(nobs, initial_noise=initial_noise, *args, **kwargs)
        if store_output:
            naction, stored_naction = naction
            action = self.unnormalize_action(naction)
            stored_action = self.unnormalize_action(stored_naction)
            return action.float(), stored_action.float()

        action = self.unnormalize_action(naction)
        return action.float()

    def load(self, filename):
        state_dict = torch.load(filename, map_location=self.device)
        self.model["ema_model"].load_state_dict(state_dict["ema_model"])
        self.model["model"].load_state_dict(state_dict["model"])
        self.model["ema"] = state_dict["ema"]
        self.obs_horizon = state_dict["obs_horizon"]
        self.pred_horizon = state_dict["pred_horizon"]
        self.normalizers = state_dict["normalizers"]
        if "diffusion_loss_cdf" in state_dict.keys():
            self.diffusion_loss_cdf = state_dict["diffusion_loss_cdf"]
        if "diffusion_loss_threshold" in state_dict.keys():
            self.diffusion_loss_threshold = state_dict["diffusion_loss_threshold"]

    def save(self, filename):
        load_dict = {
            "ema_model": self.model["ema_model"].state_dict(),
            "model": self.model["model"].state_dict(),
            "ema": self.model["ema"],
            "normalizers": self.normalizers,
            "obs_horizon": self.obs_horizon,
            "pred_horizon": self.pred_horizon,
        }
        if hasattr(self, "diffusion_loss_cdf"):
            load_dict.update(dict(diffusion_loss_cdf=self.diffusion_loss_cdf))
        if hasattr(self, "diffusion_loss_threshold"):
            load_dict.update(
                dict(diffusion_loss_threshold=self.diffusion_loss_threshold)
            )
        torch.save(load_dict, filename)


class DiffDAggerPolicy(DiffusionPolicy):
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
        noise_scheduler,
        num_inference_steps,
        optim,
        scheduler,
        ### DAgger specific ###
        alpha,
        batch_multiplier,
        num_per_batch,
        patience_window,
        patience,
        #######################
    ):

        super().__init__(
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
            noise_scheduler,
            num_inference_steps,
            optim,
            scheduler,
        )
        self.save_hyperparameters()
        self.alpha = alpha  # quantile threshold
        self.batch_multiplier = batch_multiplier
        self.num_per_batch = num_per_batch
        self.patience = patience
        self.patience_window = patience_window
        self.reset()

    def reset(self):
        self.deque = deque([], maxlen=self.patience_window)

    def train_model(
        self,
        cfg,
        dataset,
        eval_env=None,
        eval_freq=None,
        chkpt_indices=None,
        chkpt_dir=None,
        log_dir="./lightning_logs",
    ):
        """
        Trains the model using PyTorch Lightning and saves checkpoints.
        """
        train_steps = cfg.train.train_steps
        train_bs = cfg.train.train_bs

        sampler = RandomSampler(
            dataset, num_samples=train_steps * train_bs, replacement=True
        )
        dataloader_train = DataLoader(
            dataset, batch_size=train_bs, sampler=sampler, num_workers=0
        )
        callbacks = [
            DiffDAggerTrainingEndCallback(self, dataset, chkpt_indices, chkpt_dir)
        ]

        if chkpt_indices and chkpt_dir:
            callbacks.append(
                IntermediateCheckpointCallback(self, chkpt_indices, chkpt_dir)
            )

        if eval_env is not None and eval_freq is not None:
            eval_kwarg = {
                "num_ep": cfg.eval_env.num_envs,
                "max_episode_steps": cfg.env.max_episode_steps,
                "extra_steps": 0,
                "action_horizon": cfg.action_horizon,
                "seed_base": 7777,
                "render": False,
            }
            callbacks.insert(
                1, EvaluationCallback(self, eval_env, eval_kwarg, eval_freq=eval_freq)
            )
            barebones = False
        else:
            barebones = True

        trainer = pl.Trainer(
            max_steps=train_steps,
            accelerator=cfg.train.accelerator,
            devices=cfg.train.devices,
            callbacks=callbacks,
            enable_checkpointing=False,
            default_root_dir=log_dir,  # Specify the custom log directory
            # Add other trainer configurations as needed
        )

        trainer.fit(self, dataloader_train)

        if chkpt_indices and chkpt_dir:
            return [f"{chkpt_dir}/{i}.pth" for i in range(len(chkpt_indices))]
        else:
            return []

    def get_action(
        self,
        obs,
        initial_noise=None,
        extra_steps=0,
        dagger=False,
        return_dict=False,
        obs_batch_size=1,
    ):  # ,
        nobs = self.normalize_obs(obs)
        assert nobs.ndim == 3
        nactions = self.get_naction(
            nobs, initial_noise=initial_noise, extra_steps=extra_steps
        )
        action = self.unnormalize_action(nactions)
        if not dagger:
            return action.float()

        diffusion_loss = self.get_avg_diffusion_loss_ndata(nobs, nactions, repeat=True)
        self.deque.append(diffusion_loss > self.diffusion_loss_threshold)
        if return_dict:
            cdf_value = self.get_cdf_value(diffusion_loss)
            return dict(
                action=action,
                diffusion_loss=diffusion_loss,
                cdf_value=cdf_value,
                patience=max(0, self.patience - sum(self.deque)),
                query=sum(self.deque) >= self.patience,
            )
        else:
            return action, self.patience == 0

    @torch.no_grad()
    def get_avg_diffusion_loss_ndata(self, nobs, naction, repeat=True):
        assert nobs.ndim == 3 and naction.ndim == 3
        if repeat:
            KLD_loss = 0
            nobs_repeat = nobs.repeat(
                self.noise_scheduler.config.num_train_timesteps * self.batch_multiplier,
                1,
                1,
            )
            naction_repeat = naction.repeat(
                self.noise_scheduler.config.num_train_timesteps * self.batch_multiplier,
                1,
                1,
            )
            timesteps = (
                torch.arange(
                    self.noise_scheduler.config.num_train_timesteps
                    * self.batch_multiplier,
                    device=self.device,
                ).long()
                % self.noise_scheduler.config.num_train_timesteps
            )
            for _ in range(self.num_per_batch):
                noise = torch.empty_like(naction_repeat).normal_(0, 1)
                KLD_loss += self.compute_loss(
                    nobs_repeat, naction_repeat, timesteps, noise
                ).item()
        else:
            KLD_loss = 0
            for _ in range(self.num_per_batch):
                noise = torch.empty_like(naction).normal_(0, 1)
                timesteps = (
                    torch.arange(nobs.shape[0], device=self.device).long()
                    % self.noise_scheduler.config.num_train_timesteps
                )
                KLD_loss += self.compute_loss(nobs, naction, timesteps, noise).item()
        return KLD_loss / self.num_per_batch

    @torch.no_grad()
    def get_stats_from_dataset(self, dataset, num_iter=16):
        diffusion_losses = []
        with tqdm(chain.from_iterable([dataset] * num_iter)) as pbar:
            for datapoint in pbar:
                batch = {k: v[None] for k, v in datapoint.items()}
                batch = self.process_batch(batch)
                action_batch = batch.pop("action")
                img_feats, proprio_feats = {}, {}
                for key in self.image_keys:
                    if "rgb" in key or "image" in key:
                        B, N, C, H, W = batch[key].shape
                        img_feats[key] = self.train_transform(
                            batch[key].float().flatten(end_dim=1)
                        ).reshape(B, N, C, *self.eval_transform.transforms[0].size)
                        img_feats[key] = self.model["ema_model"]["image_encoder"](
                            img_feats[key], key
                        ).to(memory_format=torch.contiguous_format)
                for key in self.non_image_keys:
                    proprio_feats[key] = batch[key]
                if self.image_keys:
                    img_feats = torch.cat(
                        [img_feats[key] for key in self.image_keys], dim=-1
                    )
                proprio_feats = self.model["ema_model"]["proprio_encoder"](
                    torch.cat(
                        [proprio_feats[key] for key in self.non_image_keys], dim=-1
                    )
                )
                obs_batch = self.model["ema_model"]["fusion_fn"](
                    img_feats, proprio_feats
                )
                diffusion_losses.append(
                    self.get_avg_diffusion_loss_ndata(
                        obs_batch, action_batch, repeat=True
                    )
                )
                if len(diffusion_losses) > int(5e4):
                    break
        self.diffusion_loss_cdf = CDF(diffusion_losses)
        self.diffusion_loss_threshold = self.diffusion_loss_cdf.get_quantile(self.alpha)
        return diffusion_losses

    def get_cdf_value(self, diffusion_loss):
        cdf_value = self.diffusion_loss_cdf(diffusion_loss).item()
        return cdf_value

    def update_diffusion_threshold(self, dataset, num_iter=16):
        diffusion_losses = self.get_stats_from_dataset(dataset, num_iter=num_iter)
        self.diffusion_loss_cdf = CDF(diffusion_losses)
        self.diffusion_loss_threshold = self.diffusion_loss_cdf.get_quantile(self.alpha)

    def evaluate_with_diffusion_loss(
        self,
        env,
        num_ep,
        action_horizon,
        max_episode_steps,
        latency=0,
        action_repeat=0,
        obs_batch_size=1,
        render=False,
        seed_base=777777,
        pbar=None,
        *args,
        **kwargs,
    ):
        import cv2

        def add_text_to_image(
            image_array,
            text,
            position=(30, 30),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=1,
            color=(255, 255, 255),
            thickness=2,
        ):
            # Make a copy to avoid modifying the original array
            image = image_array.copy()

            # Add text to the image
            cv2.putText(image, text, position, font, font_scale, color, thickness)

            return image

        comp_times = []
        if render:
            imgs = []
        for ep_idx in range(num_ep):
            if pbar:
                pbar.set_description(f"Evaluation: {len(comp_times)}/{ep_idx}")
            obs, info = env.reset(seed=seed_base + ep_idx)
            done, truncated, t = False, False, 0
            obs_deque = deque(
                [obs.copy()] * (self.obs_horizon + latency),
                maxlen=self.obs_horizon + latency,
            )

            while not done and t < max_episode_steps:
                obs_seq = {
                    key: torch.stack(
                        [obs_deque[i][key] for i in range(self.obs_horizon)]
                    )
                    .swapaxes(0, 1)
                    .float()
                    for key in self.obs_keys
                }
                action_dict = self.get_action(obs_seq, dagger=True, return_dict=True)
                if t % action_horizon == 0:
                    action_seq = action_dict["action"][0]
                diff_loss = action_dict["diffusion_loss"]
                cdf_value = action_dict["cdf_value"]
                text_color = (0, 255, 0) if cdf_value < 0.99 else (255, 0, 0)
                action = action_seq[self.obs_horizon - 1 + t % action_horizon, :]
                obs, _, done, truncated, info = env.step(action.float())
                for _ in range(action_repeat):
                    obs, _, done, truncated, info = env.step(action.float())
                if "success" in info.keys():
                    done = info["success"]
                elif "is_success" in info.keys():
                    done = info["is_success"]
                obs_deque.append(obs.copy())
                t += 1
                if render:
                    img = env.render(
                        mode="rgb_array", height=512, width=512, camera_name="agentview"
                    )
                    img = add_text_to_image(
                        img,
                        f"Loss: {diff_loss:.5f}, CDF: {cdf_value:.4f}",
                        position=(30, 50),
                        color=text_color,  # Green text
                        font_scale=1,
                    )
                    imgs.append(img)
                if done or truncated:
                    break
                if done:
                    comp_times.append(t)

        if render:
            return len(comp_times) / num_ep, comp_times, imgs
        return len(comp_times) / num_ep, comp_times
