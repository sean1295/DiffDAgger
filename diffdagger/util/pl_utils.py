import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import os


class BCTrainingEndCallback(Callback):
    def __init__(self, policy, dataset, chkpt_indices, chkpt_dir):
        super().__init__()
        self.policy = policy
        self.dataset = dataset
        self.chkpt_indices = chkpt_indices
        self.chkpt_dir = chkpt_dir

    def on_train_start(self, trainer, pl_module):
        if self.chkpt_dir and self.chkpt_indices:
            os.makedirs(
                self.chkpt_dir, exist_ok=True
            )  # Create directory if it doesn't exist

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.model["ema"].step(pl_module.model["model"].parameters())

    def on_train_end(self, trainer, pl_module):
        pl_module.save_to_ema_model()


class DiffDAggerTrainingEndCallback(BCTrainingEndCallback):
    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        pl_module.model["model"].eval()
        pl_module.update_diffusion_threshold(self.dataset)


class IntermediateCheckpointCallback(Callback):
    def __init__(self, policy, chkpt_indices, chkpt_dir):
        super().__init__()
        self.policy = policy
        self.chkpt_indices = chkpt_indices
        self.chkpt_dir = chkpt_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.model["ema"].step(pl_module.model["model"].parameters())
        global_step = trainer.global_step
        if self.chkpt_indices and self.chkpt_dir and global_step in self.chkpt_indices:
            pl_module.save_to_ema_model()
            pl_module.save(
                f"{self.chkpt_dir}/{self.chkpt_indices.index(global_step)}.pth"
            )


class EvaluationCallback(pl.Callback):
    def __init__(self, policy, eval_env, eval_kwarg, eval_freq=1000):
        self.policy = policy
        self.eval_env = eval_env
        self.eval_kwarg = eval_kwarg
        self.eval_freq = eval_freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Run evaluation every `eval_freq` steps"""
        global_step = trainer.global_step
        if global_step % self.eval_freq == 0:
            pl_module.save_to_ema_model()
            sr = self.policy.parallel_evaluate(
                self.eval_env, pbar=None, **self.eval_kwarg
            )[0]
            trainer.logger.log_metrics({"success_rate": sr}, step=global_step)
            print(f"Step {global_step}: Success Rate = {sr:.4f}")
