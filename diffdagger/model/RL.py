import numpy as np
import torch.nn as nn
import torch
from torch.distributions.normal import Normal


class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float32))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.obs_normalizer = RunningMeanStd(shape=obs_dim)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 1024)),
            # nn.LayerNorm(512),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(512),
            # nn.Tanh(),
            # layer_init(nn.Linear(1024, 1024)),
            # nn.LayerNorm(512),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            nn.Tanh(),
            # layer_init(nn.Linear(1024, 1024)),
            # nn.Tanh(),
            layer_init(nn.Linear(1024, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def normalize_obs(self, obs, update=False):
        if update:
            self.obs_normalizer.update(obs)
        return (obs - self.obs_normalizer.mean) / torch.sqrt(
            self.obs_normalizer.var + 1e-8
        )

    def get_value(self, x):
        x = self.normalize_obs(x, update=False)
        return self.critic(x)

    def get_action(self, x, deterministic=False, update=False):
        x = self.normalize_obs(x, update=update)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        # action_std = torch.clamp(self.actor_std, min=1e-5, max=0.3).expand_as(action_mean)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.clamp(torch.exp(action_logstd), min=1e-5, max=0.5)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None, update=False):
        x = self.normalize_obs(x, update=update)
        action_mean = self.actor_mean(x)
        # action_std = torch.clamp(self.actor_std, min=1e-5, max=0.3).expand_as(action_mean)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.clamp(torch.exp(action_logstd), min=1e-5, max=0.5)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_logstd = nn.Linear(512, action_dim)
        # action rescaling
        # h, l = env.single_action_space.high, env.single_action_space.low
        # self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        # self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        action = self.fc_mean(x)
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        # mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
