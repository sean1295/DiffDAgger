import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from einops.layers.torch import Rearrange


class LearnablePosEmb(nn.Module):
    def __init__(self, dim, max_steps):
        super().__init__()
        self.embedding = nn.Embedding(max_steps, dim)

    def forward(self, x):
        # Assuming x is a tensor of shape (batch_size,), representing step indices
        return self.embedding(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, device="cuda"):
        super().__init__()
        self.dim = dim
        self.device = device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer("emb", torch.exp(torch.arange(half_dim) * -emb))

    def forward(self, x):
        # device = x.device
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1, 256, 16))
    o = cb(x)
