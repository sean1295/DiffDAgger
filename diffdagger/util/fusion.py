import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def forward(self, img, proprio):
        return torch.cat([img, proprio], dim=-1) if len(img) else proprio


class FiLM(nn.Module):
    def __init__(
        self, cond_dim, output_dim, append=False, hidden_dim=64, activation=nn.Mish()
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.append = append
        # Define MLP to map proprio to gamma and beta
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),  # First hidden layer
            activation,  # Activation function
            nn.Linear(hidden_dim, 2 * output_dim),  # Output layer for gamma and beta
        )

    #     # Initialize weights properly
    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     for layer in self.mlp:
    #         if isinstance(layer, nn.Linear):
    #             # He initialization works better with Mish
    #             nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # 'relu' mode works well for Mish
    #             nn.init.zeros_(layer.bias)

    #     # Final layer special initialization
    #     with torch.no_grad():
    #         # Use a small value instead of zero to break symmetry
    #         self.mlp[-1].weight[:self.output_dim].uniform_(-0.01, 0.01)
    #         self.mlp[-1].weight[self.output_dim:].uniform_(-0.01, 0.01)

    #         self.mlp[-1].bias[:self.output_dim].fill_(1)  # Gamma bias set to 1
    #         self.mlp[-1].bias[self.output_dim:].fill_(0)  # Beta bias set to 0

    def forward(self, img, proprio):
        gamma_beta = self.mlp(proprio)  # Compute gamma and beta
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)  # Split output

        img = gamma * img + beta  # Apply FiLM transformation
        return torch.cat([img, proprio], dim=-1) if self.append else img
