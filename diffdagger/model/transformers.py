import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from .conv1d_components import SinusoidalPosEmb
import einops


class TransformerForDiffusion(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        horizon,
        diffusion_step_embed_dim,
        p_drop_emb,
        p_drop_attn,
        n_cond_layers,
        n_layer,
        n_head,
        causal_attn,
        n_obs_steps,
    ):
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        t = horizon
        t_cond = 1
        t_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, diffusion_step_embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, t, diffusion_step_embed_dim))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(diffusion_step_embed_dim)
        self.cond_obs_emb = None

        self.cond_obs_emb = nn.Linear(cond_dim, diffusion_step_embed_dim)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None

        self.cond_pos_emb = nn.Parameter(
            torch.zeros(1, t_cond, diffusion_step_embed_dim)
        )
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=diffusion_step_embed_dim,
                nhead=n_head,
                dim_feedforward=4 * diffusion_step_embed_dim,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_cond_layers
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(diffusion_step_embed_dim, 4 * diffusion_step_embed_dim),
                nn.Mish(),
                nn.Linear(4 * diffusion_step_embed_dim, diffusion_step_embed_dim),
            )
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=diffusion_step_embed_dim,
            nhead=n_head,
            dim_feedforward=4 * diffusion_step_embed_dim,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # important for stability
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=n_layer
        )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = t
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            # assume conditioning over time and observation both
            p, q = torch.meshgrid(torch.arange(t), torch.arange(t_cond), indexing="ij")
            mask = p >= (
                q - 1
            )  # add one dimension since time is the first token in cond
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("memory_mask", mask)
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(diffusion_step_embed_dim)
        self.head = nn.Linear(diffusion_step_embed_dim, input_dim)

        # constants
        self.t = t
        self.t_cond = t_cond
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps

        # init
        self.apply(self._init_weights)
        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "{}.{}".format(mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        # no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        # param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = dict(self.named_parameters())
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters {} made it into both decay/no_decay sets!".format(
            str(inter_params)
        )
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters {} were not separated into either decay/no_decay set!".format(
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        global_cond: torch.Tensor,
        **kwargs
    ):
        """
        x: (B,T,input_dim)
        timesteps: (B,)
        global_cond: (B, global_cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        batch_size = sample.shape[0]
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        cond = einops.rearrange(
            global_cond, "b (s n) ... -> b s (n ...)", b=batch_size, s=self.n_obs_steps
        )
        # (B,To,n_cond)

        # process input
        input_emb = self.input_emb(sample)

        # encoder
        cond_embeddings = time_emb
        # (B,1,n_emb)

        cond_obs_emb = self.cond_obs_emb(cond)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        # (B,To + 1,n_emb)

        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)

        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
        )
        # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_inp)
        return x
