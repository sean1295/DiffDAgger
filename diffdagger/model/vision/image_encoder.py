import torch
import torch.nn as nn
import torchvision
import copy
from typing import Callable


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class vision_encoders(nn.Module):
    def __init__(
        self,
        model_name="clip",
        views=["rgb"],
        latent_dim=None,
        frozen=True,
        obs_encoder_group_norm=False,
        spatial_softmax=False,
    ):
        super(vision_encoders, self).__init__()
        self.views = views
        self.latent_dim = latent_dim
        if model_name == "clip":
            import clip

            model = clip.load("RN50")[0].visual
        elif model_name == "r3m":
            import sys

            sys.path.append("/sfs/qumulo/qhome/dcs3zc/r3m")
            import r3m

            model = r3m.load_r3m("resnet18").module.convnet
            num_features = model.layer4[-1].conv2.out_channels

        elif model_name == "dino2":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        elif model_name == "mvp":
            import mvp

            model = mvp.load("vitb-mae-egosoup")
        elif model_name == "liv":
            from model.vision.liv import load_liv

            model = load_liv().module.model.visual
        elif model_name == "e2e":
            model = torchvision.models.resnet18(pretrained=False)
            num_features = model.layer4[-1].conv2.out_channels
            model.fc = nn.Identity()

        else:
            raise ValueError("vision model name not recognized")

        ###
        # convert_weights(model)
        ###
        model = model.float().to(memory_format=torch.channels_last)

        self.encoders = nn.ModuleDict()
        for view in views:
            self.encoders[view] = copy.deepcopy(model)

            if obs_encoder_group_norm:
                self.encoders[view] = replace_bn_with_gn(self.encoders[view])
                print(self.encoders[view].bn1)
            if spatial_softmax:
                self.encoders[view] = replace_avgpool_with_spatial_softmax(
                    self.encoders[view], self.latent_dim
                )
            if frozen:
                self.encoders[view].eval()
                self.encoders[view].requires_grad_(False)

    def forward(self, x, view):
        assert view in self.views
        if x.ndim == 4:
            out = self.encoders[view](x)
        elif x.ndim == 5:
            out = self.encoders[view](x.flatten(end_dim=1))
            out = out.unflatten(0, x.shape[:2])
        else:
            raise ValueError("Input tensor must have 4 or 5 dimensions")

        return out


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


class RMSNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, num_channels, 1, 1))  # Scale parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS across spatial dimensions (H, W) but keep per-channel normalization
        norm = (
            torch.norm(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3]) ** 0.5
        )
        return x / norm.clamp(min=self.eps) * self.g


def replace_bn_with_gn(
    root_module: nn.Module, features_per_group: int = 8
) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        # func=lambda x: nn.GroupNorm(num_groups=x.num_features//features_per_group, num_channels=x.num_features)
        # func=lambda x: nn.InstanceNorm2d(x.num_features)
        func=lambda x: RMSNorm2d(x.num_features),
    )
    return root_module


def replace_avgpool_with_spatial_softmax(model: nn.Module, latent_dim) -> nn.Module:
    # Create an instance of SpatialSoftArgmax
    spatial_softmax = SpatialSoftArgmax(temperature=None, normalise=True)

    # Replace AdaptiveAvgPool2d with SpatialSoftArgmax
    model = replace_submodules(
        root_module=model,
        predicate=lambda x: isinstance(x, nn.AdaptiveAvgPool2d),
        func=lambda x: spatial_softmax,
    )

    return model


class CoordinateUtils(object):
    @staticmethod
    def get_image_coordinates(h, w, normalise):
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        return image_x, image_y


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = (
            nn.Parameter(torch.ones(1))
            if temperature is None
            else torch.tensor([temperature])
        )
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(
            x.reshape(n * c, h * w) / self.temperature, dim=1
        )
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        image_x, image_y = CoordinateUtils.get_image_coordinates(
            h, w, normalise=self.normalise
        )
        # size (H, W, 2)
        image_coordinates = torch.cat(
            (image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1
        )
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out
