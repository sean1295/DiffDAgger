import torch
import scipy.interpolate as interpolate


class Normalizer:
    """
    parent class, subclass by defining the `normalize` and `unnormalize` methods
    """

    def __init__(self, X):
        self.X = X.cpu()
        self.mins = self.X.min(dim=0)[0]
        self.maxs = self.X.max(dim=0)[0]
        # self.to_device(device)

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size()}\n    -: """
            f"""{self.mins}\n    +: {self.maxs}\n"""
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, x):
        device = self.mins.device
        new_x = torch.cat((self.X, x.cpu()), dim=0)
        self.__init__(new_x)
        self.to_device(device)

    def to_device(self, device):
        self.device = device
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)


class DebugNormalizer(Normalizer):
    """
    identity function
    """

    def normalize(self, x, *args, **kwargs):
        return x

    def unnormalize(self, x, *args, **kwargs):
        return x


class GaussianNormalizer(Normalizer):
    """
    normalizes to zero mean and unit variance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(dim=0)[0]
        self.stds = self.X.std(dim=0)[0]
        self.z = 1

    def __repr__(self):
        return (
            f"""[ Normalizer ] dim: {self.mins.size()}\n    """
            f"""means: {self.means}\n    """
            f"""stds: {self.z * self.stds}\n"""
        )

    def normalize(self, x):
        return (x - self.means) / self.stds

    def unnormalize(self, x):
        return x * self.stds + self.means

    def to_device(self, device):
        self.device = device
        self.means = self.means.to(device)
        self.stds = self.stds.to(device)


class LimitsNormalizer(Normalizer):
    """
    maps [ xmin, xmax ] to [ -1, 1 ]
    """

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]s
        """
        if x.max() > 1 + eps or x.min() < -1 - eps:
            # print(f'[ datasets/mujoco ] Warning: sample out of range | ({x.min():.4f}, {x.max():.4f})')
            x = torch.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0

        return x * (self.maxs - self.mins) + self.mins


class SafeLimitsNormalizer(LimitsNormalizer):
    """
    functions like LimitsNormalizer, but can handle data for which a dimension is constant
    """

    def __init__(self, *args, eps=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(len(self.mins)):
            if self.mins[i] == self.maxs[i]:
                self.mins -= eps
                self.maxs += eps


def get_vision_normalizer(device="cpu"):
    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    # Create normalizer instance with dummy data
    normalizer = GaussianNormalizer(torch.rand(1, 1))

    normalizer.means = (torch.Tensor(_MEAN) * 255.0).to(device)
    normalizer.stds = (torch.Tensor(_STD) * 255.0).to(device)
    return normalizer


import scipy.interpolate as interpolate
import numpy as np


class CDFNormalizer(Normalizer):
    """
    Makes training data uniform (over each dimension) by transforming it with marginal CDFs
    """

    def __init__(self, X):
        super().__init__(atleast_2d(X))
        self.dim = self.X.shape[1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def __repr__(self):
        return f"[ CDFNormalizer ] dim: {self.mins.size(0)}\n" + "    |    ".join(
            f"{i:3d}: {cdf}" for i, cdf in enumerate(self.cdfs)
        )

    def wrap(self, fn_name, x):
        shape = x.shape
        # reshape to 2d
        x = x.reshape(-1, self.dim)
        out = torch.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap("normalize", x)

    def unnormalize(self, x):
        return self.wrap("unnormalize", x)

    def to_device(self, device):
        super().to_device(device)
        for cdf in self.cdfs:
            cdf.to_device(device)


class CDFNormalizer1d:
    """
    CDF normalizer for a single dimension
    """

    def __init__(self, X):
        assert X.dim() == 1
        self.X = X.float()
        if self.X.max() == self.X.min():
            self.constant = True
        else:
            self.constant = False
            quantiles, cumprob = empirical_cdf(self.X)
            self.fn = interpolate.CubicSpline(quantiles, cumprob)
            self.inv = interpolate.CubicSpline(cumprob, quantiles)
            self.xmin, self.xmax = quantiles.min(), quantiles.max()
            self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def __repr__(self):
        return f"[{self.xmin:.4f}, {self.xmax:.4f}]"

    def normalize(self, x):
        if self.constant:
            return x
        x = torch.clamp(x, self.xmin, self.xmax)
        # [ 0, 1 ]
        y = torch.tensor(self.fn(x.cpu().numpy())).to(x.device)
        # [ -1, 1 ]
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        """
        x : [ -1, 1 ]
        """
        if self.constant:
            return x
        # [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.0
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            print(
                f"[ dataset/normalization ] Warning: out of range in unnormalize: "
                f"[{x.min().item()}, {x.max().item()}] | "
                f"x : [{self.xmin}, {self.xmax}] | "
                f"y: [{self.ymin}, {self.ymax}]"
            )
        x = torch.clamp(x, self.ymin, self.ymax)
        y = torch.tensor(self.inv(x.cpu().numpy())).to(x.device)
        return y

    def to_device(self, device):
        self.X = self.X.to(device)


def empirical_cdf(sample):
    # Convert to numpy for unique and cumsum operations
    sample_np = sample.cpu().numpy()
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample_np, return_counts=True)
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.float64) / sample_np.size
    return quantiles, cumprob


def atleast_2d(x):
    if x.dim() < 2:
        x = x.unsqueeze(1)
    return x


# import scipy.interpolate as interpolate
# import numpy as np
# import torch
# from scipy.stats import gaussian_kde

# class CDFNormalizer(Normalizer):
#     '''
#     Makes training data uniform (over each dimension) by transforming it with marginal CDFs
#     '''
#     def __init__(self, X, tolerance=1e-6):
#         super().__init__(atleast_2d(X))
#         self.dim = self.X.shape[1]
#         self.tolerance = tolerance
#         self.cdfs = [CDFNormalizer1d(self.X[:, i], tolerance=self.tolerance) for i in range(self.dim)]

#     def __repr__(self):
#         return f'[ CDFNormalizer ] dim: {self.mins.size(0)}\n' + '    |    '.join(
#             f'{i:3d}: {cdf}' for i, cdf in enumerate(self.cdfs)
#         )

#     def wrap(self, fn_name, x):
#         shape = x.shape
#         x = x.reshape(-1, self.dim)
#         out = torch.zeros_like(x)
#         for i, cdf in enumerate(self.cdfs):
#             fn = getattr(cdf, fn_name)
#             out[:, i] = fn(x[:, i])
#         return out.reshape(shape)

#     def normalize(self, x):
#         return self.wrap('normalize', x)

#     def unnormalize(self, x):
#         return self.wrap('unnormalize', x)

#     def to_device(self, device):
#         super().to_device(device)
#         for cdf in self.cdfs:
#             cdf.to_device(device)

# class CDFNormalizer1d:
#     '''
#     CDF normalizer for a single dimension
#     '''
#     def __init__(self, X, tolerance=1e-6, use_kde=True):
#         assert X.dim() == 1
#         self.X = X.float()
#         self.tolerance = tolerance
#         self.use_kde = use_kde

#         if torch.abs(self.X.max() - self.X.min()) < self.tolerance:
#             self.constant = True
#         else:
#             self.constant = False
#             if self.use_kde:
#                 quantiles, cumprob = kde_cdf(self.X.cpu().numpy())
#             else:
#                 quantiles, cumprob = empirical_cdf(self.X)
#             self.fn = interpolate.splrep(quantiles, cumprob, bounds_error=False, fill_value=(0, 1))
#             self.inv = interpolate.splrep(cumprob, quantiles, bounds_error=False, fill_value=(quantiles.min(), quantiles.max()))
#             self.xmin, self.xmax = quantiles.min(), quantiles.max()
#             self.ymin, self.ymax = cumprob.min(), cumprob.max()

#     def __repr__(self):
#         return f'[{self.xmin:.4f}, {self.xmax:.4f}]'

#     def normalize(self, x):
#         if self.constant:
#             return torch.zeros_like(x)  # Normalize constant values to zero
#         x = torch.clamp(x, self.xmin, self.xmax)
#         # Normalize x from its original range to [0, 1] using the CDF
#         y = torch.tensor(self.fn(x.cpu().numpy())).to(x.device)
#         # Rescale y from [0, 1] to [-1, 1]
#         y = 2 * y - 1
#         return y

#     def unnormalize(self, x, eps=1e-4):
#         '''
#         x : [-1, 1]
#         '''
#         if self.constant:
#             return torch.zeros_like(x)  # Unnormalize constant values back to zero
#         # Rescale x from [-1, 1] to [0, 1]
#         x = (x + 1) / 2.
#         if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
#             print(
#                 f'[ dataset/normalization ] Warning: out of range in unnormalize: '
#                 f'[{x.min().item()}, {x.max().item()}] | '
#                 f'x : [{self.xmin}, {self.xmax}] | '
#                 f'y: [{self.ymin}, {self.ymax}]'
#             )
#         x = torch.clamp(x, self.ymin, self.ymax)
#         # Inverse CDF: map from [0, 1] back to the original data range
#         y = torch.tensor(self.inv(x.cpu().numpy())).to(x.device)
#         return y

#     def to_device(self, device):
#         self.X = self.X.to(device)

# def kde_cdf(sample):
#     """
#     Kernel Density Estimation (KDE) based CDF for smoother distribution approximation.
#     """
#     kde = gaussian_kde(sample)
#     x_grid = np.linspace(sample.min(), sample.max(), 1000)
#     cdf = np.cumsum(kde(x_grid))
#     cdf /= cdf[-1]  # Normalize
#     return x_grid, cdf

# def empirical_cdf(sample):
#     # Convert to numpy for unique and cumsum operations
#     sample_np = sample.cpu().numpy()
#     # find the unique values and their corresponding counts
#     quantiles, counts = np.unique(sample_np, return_counts=True)
#     # take the cumulative sum of the counts and divide by the sample size to
#     # get the cumulative probabilities between 0 and 1
#     cumprob = np.cumsum(counts).astype(np.float64) / sample_np.size
#     return quantiles, cumprob

# def atleast_2d(x):
#     if x.dim() < 2:
#         x = x.unsqueeze(1)
#     return x
