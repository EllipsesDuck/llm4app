import torch
import torch.nn as nn
from torch.nn import init
from dataclasses import dataclass
from typing import Optional, Sequence, Union, Type

from sklearn.cluster import KMeans  


def get_activation(
    activation: Optional[Union[str, Type[nn.Module], nn.Module]],
) -> Optional[nn.Module]:
    if activation is None:
        return None

    if isinstance(activation, nn.Module):
        return activation

    if isinstance(activation, str):
        name = activation.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "leakyrelu":
            return nn.LeakyReLU()
        elif name in ("none", "identity"):
            return None
        else:
            raise ValueError(f"Unsupported activation name: {activation}")

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation()

    raise TypeError(f"Unsupported activation type: {type(activation)}")

class MLPLayers(nn.Module):
    def __init__(
        self,
        layers: Sequence[int],
        dropout: float = 0.0,
        hidden_activation: Union[str, Type[nn.Module], nn.Module] = "relu",
        out_activation: Optional[Union[str, Type[nn.Module], nn.Module]] = None,
        use_bn: bool = False,
        use_ln: bool = False,
        init_method: str = "xavier_normal",
    ):
        super().__init__()

        if len(layers) < 2:
            raise ValueError(f"`layers` length must be >= 2, got {len(layers)}")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"`dropout` must be in [0,1], got {dropout}")
        if use_bn and use_ln:
            raise ValueError("Cannot use both BatchNorm and LayerNorm at the same time.")

        self.layers = list(layers)
        self.dropout_p = dropout
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.init_method = init_method

        hidden_act = get_activation(hidden_activation)
        out_act = get_activation(out_activation)

        mlp_modules = []
        num_layers = len(self.layers) - 1

        for idx, (in_dim, out_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            is_last = (idx == num_layers - 1)
            # linear
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            # norm
            if not is_last:
                if self.use_bn:
                    mlp_modules.append(nn.BatchNorm1d(out_dim))
                elif self.use_ln:
                    mlp_modules.append(nn.LayerNorm(out_dim))

            # activation
            act = out_act if is_last else hidden_act
            if act is not None:
                mlp_modules.append(act)

            # dropout
            if self.dropout_p > 0.0 and (not is_last):
                mlp_modules.append(nn.Dropout(p=self.dropout_p))

        self.mlp = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if self.init_method == "xavier_normal":
                init.xavier_normal_(module.weight)
            elif self.init_method == "xavier_uniform":
                init.xavier_uniform_(module.weight)
            elif self.init_method == "kaiming_normal":
                init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif self.init_method == "none":
                pass
            else:
                raise ValueError(f"Unsupported init_method: {self.init_method}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

# sklearn version
class KMeansClustering:
    def __init__(self, num_clusters: int, max_iter: int = 10):
        if num_clusters <= 0:
            raise ValueError(f"`num_clusters` must be > 0, got {num_clusters}")
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self._centers: Optional[torch.Tensor] = None

    @property
    def centers(self) -> torch.Tensor:
        if self._centers is None:
            raise RuntimeError("KMeansClustering has not been fit yet.")
        return self._centers

    @classmethod
    def from_sklearn(cls, num_clusters: int, max_iter: int = 10) -> "KMeansClustering":
        return cls(num_clusters=num_clusters, max_iter=max_iter)

    def fit(self, samples: torch.Tensor) -> "KMeansClustering":
        """
        samples: (B, D)
        """
        if samples.dim() != 2:
            raise ValueError(f"`samples` must be 2D (B, D), got shape {samples.shape}")
        B, D = samples.shape
        if B < self.num_clusters:
            raise ValueError(
                f"Number of samples B={B} must be >= num_clusters={self.num_clusters}"
            )

        device = samples.device
        dtype = samples.dtype

        x_np = samples.detach().cpu().numpy()

        kmeans = KMeans(n_clusters=self.num_clusters, max_iter=self.max_iter)
        kmeans.fit(x_np)

        centers_np = kmeans.cluster_centers_  # (K, D)
        centers = torch.from_numpy(centers_np).to(device=device, dtype=dtype)
        self._centers = centers
        return self

    def predict(self, samples: torch.Tensor) -> torch.Tensor:
        if self._centers is None:
            raise RuntimeError("You must call `fit` before `predict`.")

        if samples.dim() != 2:
            raise ValueError(f"`samples` must be 2D (B, D), got shape {samples.shape}")

        x = samples
        centers = self._centers

        x_norm2 = (x ** 2).sum(dim=1, keepdim=True)           # (B, 1)
        c_norm2 = (centers ** 2).sum(dim=1, keepdim=True).T   # (1, K)
        sim = x @ centers.T                                   # (B, K)
        dist2 = x_norm2 + c_norm2 - 2 * sim                   # (B, K)

        labels = dist2.argmin(dim=1)
        return labels

    def fit_predict(self, samples: torch.Tensor) -> torch.Tensor:
        self.fit(samples)
        return self.predict(samples)


# torch version,can acclearte with gpu
# class KMeansClustering:
#     def __init__(self, num_clusters: int, max_iter: int = 10, tol: float = 1e-4):
#         if num_clusters <= 0:
#             raise ValueError(f"`num_clusters` must be > 0, got {num_clusters}")
#         self.num_clusters = num_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self._centers: Optional[torch.Tensor] = None

#     @property
#     def centers(self) -> torch.Tensor:
#         if self._centers is None:
#             raise RuntimeError("KMeansClustering has not been fit yet.")
#         return self._centers

#     @classmethod
#     def from_sklearn(cls, num_clusters: int, max_iter: int = 10) -> "KMeansClustering":
#         return cls(num_clusters=num_clusters, max_iter=max_iter)

#     def fit(self, samples: torch.Tensor) -> "KMeansClustering":
#         if samples.dim() != 2:
#             raise ValueError(f"`samples` must be 2D (B,D), got shape {samples.shape}")

#         B, D = samples.shape
#         if B < self.num_clusters:
#             raise ValueError(
#                 f"Number of samples B={B} must be >= num_clusters={self.num_clusters}"
#             )

#         device = samples.device
#         dtype = samples.dtype

#         indices = torch.randperm(B, device=device)[:self.num_clusters]
#         centers = samples[indices]  # (K, D)

#         for it in range(self.max_iter):
#             # dist^2 = ||x||^2 + ||c||^2 - 2x·c
#             x_norm2 = (samples ** 2).sum(dim=1, keepdim=True)         # (B, 1)
#             c_norm2 = (centers ** 2).sum(dim=1).unsqueeze(0)          # (1, K)
#             sim = samples @ centers.T                                 # (B, K)
#             dist2 = x_norm2 + c_norm2 - 2 * sim                       # (B, K)

#             labels = dist2.argmin(dim=1)                              # (B,)

#             new_centers = torch.zeros_like(centers)

#             for k in range(self.num_clusters):
#                 mask = (labels == k)
#                 if mask.any():
#                     new_centers[k] = samples[mask].mean(dim=0)
#                 else:
#                     new_centers[k] = samples[torch.randint(0, B, (1,), device=device)]

#             shift = (centers - new_centers).pow(2).sum()
#             centers = new_centers

#             if shift < self.tol:
#                 break

#         self._centers = centers.to(dtype=dtype, device=device)
#         return self

#     def predict(self, samples: torch.Tensor) -> torch.Tensor:
#         if self._centers is None:
#             raise RuntimeError("You must call `fit` before `predict`.")

#         if samples.dim() != 2:
#             raise ValueError(f"`samples` must be 2D (B,D), got {samples.shape}")

#         x_norm2 = (samples ** 2).sum(dim=1, keepdim=True)            # (B, 1)
#         c_norm2 = (self._centers ** 2).sum(dim=1).unsqueeze(0)       # (1, K)
#         sim = samples @ self._centers.T                              # (B, K)
#         dist2 = x_norm2 + c_norm2 - 2 * sim                          # (B, K)

#         labels = dist2.argmin(dim=1)
#         return labels

#     def fit_predict(self, samples: torch.Tensor) -> torch.Tensor:
#         self.fit(samples)
#         return self.predict(samples)

def kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
) -> torch.Tensor:
    km = KMeansClustering.from_sklearn(num_clusters=num_clusters, max_iter=num_iters)
    km.fit(samples)
    return km.centers


@dataclass
class SinkhornConfig:
    epsilon: float = 0.05
    num_iters: int = 3
    clamp_min: float = 1e-8  


class SinkhornKnopp(nn.Module):
    def __init__(self, config: SinkhornConfig):
        super().__init__()
        if config.epsilon <= 0.0:
            raise ValueError(f"`epsilon` must be > 0, got {config.epsilon}")
        if config.num_iters <= 0:
            raise ValueError(f"`num_iters` must be > 0, got {config.num_iters}")

        self.config = config

    @torch.no_grad()
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.dim() != 2:
            raise ValueError(f"`distances` must be 2D (B, K), got {distances.shape}")

        eps = self.config.epsilon
        B, K = distances.shape

        Q = torch.exp(-distances / eps)  # (B, K)

        sum_Q = Q.sum()
        if sum_Q <= 0:
            raise RuntimeError("Sum of Q is non-positive, check distances / epsilon.")
        Q = Q / sum_Q

        for _ in range(self.config.num_iters):
            # normalize columns: sum over rows = 1/B
            col_sums = Q.sum(dim=0, keepdim=True)  # (1, K)
            Q = Q / (col_sums + 1e-8)
            Q = Q / B

            # normalize rows: sum over cols = 1/K
            row_sums = Q.sum(dim=1, keepdim=True)  # (B, 1)
            Q = Q / (row_sums + 1e-8)
            Q = Q / K

        Q = Q * B

        if self.config.clamp_min is not None:
            Q = torch.clamp(Q, min=self.config.clamp_min)

        return Q


@torch.no_grad()
def sinkhorn_algorithm(
    distances: torch.Tensor,
    epsilon: float,
    sinkhorn_iterations: int,
) -> torch.Tensor:
    cfg = SinkhornConfig(epsilon=epsilon, num_iters=sinkhorn_iterations)
    sinkhorn = SinkhornKnopp(cfg)
    return sinkhorn(distances)



