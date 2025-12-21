import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import kmeans, sinkhorn_algorithm


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,                # codebook size
        e_dim,              # embedding dim
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.003,
        sk_iters=100,
    ):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        if not self.kmeans_init:
            self.initted = True
            bound = 1.0 / (self.n_e ** 0.5)
            nn.init.uniform_(self.embedding.weight, -bound, bound)
        else:
            self.initted = False
            nn.init.zeros_(self.embedding.weight)

        self.register_buffer("emb_sq", None)

    def _update_emb_sq(self):
        self.emb_sq = (self.embedding.weight ** 2).sum(dim=1)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        return z_q.view(shape) if shape is not None else z_q

    @torch.no_grad()
    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)  # (K, D)
        self.embedding.weight.copy_(centers)
        self.initted = True
        self._update_emb_sq()

    @staticmethod
    def normalize_distances(d):
        mean = d.mean(dim=1, keepdim=True)
        std = d.std(dim=1, keepdim=True).clamp(min=1e-6)
        d = (d - mean) / std
        return d

    def compute_distances(self, x):
        if self.emb_sq is None:
            self._update_emb_sq()

        # (N, 1)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)

        # (1, K)
        emb_sq = self.emb_sq.unsqueeze(0)

        # (N, K)
        cross = x @ self.embedding.weight.T

        # dist^2 = ||x||^2 + ||e||^2 - 2 x·e
        d = x_sq + emb_sq - 2 * cross

        return d

    def forward(self, x, use_sk=True):
        # flatten
        latent = x.reshape(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)

        d = self.compute_distances(latent)   # (N, K)

        if not use_sk or self.sk_epsilon <= 0:
            # Hard argmin
            indices = torch.argmin(d, dim=-1)
        else:
            # Sinkhorn soft assignment
            with torch.no_grad():
                d_norm = self.normalize_distances(d)
                Q = sinkhorn_algorithm(d_norm, self.sk_epsilon, self.sk_iters)

                if torch.isnan(Q).any() or torch.isinf(Q).any():
                    print("[Warning] Sinkhorn returns nan/inf")

                indices = torch.argmax(Q, dim=-1)

        x_q = self.embedding(indices).reshape(x.shape)

        codebook_loss = F.mse_loss(x_q, x.detach())
        commitment_loss = F.mse_loss(x_q.detach(), x)
        loss = codebook_loss + self.beta * commitment_loss

        x_q = x + (x_q - x).detach()

        indices = indices.reshape(x.shape[:-1])

        return x_q, loss, indices

    def quantize(self, x, use_sinkhorn=False):
        return self.forward(x, use_sk=use_sinkhorn)[0]


