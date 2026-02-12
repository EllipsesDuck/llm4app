# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_ffn_weight_


def gelu(x: torch.Tensor) -> torch.Tensor:
    # TF1 tanh-approx GELU (same formula)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


class PerTokenFFN(nn.Module):
    """
    Per-token FFN with independent parameters for each token.

    x: (B, T, D) -> y: (B, T, D)

    Dropout follows PyTorch standard:
      - enabled when model.train()
      - disabled when model.eval()

    Backward-compatible option:
      - forward(x, training=True/False) if you still pass it somewhere;
        if provided, it overrides self.training for that call.
    """

    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0, name=None):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = float(dropout)
        self.name = name  # unused, kept for interface familiarity

        hidden_dim = self.d_model * self.mult

        # Per-token parameters
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))

        self.reset_parameters()

    def reset_parameters(self):
        # Approximate TF variance_scaling(scale=2.0) with Kaiming init
        init_ffn_weight_(self.W1)
        init_ffn_weight_(self.W2)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            training: optional bool. If None -> use self.training (PyTorch standard).
                      If provided -> overrides self.training for this call.
        """
        # decide training mode for dropout
        use_train = self.training if training is None else bool(training)

        # (B,T,D) @ (T,D,H) -> (B,T,H), token-wise independent weights
        h = torch.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=use_train)

        # (B,T,H) @ (T,H,D) -> (B,T,D)
        y = torch.einsum("bth,thd->btd", h, self.W2) + self.b2
        if self.dropout:
            y = F.dropout(y, p=self.dropout, training=use_train)
        return y
