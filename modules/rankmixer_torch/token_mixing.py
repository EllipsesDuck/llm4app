# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParameterFreeTokenMixer(nn.Module):
    """
    Paper-style parameter-free token mixing with strict H = T.

    Input:
        x : (B, T, D)

    Output:
        mixed : (B, T, D)

    No learnable parameters.
    Dropout follows PyTorch standard (model.train()/eval()).
    """

    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0, name=None):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = float(dropout)
        self.name = name  # kept only for compatibility / logging

        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model} "
                f"num_heads={self.num_heads}"
            )

        self.d_head = self.d_model // self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape

        if T != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, but got T={T}")
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, but got D={D}")

        H = self.num_heads
        Dh = self.d_head

        # ---- TF logic reproduced exactly ----
        # reshape -> (B, T, H, Dh)
        split = x.view(B, T, H, Dh)

        # transpose token/head -> (B, H, T, Dh)
        shuffled = split.transpose(1, 2).contiguous()

        # merge token into channel -> (B, H, T*Dh) == (B, H, D)
        merged = shuffled.view(B, H, T * Dh)

        # reshape back -> (B, T, D)
        mixed = merged.view(B, T, self.d_model)

        if self.dropout:
            mixed = F.dropout(mixed, p=self.dropout, training=self.training)

        return mixed
