# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_ffn_weight_


def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


class PerTokenSparseMoE(nn.Module):
    """
    Per-token Sparse MoE with top-k sparse routing + optional DTSI.

    Inputs:
      x: (B, T, D)

    Outputs:
      y: (B, T, D)
      aux_loss: scalar tensor  (l1_loss + lb_coef * lb_loss)
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        mult=4,
        num_experts=4,
        dropout=0.0,
        l1_coef=0.0,
        sparsity_ratio=1.0,
        use_dtsi=True,
        routing_type="relu_dtsi",
        name=None,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)

        # L1 sparsity (your original behavior)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0

        # Load-balancing (new): you can override after init: moe.lb_coef = 0.01
        self.lb_coef = 0.01  # common small value; set 0.0 to disable

        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()
        self.name = name

        hidden_dim = self.d_model * self.mult

        # Experts: (T, E, D, H) etc.
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.d_model))

        # Routers: (T, D, E)
        self.gate_w_train = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
        self.gate_b_train = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))

        if self.use_dtsi:
            self.gate_w_infer = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
            self.gate_b_infer = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))
        else:
            self.gate_w_infer = None
            self.gate_b_infer = None

        self.reset_parameters()

    def reset_parameters(self):
        init_ffn_weight_(self.W1)
        init_ffn_weight_(self.W2)
        init_ffn_weight_(self.gate_w_train)
        if self.use_dtsi:
            init_ffn_weight_(self.gate_w_infer)

    def _router_logits(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,tde->bte", x, w) + b  # (B,T,E)

    def _k_from_ratio(self) -> int:
        r = self.sparsity_ratio
        if r <= 0:
            k = 1
        else:
            k = int(math.ceil(r * self.num_experts))
        return max(1, min(self.num_experts, k))

    def _topk_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: (B,T,E) -> mask bool (B,T,E)
        _, idx = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter_(-1, idx, True)

    def _apply_topk_softmax(self, logits: torch.Tensor, k: int):
        """
        returns:
          gate: (B,T,E) nonzero on top-k, sums to 1 over experts per token
          mask: (B,T,E) bool top-k mask
        """
        mask = self._topk_mask(logits, k)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        gate = F.softmax(masked_logits, dim=-1)
        gate = torch.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
        return gate, mask

    def _apply_topk_relu(self, logits: torch.Tensor, k: int):
        """
        returns:
          gate: (B,T,E) nonzero on top-k after ReLU (unnormalized)
          mask: (B,T,E) bool top-k mask (based on ReLU activations)
        """
        a = F.relu(logits)
        mask = self._topk_mask(a, k)
        gate = a * mask.to(a.dtype)
        return gate, mask

    def _load_balance_loss(self, gate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Switch-style load balancing:
          importance[e] = sum_{tokens} gate[..., e]
          load[e]       = count_{tokens} mask[..., e]
        Normalize both to sum=1 and compute E * sum(importance * load)

        gate: (B,T,E)
        mask: (B,T,E) bool
        """
        B, T, E = gate.shape
        # importance (soft usage)
        importance = gate.sum(dim=(0, 1))  # (E,)
        # load (hard assignment count)
        load = mask.to(gate.dtype).sum(dim=(0, 1))  # (E,)

        # normalize to distributions
        importance = importance / (importance.sum().clamp_min(1e-9))
        load = load / (load.sum().clamp_min(1e-9))

        lb = E * torch.sum(importance * load)
        return lb

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        if T != self.num_tokens:
            raise ValueError(f"Expected num_tokens(T)={self.num_tokens}, but got T={T}")

        k = self._k_from_ratio()

        # ---- experts ----
        h = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        expert_out = torch.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout:
            expert_out = F.dropout(expert_out, p=self.dropout, training=self.training)

        # ---- logits ----
        train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        infer_logits = (
            self._router_logits(x, self.gate_w_infer, self.gate_b_infer) if self.use_dtsi else train_logits
        )

        # ---- gates + masks ----
        if self.routing_type == "relu_dtsi":
            gate_train, mask_train = self._apply_topk_softmax(train_logits, k=k)
            gate_infer, mask_infer = self._apply_topk_relu(infer_logits, k=k)
        elif self.routing_type == "relu":
            gate_train, mask_train = self._apply_topk_relu(train_logits, k=k)
            gate_infer, mask_infer = self._apply_topk_relu(infer_logits, k=k)
        else:
            raise ValueError(f"Unsupported routing_type: {self.routing_type}")

        gate = gate_train if self.training else gate_infer
        y = torch.sum(expert_out * gate.unsqueeze(-1), dim=2)  # (B,T,D)

        # ---- L1 sparsity (uses infer gate like your TF code) ----
        if self.l1_coef > 0.0:
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            l1_loss = (self.l1_coef * scale) * gate_infer.sum(dim=-1).mean()
        else:
            l1_loss = x.new_zeros(())

        # ---- Load balancing (only meaningful during training) ----
        if (self.lb_coef > 0.0) and self.training:
            lb_loss = self._load_balance_loss(gate_train, mask_train)
            aux_loss = l1_loss + (self.lb_coef * lb_loss)
        else:
            aux_loss = l1_loss

        return y, aux_loss
