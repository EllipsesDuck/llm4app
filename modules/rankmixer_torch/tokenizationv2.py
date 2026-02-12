# -*- coding: utf-8 -*-
import math
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trunc_normal_(t: torch.Tensor, mean: float = 0.0, std: float = 0.36, a: float = -2.0, b: float = 2.0):
    """
    TF truncated_normal_initializer roughly clips at ±2 std by default.
    Here a/b are in std units (like TF behavior intent).
    """
    aa = a * std + mean
    bb = b * std + mean
    return nn.init.trunc_normal_(t, mean=mean, std=std, a=aa, b=bb)


class SemanticTokenizer(nn.Module):
    """
    PyTorch version of TF1 SemanticTokenizer.

    - token allocation: floor + remainder, strictly sum == total_tokens
    - scope safety: TF variable_scope/AUTO_REUSE -> PyTorch caches Linear layers by input-dim signature
    """

    def __init__(
        self,
        d_model: int,
        embedding_dim: int,
        total_tokens: int = 64,
        min_tokens_per_group: int = 1,
        name: str = "semantic_tokenizer",
        experiment_tag: Optional[str] = None,
        isolate_by_config: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.embedding_dim = int(embedding_dim)
        self.total_tokens = int(total_tokens)
        self.min_tokens_per_group = int(min_tokens_per_group)
        self.base_name = str(name)
        self.experiment_tag = None if experiment_tag is None else str(experiment_tag)
        self.isolate_by_config = bool(isolate_by_config)

        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")
        if self.min_tokens_per_group < 0:
            raise ValueError("min_tokens_per_group must be >= 0")

        self.name = self._make_safe_scope_name()
        self._built_signature: Optional[Tuple[int, int, int]] = None

        # Per-group projection layer cache keyed by input_dim signature
        # (TF used name proj_in{in}_out{d_model} under group scope)
        self.user_proj = nn.ModuleDict()
        self.aud_proj = nn.ModuleDict()
        self.shop_proj = nn.ModuleDict()

        # store latest allocation info if you want to inspect
        self.last_alloc: Optional[Tuple[int, int, int]] = None

    def _make_safe_scope_name(self) -> str:
        parts = [self.base_name]
        if self.experiment_tag:
            parts.append(self.experiment_tag)
        if self.isolate_by_config:
            parts.append(f"T{self.total_tokens}")
            parts.append(f"Dm{self.d_model}")
            parts.append(f"De{self.embedding_dim}")
            parts.append(f"Min{self.min_tokens_per_group}")
        return "/".join(parts)

    def _alloc_3_floor_remainder(self, n1: int, n2: int, n3: int) -> Tuple[int, int, int]:
        """
        Allocate total_tokens to 3 groups by proportional sizes with floor + remainder.
        Strictly guarantee sum == T.
        Constraint: each group >= m if feasible (m*3 <= T), else degrade to near-even split.
        """
        T = self.total_tokens
        m = self.min_tokens_per_group

        if m * 3 > T:
            base = T // 3
            rem = T - 3 * base
            a = [base, base, base]
            for i in range(rem):
                a[i] += 1
            return a[0], a[1], a[2]

        sizes = [max(0, int(n1)), max(0, int(n2)), max(0, int(n3))]
        s = sum(sizes)

        if s == 0:
            base = T // 3
            rem = T - 3 * base
            a = [base, base, base]
            for i in range(rem):
                a[i] += 1
            for i in range(3):
                if a[i] < m:
                    a[i] = m
            while sum(a) > T:
                k = max(range(3), key=lambda i: a[i])
                if a[k] > m:
                    a[k] -= 1
            while sum(a) < T:
                k = min(range(3), key=lambda i: a[i])
                a[k] += 1
            return a[0], a[1], a[2]

        T_left = T - 3 * m
        raw = [sizes[i] * float(T_left) / float(s) for i in range(3)]
        base = [int(x) for x in raw]
        rem = T_left - sum(base)

        frac = [(i, raw[i] - float(base[i])) for i in range(3)]
        frac.sort(key=lambda x: x[1], reverse=True)
        for k in range(rem):
            base[frac[k % 3][0]] += 1

        a = [m + base[i] for i in range(3)]
        assert sum(a) == T, "allocation bug: sum != total_tokens"
        return a[0], a[1], a[2]

    def _get_or_create_proj(self, group_dict: nn.ModuleDict, in_dim: int) -> nn.Linear:
        """
        Cache per (group, in_dim) projection layer:
          Linear(in_dim -> d_model)
        """
        key = f"in{in_dim}_out{self.d_model}"
        if key not in group_dict:
            layer = nn.Linear(in_dim, self.d_model, bias=True)

            # Match TF initializer choices in your dense:
            # kernel: truncated_normal(mean=0, std=0.36), bias: truncated_normal(std=0.001)
            _trunc_normal_(layer.weight, mean=0.0, std=0.36, a=-2.0, b=2.0)
            _trunc_normal_(layer.bias, mean=0.0, std=0.001, a=-2.0, b=2.0)

            group_dict[key] = layer
        return group_dict[key]

    def _chunk_project(self, x_3d: torch.Tensor, Tg: int, group: str) -> torch.Tensor:
        """
        x_3d: (B, Ng, De) -> (B, Tg, Dm)

        chunk scheme:
          token_size = ceil(Ng / Tg)
          pad to Tg*token_size
          reshape to (B, Tg, token_size*De)
          linear projection to d_model
        """
        Tg = max(1, int(Tg))
        B, Ng, De = x_3d.shape
        if De != self.embedding_dim:
            raise ValueError(f"embedding_dim mismatch: expected {self.embedding_dim}, got {De}")

        if Ng == 0:
            return x_3d.new_zeros((B, Tg, self.d_model))

        token_size = int((Ng + Tg - 1) / Tg)
        pad_needed = Tg * token_size - Ng
        if pad_needed > 0:
            pad = x_3d.new_zeros((B, pad_needed, De))
            x_3d = torch.cat([x_3d, pad], dim=1)

        flat = x_3d.reshape(B, Tg, token_size * De)
        in_dim = token_size * De

        if group == "user":
            proj = self._get_or_create_proj(self.user_proj, in_dim)
        elif group == "audience":
            proj = self._get_or_create_proj(self.aud_proj, in_dim)
        elif group == "shop":
            proj = self._get_or_create_proj(self.shop_proj, in_dim)
        else:
            raise ValueError(f"Unknown group: {group}")

        y = proj(flat)  # (B, Tg, d_model)
        return y

    @staticmethod
    def _stack_list(lst: Optional[List[torch.Tensor]], list_name: str, embedding_dim: int) -> Optional[torch.Tensor]:
        if not lst:
            return None
        x0 = lst[0]
        if x0.dim() != 2:
            raise ValueError(f"{list_name} elements must be [B, D], got dim={x0.dim()}")
        if x0.shape[1] != embedding_dim:
            raise ValueError(f"{list_name} element D mismatch: expected {embedding_dim}, got {x0.shape[1]}")

        for i, t in enumerate(lst[1:], start=1):
            if t.dim() != 2:
                raise ValueError(f"{list_name}[{i}] must be [B, D], got dim={t.dim()}")
            if t.shape[1] != embedding_dim:
                raise ValueError(f"{list_name}[{i}] D mismatch: expected {embedding_dim}, got {t.shape[1]}")
            if t.shape[0] != x0.shape[0]:
                raise ValueError(f"{list_name}[{i}] batch mismatch: expected {x0.shape[0]}, got {t.shape[0]}")

        # (B, Ng, D)
        return torch.stack(lst, dim=1)

    def l2_reg_loss(self, coef: float = 1e-5) -> torch.Tensor:
        """
        Optional: mimic TF kernel_regularizer=l2(1e-5) only for these proj layers.
        If you already use optimizer weight_decay, you can ignore this.
        """
        reg = None
        for md in (self.user_proj, self.aud_proj, self.shop_proj):
            for layer in md.values():
                w = layer.weight
                term = (w * w).sum()
                reg = term if reg is None else (reg + term)
        if reg is None:
            # no layers created yet
            return torch.tensor(0.0)
        return coef * reg

    def tokenize(
        self,
        user_emb_list: Optional[List[torch.Tensor]],
        aud_emb_list: Optional[List[torch.Tensor]],
        shop_emb_list: Optional[List[torch.Tensor]],
        enforce_static_signature: bool = True,
    ) -> Tuple[torch.Tensor, int, Tuple[int, int, int]]:
        """
        Returns:
          tokens: (B, total_T, d_model)
          total_T: int
          (Tu, Ta, Ts): tuple[int,int,int]
        """
        U = self._stack_list(user_emb_list, "user_emb_list", self.embedding_dim)
        A = self._stack_list(aud_emb_list, "aud_emb_list", self.embedding_dim)
        S = self._stack_list(shop_emb_list, "shop_emb_list", self.embedding_dim)

        ref = U if U is not None else (A if A is not None else S)
        if ref is None:
            raise ValueError("All groups are empty; need at least one embedding list non-empty.")
        B = ref.shape[0]
        device = ref.device
        dtype = ref.dtype

        nU = len(user_emb_list) if user_emb_list else 0
        nA = len(aud_emb_list) if aud_emb_list else 0
        nS = len(shop_emb_list) if shop_emb_list else 0

        if enforce_static_signature:
            sig = (nU, nA, nS)
            if self._built_signature is None:
                self._built_signature = sig
            elif self._built_signature != sig:
                raise ValueError(
                    f"Tokenizer group sizes changed across calls: prev={self._built_signature} now={sig}. "
                    "This can create different projection layers (input dims change). "
                    "Use a different tokenizer name/experiment_tag or disable enforce_static_signature."
                )

        Tu, Ta, Ts = self._alloc_3_floor_remainder(nU, nA, nS)
        self.last_alloc = (Tu, Ta, Ts)

        # If a group is empty, return zeros like TF
        u_tok = self._chunk_project(U, Tu, "user") if U is not None else torch.zeros((B, Tu, self.d_model), device=device, dtype=dtype)
        a_tok = self._chunk_project(A, Ta, "audience") if A is not None else torch.zeros((B, Ta, self.d_model), device=device, dtype=dtype)
        s_tok = self._chunk_project(S, Ts, "shop") if S is not None else torch.zeros((B, Ts, self.d_model), device=device, dtype=dtype)

        tokens = torch.cat([u_tok, a_tok, s_tok], dim=1)  # (B, total_T, d_model)
        total_T = Tu + Ta + Ts
        return tokens, total_T, (Tu, Ta, Ts)
