import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from types import SimpleNamespace
from transformers import PreTrainedModel, PretrainedConfig, LogitsProcessor

class HierarchicalSemanticIDProcessor(LogitsProcessor):
    def __init__(
        self,
        trie,
        bos_id: int = 1,
        eos_id: int | None = None,
        force_eos_on_leaf: bool = True,
        allow_invalid_prefix: bool = False,
    ):
        self.trie = trie
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.force_eos_on_leaf = force_eos_on_leaf
        self.allow_invalid_prefix = allow_invalid_prefix

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        B, V = scores.shape
        device = scores.device

        mask = torch.full_like(scores, -float("inf"))

        for b in range(B):
            seq = input_ids[b].tolist()

            if seq and seq[0] == self.bos_id:
                prefix = seq[1:]
            else:
                prefix = seq

            allowed = self.trie.get_next_tokens(prefix)

            if allowed is None:
                if self.allow_invalid_prefix:
                    mask[b, :] = 0.0
                else:
                    if self.eos_id is not None:
                        mask[b, self.eos_id] = 0.0

            elif len(allowed) == 0:
                if self.force_eos_on_leaf and self.eos_id is not None:
                    mask[b, self.eos_id] = 0.0
                else:
                    mask[b, :] = 0.0

            else:
                mask[b, allowed] = 0.0

        return scores + mask
    

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight
    
def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, T, D = x.shape
    assert D % num_heads == 0, f"Hidden size {D} not divisible by heads {num_heads}"
    Dh = D // num_heads
    return x.view(B, T, num_heads, Dh).transpose(1, 2).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * Dh)


def _causal_additive_mask(Tq: int, Tk: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    full = torch.full((Tk, Tk), float("-inf"), device=device, dtype=dtype)
    full = torch.triu(full, diagonal=1)
    m = full[-Tq:, :]
    return m.view(1, 1, Tq, Tk)


def _merge_attn_masks(
    *,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    B: int,
    H: int,
    Tq: int,
    Tk: int,
    device: torch.device,
    is_causal: bool,
    mask_dtype=torch.float32,
) -> torch.Tensor | None:
    neg_inf = -1e4
    mask = None

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            m = torch.zeros_like(attn_mask, dtype=mask_dtype, device=device)
            m = m.masked_fill(~attn_mask.to(device=device), neg_inf)
        else:
            m = attn_mask.to(device=device, dtype=mask_dtype)
        mask = m

    if key_padding_mask is not None:
        keep = key_padding_mask
        if keep.dtype != torch.bool:
            keep = keep.to(torch.bool)

        m_kp = torch.zeros((B, 1, 1, Tk), device=device, dtype=mask_dtype)
        m_kp = m_kp.masked_fill(~keep.view(B, 1, 1, Tk), neg_inf)
        m_kp = m_kp.expand(B, 1, Tq, Tk)

        mask = m_kp if mask is None else (mask + m_kp)

    if is_causal:
        i = torch.arange(Tq, device=device).view(Tq, 1)
        j = torch.arange(Tk, device=device).view(1, Tk)
        cutoff = (Tk - Tq) + i
        causal_allow = (j <= cutoff)

        m_causal = torch.zeros((1, 1, Tq, Tk), device=device, dtype=mask_dtype)
        m_causal = m_causal.masked_fill(~causal_allow.view(1, 1, Tq, Tk), neg_inf)
        m_causal = m_causal.expand(B, 1, Tq, Tk)

        mask = m_causal if mask is None else (mask + m_causal)

    return mask


def _check_tensor(name: str, t: torch.Tensor):
    if t is None:
        return
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"\n❌ [NaN/Inf] {name}")
        print(f"  shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
        with torch.no_grad():
            tmin = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).min().item()
            tmax = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).max().item()
        print(f"  finite min={tmin:.6g} max={tmax:.6g}")
        raise RuntimeError(f"NaN/Inf detected in {name}")


class MultiHeadSelfAttentionSDPA(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape
        _check_tensor("input_x", x)

        x = self.norm(x)
        _check_tensor("x_after_norm", x)

        qkv = self.qkv(x)
        _check_tensor("qkv", qkv)

        q, k, v = qkv.chunk(3, dim=-1)
        _check_tensor("q_raw", q)
        _check_tensor("k_raw", k)
        _check_tensor("v_raw", v)

        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)
        _check_tensor("q_split", q)
        _check_tensor("k_split", k)
        _check_tensor("v_split", v)

        if past_kv is not None:
            k_past, v_past = past_kv
            _check_tensor("k_past", k_past)
            _check_tensor("v_past", v_past)

            Tp = k_past.size(2)
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
            _check_tensor("k_after_cache", k)
            _check_tensor("v_after_cache", v)

            if key_padding_mask is not None:
                if key_padding_mask.dtype != torch.bool:
                    key_padding_mask = key_padding_mask.to(torch.bool)
                pad_past = torch.ones((B, Tp), device=key_padding_mask.device, dtype=torch.bool)
                key_padding_mask = torch.cat([pad_past, key_padding_mask], dim=1)

        Tk = k.size(2)
        Tq = q.size(2)

        sdpa_mask = _merge_attn_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B, H=self.n_heads, Tq=Tq, Tk=Tk,
            device=x.device,
            is_causal=is_causal,
            mask_dtype=torch.float32,
        )

        if sdpa_mask is not None:
            _check_tensor("sdpa_mask", sdpa_mask)
            bad = torch.isneginf(sdpa_mask).all(dim=-1)
            if bad.any():
                print("❌ [Mask Error] some query rows fully masked!")
                raise RuntimeError("All keys masked for some queries")

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        _check_tensor("attn_out", y)

        y = _merge_heads(y)
        _check_tensor("attn_out_merged", y)

        y = self.out_proj(y)
        _check_tensor("out_proj", y)

        if self.proj_drop > 0:
            y = F.dropout(y, p=self.proj_drop, training=self.training)
            _check_tensor("out_after_dropout", y)

        present = (k, v) if use_cache else None
        return y, present


class LazyCrossAttentionGQASDPA(nn.Module):
    def __init__(self, d_model, n_heads_q, gkv, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert n_heads_q % gkv == 0
        self.d_model = d_model
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_head = d_model // n_heads_q

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)
        self.norm_q = RMSNorm(d_model)

    def forward(
        self,
        x_q: torch.Tensor,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Tq, D = x_q.shape
        B2, Tk, Gkv, d = k_ctx.shape
        assert B2 == B and Gkv == self.gkv and d == self.d_head

        q = self.q_proj(self.norm_q(x_q))
        q = _split_heads(q, self.n_heads_q)

        repeat = self.n_heads_q // self.gkv
        k = k_ctx.unsqueeze(3).expand(B, Tk, Gkv, repeat, d).reshape(B, Tk, self.n_heads_q, d)
        v = v_ctx.unsqueeze(3).expand(B, Tk, Gkv, repeat, d).reshape(B, Tk, self.n_heads_q, d)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        sdpa_mask = _merge_attn_masks(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            B=B, H=self.n_heads_q, Tq=Tq, Tk=Tk, device=x_q.device,
            is_causal=False,
            mask_dtype=torch.float32,
        )

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )

        y = _merge_heads(y)
        y = self.out_proj(y)
        if self.proj_drop > 0:
            y = F.dropout(y, p=self.proj_drop, training=self.training)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop=0.0, activation="silu"):
        super().__init__()
        act = nn.SiLU() if activation == "silu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            act,
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.drop = drop
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        x_in = self.norm(x)
        y = self.net(x_in)
        if self.drop > 0:
            y = F.dropout(y, p=self.drop, training=self.training)
        return y


class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, drop=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.drop = drop
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        x_norm = self.norm(x)

        gate_logits = self.gate(x_norm)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        top1_idx = gate_scores.argmax(dim=-1)
        top1_score = gate_scores.max(dim=-1).values

        out = torch.zeros_like(x_norm)

        for e in range(self.num_experts):
            mask = (top1_idx == e)
            if mask.sum() == 0:
                continue
            xe = x_norm[mask]
            ye = self.experts[e](xe)
            out[mask] = ye.to(out.dtype)

        out = out * top1_score.unsqueeze(-1)
        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)
        return out


class ContextProcessor(nn.Module):
    def __init__(self, d_in, d_head, gkv, lkv=1, skv=1, use_norm_k=True, use_norm_v=True):
        super().__init__()
        assert skv in (1, 2)
        self.d_in = d_in
        self.d_head = d_head
        self.gkv = gkv
        self.lkv = lkv
        self.skv = skv
        self.d_context = skv * lkv * gkv * d_head

        self.proj = nn.Linear(d_in, self.d_context, bias=False)
        self.norm_k_layers = nn.ModuleList([RMSNorm(gkv * d_head) if use_norm_k else nn.Identity()
                                            for _ in range(lkv)])
        self.norm_v_layers = nn.ModuleList([RMSNorm(gkv * d_head) if use_norm_v else nn.Identity()
                                            for _ in range(lkv)])

    def forward(self, user_static, short_term, long_term):
        ctx_parts = []
        for x in (user_static, short_term, long_term):
            if x is not None:
                ctx_parts.append(self.proj(x))
        assert len(ctx_parts) > 0
        ctx = torch.cat(ctx_parts, dim=1) if len(ctx_parts) > 1 else ctx_parts[0]

        B, Tctx, D = ctx.shape
        assert D == self.d_context

        chunk_size = self.skv * self.gkv * self.d_head
        chunks = ctx.split(chunk_size, dim=-1)
        assert len(chunks) == self.lkv, f"期望 {self.lkv} 份，得到 {len(chunks)}"

        kv_list = []
        for l, ch in enumerate(chunks):
            if self.skv == 1:
                k = ch
                k = self.norm_k_layers[l](k)
                k = k.view(B, Tctx, self.gkv, self.d_head)
                v = k
            else:
                mid = (self.gkv * self.d_head)
                k, v = ch[..., :mid], ch[..., mid:]
                k = self.norm_k_layers[l](k)
                v = self.norm_v_layers[l](v)
                k = k.view(B, Tctx, self.gkv, self.d_head)
                v = v.view(B, Tctx, self.gkv, self.d_head)
            kv_list.append((k, v))
        return kv_list
























