import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from types import SimpleNamespace
from transformers import PreTrainedModel, PretrainedConfig, LogitsProcessor

# from utils import HierarchicalSemanticIDProcessor
# from gbpo import GBPOTrainer

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


_major_minor = tuple(map(int, torch.__version__.split(".")[:2]))
assert _major_minor >= (2, 0), f"Need torch>=2.0 for SDPA, got torch=={torch.__version__}"


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


def compute_sft_loss(logits, labels, pad_id, bos_id):
    B, T, V = logits.shape

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    loss_mask = (labels != pad_id)
    if bos_id is not None:
        loss_mask = loss_mask & (labels != bos_id)

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
    ).view(B, -1)

    loss = loss * loss_mask.to(loss.dtype)

    den = loss_mask.sum().clamp_min(1).to(loss.dtype)
    return loss.sum() / den


class LazyDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads_q, gkv, d_ff, use_moe: bool = False, attn_drop=0.0, resid_drop=0.0):
        super().__init__()
        self.cross_attn = LazyCrossAttentionGQASDPA(
            d_model, n_heads_q, gkv, attn_drop=attn_drop, proj_drop=resid_drop
        )
        self.self_attn = MultiHeadSelfAttentionSDPA(
            d_model, n_heads_q, attn_drop=attn_drop, proj_drop=resid_drop
        )

        if use_moe:
            self.ffn = MoEFeedForward(d_model=d_model, d_ff=d_ff, num_experts=8, drop=resid_drop)
        else:
            self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, drop=resid_drop)

    def forward(
        self,
        x: torch.Tensor,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        *,
        causal: bool = True,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x = x + self.cross_attn(
            x, k_ctx, v_ctx,
            key_padding_mask=cross_key_padding_mask,
        )

        sa_out, present = self.self_attn(
            x,
            is_causal=causal,
            key_padding_mask=target_key_padding_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + sa_out

        x = x + self.ffn(x)
        return x, present


class LazyDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_layers=12,
        moe_layers=4,
        n_heads_q=12,
        gkv=3,
        d_ff=2048,
        d_ctx_in=256,
        lkv=1,
        skv=1,
        pad_id=0,
        bos_id=1,
        attn_drop=0.0,
        resid_drop=0.0,
    ):
        super().__init__()
        assert d_model % n_heads_q == 0
        assert n_heads_q % gkv == 0

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_head = d_model // n_heads_q
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.lkv = lkv
        self.skv = skv

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.ctx_proc = ContextProcessor(
            d_in=d_ctx_in,
            d_head=self.d_head,
            gkv=gkv,
            lkv=lkv,
            skv=skv,
        )

        self.blocks = nn.ModuleList()
        for l in range(n_layers):
            use_moe = False
            self.blocks.append(
                LazyDecoderBlock(
                    d_model=d_model,
                    n_heads_q=n_heads_q,
                    gkv=gkv,
                    d_ff=d_ff,
                    use_moe=use_moe,
                    attn_drop=attn_drop,
                    resid_drop=resid_drop,
                )
            )

    def _kv_index_for_layer(self, l: int) -> int:
        return (l * self.lkv) // self.n_layers

    def forward(
        self,
        target_ids: torch.Tensor,
        user_static: Optional[torch.Tensor],
        short_term: Optional[torch.Tensor],
        long_term: Optional[torch.Tensor],
        *,
        return_hidden: bool = False,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        kv_list = self.ctx_proc(user_static, short_term, long_term)

        x = self.tok_emb(target_ids)

        tgt_kpm = (target_ids != self.pad_id)

        present_key_values = [] if use_cache else None
        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        assert len(past_key_values) == self.n_layers

        for l, blk in enumerate(self.blocks):
            idx = self._kv_index_for_layer(l)
            k_ctx, v_ctx = kv_list[idx]

            x, present = blk(
                x,
                k_ctx,
                v_ctx,
                causal=True,
                cross_key_padding_mask=cross_key_padding_mask,
                target_key_padding_mask=tgt_kpm,
                past_kv=past_key_values[l],
                use_cache=use_cache,
            )
            if use_cache:
                present_key_values.append(present)

        h = self.out_norm(x)
        logits = self.out_proj(h)

        out: Dict[str, Any] = {"logits": logits}
        if return_hidden:
            out["hidden"] = h
        if use_cache:
            out["past_key_values"] = present_key_values
        return out

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        user_static=None,
        short_term=None,
        long_term=None,
        cross_key_padding_mask=None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        out = self.forward(
            input_ids,
            user_static,
            short_term,
            long_term,
            use_cache=True,
            past_key_values=None,
            cross_key_padding_mask=cross_key_padding_mask,
        )
        logits_last = out["logits"][:, -1, :]
        return logits_last, out["past_key_values"]

    @torch.no_grad()
    def decode_one(
        self,
        last_token: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        user_static=None,
        short_term=None,
        long_term=None,
        cross_key_padding_mask=None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        out = self.forward(
            last_token,
            user_static,
            short_term,
            long_term,
            use_cache=True,
            past_key_values=past_key_values,
            cross_key_padding_mask=cross_key_padding_mask,
        )
        logits_last = out["logits"][:, -1, :]
        return logits_last, out["past_key_values"]

    def _apply_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth, float("-inf"))
        return logits

    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        return torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        user_static=None,
        short_term=None,
        long_term=None,
        max_new_tokens: int = 20,
        eos_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        logits_processor=None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids

        logits, past = self.prefill(
            input_ids=generated,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            cross_key_padding_mask=cross_key_padding_mask,
        )

        for _ in range(max_new_tokens):
            if logits_processor is not None:
                for proc in logits_processor:
                    logits = proc(generated, logits)

            if temperature is not None and temperature != 1.0:
                temp_logits = logits / temperature
            else:
                temp_logits = logits

            if top_k is not None:
                temp_logits = self._apply_top_k(temp_logits, int(top_k))
            if top_p is not None:
                temp_logits = self._apply_top_p(temp_logits, float(top_p))

            probs = torch.softmax(temp_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

            logits, past = self.decode_one(
                last_token=next_token,
                past_key_values=past,
                user_static=user_static,
                short_term=short_term,
                long_term=long_term,
                cross_key_padding_mask=cross_key_padding_mask,
            )

        return generated

    @torch.no_grad()
    def generate_with_logprobs(
        self,
        input_ids: torch.Tensor,
        user_static=None,
        short_term=None,
        long_term=None,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
        logits_processor=None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        generated = input_ids
        logprobs_list = []
        entropy_list = []

        logits, past = self.prefill(
            input_ids=generated,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            cross_key_padding_mask=cross_key_padding_mask,
        )

        for _ in range(max_new_tokens):
            if logits_processor is not None:
                for proc in logits_processor:
                    logits = proc(generated, logits)

            if temperature is not None and temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                logits = self._apply_top_k(logits, int(top_k))
            if top_p is not None:
                logits = self._apply_top_p(logits, float(top_p))

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            logprob = log_probs.gather(1, next_token)
            entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

            logprobs_list.append(logprob)
            entropy_list.append(entropy)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

            logits, past = self.decode_one(
                last_token=next_token,
                past_key_values=past,
                user_static=user_static,
                short_term=short_term,
                long_term=long_term,
                cross_key_padding_mask=cross_key_padding_mask,
            )

        logprobs = torch.cat(logprobs_list, dim=1) if len(logprobs_list) else torch.empty(
            (generated.size(0), 0), device=generated.device
        )
        entropies = torch.cat(entropy_list, dim=1) if len(entropy_list) else torch.empty(
            (generated.size(0), 0), device=generated.device
        )

        return generated, logprobs, entropies

    @torch.no_grad()
    def generate_multiple(self, input_ids, num_return_sequences=4, **kwargs):
        B = input_ids.size(0)
        input_expand = input_ids.unsqueeze(1).repeat(1, num_return_sequences, 1)
        input_expand = input_expand.view(B * num_return_sequences, -1)
        seqs, logps, ents = self.generate_with_logprobs(input_expand, **kwargs)
        return seqs, logps, ents


class LazyDecoderLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        out = self.model.forward(
            target_ids=input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
        )
        return SimpleNamespace(logits=out["logits"], labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        return self.model.generate(
            input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
            max_new_tokens=kwargs.get("max_new_tokens", 20),
            eos_id=kwargs.get("eos_id", None),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", None),
            top_p=kwargs.get("top_p", None),
            logits_processor=kwargs.get("logits_processor", None),
        )

    @torch.no_grad()
    def generate_rl(self, input_ids, **kwargs):
        return self.model.generate_with_logprobs(
            input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
            max_new_tokens=kwargs.get("max_new_tokens", 20),
            eos_id=kwargs.get("eos_id", None),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", None),
            top_p=kwargs.get("top_p", None),
            logits_processor=kwargs.get("logits_processor", None),
        )

    @torch.no_grad()
    def generate_multiple(self, input_ids, **kwargs):
        return self.model.generate_multiple(
            input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
            num_return_sequences=kwargs.get("num_return_sequences", 4),
            max_new_tokens=kwargs.get("max_new_tokens", 20),
            eos_id=kwargs.get("eos_id", None),
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", None),
            top_p=kwargs.get("top_p", None),
            logits_processor=kwargs.get("logits_processor", None),
        )


class LazyDecoderConfig(PretrainedConfig):
    model_type = "lazy_decoder"

    def __init__(
        self,
        vocab_size=32000,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=1,
        d_model=256,
        n_layers=4,
        n_heads_q=4,
        gkv=1,
        d_ff=512,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )

        self.vocab_size = vocab_size
        self._name_or_path = "LazyDecoder"

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_ff = d_ff


class LazyDecoderForCausalLM(PreTrainedModel):
    config_class = LazyDecoderConfig

    def __init__(self, config: LazyDecoderConfig, lazy_decoder):
        super().__init__(config)
        self.model = lazy_decoder
        self.config = config
        self.warnings_issued = {"estimate_tokens": True}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        logits_to_keep=None,
        **kwargs
    ):
        out = self.model.forward(
            target_ids=input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
        )

        logits = out["logits"]
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        generation_config=None,
        logits_processor=None,
        **kwargs
    ):
        if generation_config is None:
            generation_config = SimpleNamespace(
                max_new_tokens=kwargs.get("max_new_tokens", 20),
                temperature=kwargs.get("temperature", 1.0),
                top_k=kwargs.get("top_k", None),
                top_p=kwargs.get("top_p", None),
                num_beams=1,
                eos_token_id=self.config.eos_token_id,
            )

        max_new_tokens = generation_config.max_new_tokens
        num_beams = getattr(generation_config, "num_beams", 1)
        if num_beams > 1:
            raise NotImplementedError("LazyDecoder 不支持 beam search。")

        return self.model.generate(
            input_ids=input_ids,
            user_static=kwargs.get("user_static"),
            short_term=kwargs.get("short_term"),
            long_term=kwargs.get("long_term"),
            cross_key_padding_mask=kwargs.get("cross_key_padding_mask", None),
            max_new_tokens=max_new_tokens,
            temperature=getattr(generation_config, "temperature", 1.0),
            top_k=getattr(generation_config, "top_k", None),
            top_p=getattr(generation_config, "top_p", None),
            eos_id=generation_config.eos_token_id,
            logits_processor=logits_processor,
        )

    def save_pretrained(self, save_directory, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        torch.save(self.model.state_dict(), f"{save_directory}/lazy_decoder.bin")

    @classmethod
    def from_pretrained(cls, load_directory, lazy_decoder_class=None, **kwargs):
        config = LazyDecoderConfig.from_pretrained(load_directory)
        if lazy_decoder_class is None:
            raise ValueError("必须提供 lazy_decoder_class 用于构建底层模型")

        lazy_decoder = lazy_decoder_class(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads_q=config.n_heads_q,
            gkv=config.gkv,
            d_ff=config.d_ff,
        )

        weights = torch.load(f"{load_directory}/lazy_decoder.bin", map_location="cpu")
        lazy_decoder.load_state_dict(weights)
        return cls(config=config, lazy_decoder=lazy_decoder)


class GBPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        lambda_rl: float = 0.1,
        clip_eps: float = 0.2,
        pad_id: int = 0,
        bos_id: int = 1,
        gbpo_level: str = "sequence",
        gbpo_use_clip: bool = False,
        gbpo_mask_bos: bool = True,
        device: Optional[torch.device] = None,
        semantic_processor=None,
        fixed_prefix_len: int = 0,
        use_auto_prefix: bool = False,
        use_hybrid_prefix: bool = False,
        auto_prefix_mode: str = "linear",
    ):
        self.model = model
        self.lambda_rl = lambda_rl
        self.clip_eps = clip_eps
        self.pad_id = pad_id
        self.bos_id = bos_id

        assert gbpo_level in ("sequence", "token")
        self.gbpo_level = gbpo_level
        self.gbpo_use_clip = gbpo_use_clip
        self.gbpo_mask_bos = gbpo_mask_bos

        self.device = device or next(model.parameters()).device
        self.semantic_processor = semantic_processor

        self.fixed_prefix_len = fixed_prefix_len
        self.use_auto_prefix = use_auto_prefix
        self.use_hybrid_prefix = use_hybrid_prefix

        assert auto_prefix_mode in ("linear", "cosine", "exp")
        self.auto_prefix_mode = auto_prefix_mode

        self.old_model = copy.deepcopy(model).to(self.device)
        self.old_model.eval()
        for p in self.old_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def sync_old_policy(self):
        self.old_model.load_state_dict(self.model.state_dict())
        self.old_model.eval()

    def _decay_linear(self, progress, max_prefix_len):
        return max_prefix_len * (1 - progress)

    def _decay_cosine(self, progress, max_prefix_len):
        return max_prefix_len * (0.5 * (1 + math.cos(math.pi * progress)))

    def _decay_exponential(self, progress, max_prefix_len, k=5):
        return max_prefix_len * math.exp(-k * progress)

    def _sequence_mask(self, target_ids: torch.Tensor) -> torch.Tensor:
        mask = (target_ids != self.pad_id)
        if self.gbpo_mask_bos:
            mask = mask & (target_ids != self.bos_id)
        return mask.float()

    def _normalize_advantage(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if group_ids is None:
            mean = rewards.mean()
            std = rewards.std(unbiased=False)
            return (rewards - mean) / (std + eps)

        A = torch.zeros_like(rewards)
        unique_groups = torch.unique(group_ids)
        for g in unique_groups:
            idx = (group_ids == g)
            r_g = rewards[idx]
            if r_g.numel() == 0:
                continue
            mean_g = r_g.mean()
            std_g = r_g.std(unbiased=False)
            A[idx] = (r_g - mean_g) / (std_g + eps)
        return A

    def compute_supervised_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return compute_sft_loss(
            logits=logits,
            labels=targets,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
        )

    def compute_gbpo_loss_from_logprob(
        self,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ):
        if rewards.dim() == 2:
            seq_reward = (rewards * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        else:
            seq_reward = rewards

        A = self._normalize_advantage(seq_reward, group_ids)

        if self.gbpo_level == "sequence":
            logp_new_seq = (logp_new * mask).sum(dim=1)
            logp_old_seq = (logp_old * mask).sum(dim=1)
            ratio = torch.exp(logp_new_seq - logp_old_seq)

            if self.gbpo_use_clip:
                ratio_clip = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = -(ratio * A)
                loss_clipped = -(ratio_clip * A)
                loss = torch.max(loss_unclipped, loss_clipped).mean()
            else:
                loss = -(ratio * A).mean()
            return loss

        log_ratio = logp_new - logp_old
        ratio_tok = torch.exp(log_ratio)
        adv_tok = A.unsqueeze(1).expand_as(mask)

        if self.gbpo_use_clip:
            ratio_clip = torch.clamp(ratio_tok, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            loss_unclipped = -(ratio_tok * adv_tok)
            loss_clipped = -(ratio_clip * adv_tok)
            loss_tok = torch.max(loss_unclipped, loss_clipped)
        else:
            loss_tok = -(ratio_tok * adv_tok)

        loss_tok = loss_tok * mask
        return loss_tok.sum() / mask.sum().clamp_min(1.0)

    def compute_auto_prefix_len(self, step, total_steps, max_prefix_len):
        progress = step / max(total_steps, 1)
        if self.auto_prefix_mode == "linear":
            val = self._decay_linear(progress, max_prefix_len)
        elif self.auto_prefix_mode == "cosine":
            val = self._decay_cosine(progress, max_prefix_len)
        elif self.auto_prefix_mode == "exp":
            val = self._decay_exponential(progress, max_prefix_len)
        else:
            val = max_prefix_len
        return int(max(val, 0))

    def compute_hybrid_prefix_len(self, prefix_len):
        import random
        if random.random() < 0.5:
            return 0
        return prefix_len

    def train_step(
        self,
        batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        use_rl: bool = False,
    ) -> Dict[str, float]:
        self.model.train()

        target_ids = batch["target_ids"].to(self.device)

        user_static = batch.get("user_static", None)
        short_term = batch.get("short_term", None)
        long_term = batch.get("long_term", None)

        if user_static is not None:
            user_static = user_static.to(self.device)
        if short_term is not None:
            short_term = short_term.to(self.device)
        if long_term is not None:
            long_term = long_term.to(self.device)

        cross_kpm = batch.get("cross_key_padding_mask", None)
        if cross_kpm is not None:
            cross_kpm = cross_kpm.to(self.device)

        out_new = self.model(
            target_ids,
            user_static,
            short_term,
            long_term,
            cross_key_padding_mask=cross_kpm,
        )
        logits_new = out_new["logits"]

        loss_ce = self.compute_supervised_loss(logits_new, target_ids)

        if not use_rl:
            loss = loss_ce
            loss_gbpo = torch.tensor(0.0, device=self.device)
        else:
            B = target_ids.size(0)
            prefix_len = self.fixed_prefix_len

            if self.use_auto_prefix:
                step = batch.get("step", 0)
                total_steps = batch.get("total_steps", 1000)
                valid_lengths = (target_ids != self.pad_id).sum(dim=1)
                max_prefix_len = int(valid_lengths.max().item()) - 1
                prefix_len = self.compute_auto_prefix_len(step, total_steps, max_prefix_len)

            if self.use_hybrid_prefix:
                prefix_len = self.compute_hybrid_prefix_len(prefix_len)

            if prefix_len <= 0:
                start_ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
            else:
                start_ids = target_ids[:, :prefix_len]

            max_new = batch.get("max_new_tokens", 3)
            temperature = batch.get("temperature", 1.0)
            eos_id = batch.get("eos_id", None)

            lp = [self.semantic_processor] if self.semantic_processor is not None else None

            with torch.no_grad():
                gen_ids, logp_new, _ = self.model.generate_with_logprobs(
                    input_ids=start_ids,
                    user_static=user_static,
                    short_term=short_term,
                    long_term=long_term,
                    cross_key_padding_mask=cross_kpm,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    eos_id=eos_id,
                    logits_processor=lp,
                )

                _, logp_old, _ = self.old_model.generate_with_logprobs(
                    input_ids=start_ids,
                    user_static=user_static,
                    short_term=short_term,
                    long_term=long_term,
                    cross_key_padding_mask=cross_kpm,
                    max_new_tokens=max_new,
                    temperature=temperature,
                    eos_id=eos_id,
                    logits_processor=lp,
                )

            rewards = batch["rewards"].to(self.device)
            group_ids = batch.get("group_ids", None)
            if group_ids is not None:
                group_ids = group_ids.to(self.device)

            logp_new = logp_new.detach()
            logp_old = logp_old.detach()

            T = logp_new.size(1)
            action_ids = gen_ids[:, 1:1 + T]
            mask = self._sequence_mask(action_ids)

            loss_gbpo = self.compute_gbpo_loss_from_logprob(
                logp_new=logp_new,
                logp_old=logp_old,
                rewards=rewards,
                mask=mask,
                group_ids=group_ids,
            )

            loss = loss_ce + self.lambda_rl * loss_gbpo

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "loss": float(loss.item()),
            "loss_ce": float(loss_ce.item()),
            "loss_gbpo": float(loss_gbpo.item()),
        }
