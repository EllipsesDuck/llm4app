from typing import List, Tuple, Optional, Dict, Any
from types import SimpleNamespace
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import PreTrainedModel, PretrainedConfig,LogitsProcessor


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
        input_ids: torch.LongTensor,  # (B, T)
        scores: torch.FloatTensor,    # (B, V)
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
    """Root Mean Square Layer Normalization.

    y = x * weight / sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


def _split_heads(x, num_heads):
    """将最后一维分解为多头：(B, T, H*d) -> (B, T, H, d)
    """
    B, T, D = x.shape
    assert D % num_heads == 0, f"Hidden size {D} not divisible by heads {num_heads}"
    d = D // num_heads
    return x.view(B, T, num_heads, d)


def _merge_heads(x):
    """(B, T, H, d) -> (B, T, H*d)
    """
    B, T, H, d = x.shape
    return x.contiguous().view(B, T, H * d)


def scaled_dot_product_attention(q,k,v,attn_mask,dropout_p=0.0,training=True):
    d = q.size(-1)
    # (B, H_q, T_q, d) @ (B, H_k, d, T_k) → (B, H_q, T_q, T_k)
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 3, 1)
    attn_scores = torch.matmul(q_, k_) / math.sqrt(d)

    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask  

    attn_probs = F.softmax(attn_scores, dim=-1)
    if dropout_p > 0 and training:
        attn_probs = F.dropout(attn_probs, p=dropout_p)

    # (B, H_q, T_q, T_k) @ (B, H_k, T_k, d) → (B, H_q, T_q, d)
    v_ = v.permute(0, 2, 1, 3)
    context = torch.matmul(attn_probs, v_)
    # -> (B, T_q, H_q, d)
    context = context.permute(0, 2, 1, 3)
    return context, attn_probs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.norm_qkv = RMSNorm(d_model)

    def forward(self, x, causal=True):
        # x: (B, T, D)
        B, T, D = x.shape
        x = self.norm_qkv(x)
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = _split_heads(q, self.n_heads)
        k = _split_heads(k, self.n_heads)
        v = _split_heads(v, self.n_heads)

        # causal mask: (1, 1, T, T)
        attn_mask = None
        if causal:
            mask = torch.full((T, T), float('-inf'), device=x.device)
            mask = torch.triu(mask, diagonal=1)
            attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

        ctx, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.attn_drop, training=self.training)
        out = _merge_heads(ctx)  # (B, T, D)
        out = self.out_proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out


class LazyCrossAttentionGQA(nn.Module):
    def __init__(self, d_model, n_heads_q, gkv, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert n_heads_q % gkv == 0 
        self.d_model = d_model
        self.n_heads_q = n_heads_q
        self.gkv = gkv
        self.d_head = d_model // n_heads_q

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.norm_q = RMSNorm(d_model)

    def forward(self,x_q,k_ctx,v_ctx,attn_mask=None):
        # x_q: (B, T_q, D); k_ctx/v_ctx: (B, T_k, Gkv, d_head)
        B, Tq, D = x_q.shape
        _, Tk, Gkv, d = k_ctx.shape
        assert Gkv == self.gkv and d == self.d_head

        q = self.q_proj(self.norm_q(x_q))  # (B, Tq, D)
        q = _split_heads(q, self.n_heads_q)  # (B, Tq, Hq, d)

        repeat = self.n_heads_q // self.gkv
        k = k_ctx.repeat_interleave(repeat, dim=2)
        v = v_ctx.repeat_interleave(repeat, dim=2)

        ctx, _ = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.attn_drop, training=self.training)
        out = _merge_heads(ctx)  # (B, Tq, D)
        out = self.out_proj(out)
        if self.proj_drop > 0:
            out = F.dropout(out, p=self.proj_drop, training=self.training)
        return out


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

        gate_logits = self.gate(x_norm)         # (B,T,E)
        gate_scores = torch.softmax(gate_logits, dim=-1)
        top1_idx = gate_scores.argmax(dim=-1)   # (B,T)
        top1_score = gate_scores.max(dim=-1).values  # (B,T)

        out = torch.zeros_like(x_norm)

        for e in range(self.num_experts):
            # mask: (B,T)
            mask = (top1_idx == e)
            if mask.sum() == 0:
                continue
            xe = x_norm[mask]      
            # expert forward
            ye = self.experts[e](xe)   # (Ne, D)
            out[mask] = ye

        out = out * top1_score.unsqueeze(-1)

        if self.drop > 0:
            out = F.dropout(out, p=self.drop, training=self.training)

        return out



class ContextProcessor(nn.Module):
    def __init__(self,
                 d_in,
                 d_head,
                 gkv,
                 lkv=1,
                 skv=1,
                 use_norm_k=True,
                 use_norm_v=True):
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

    def forward(self,user_static,short_term,long_term):
        ctx_parts = []
        for x in (user_static, short_term, long_term):
            if x is not None:
                # x: (B, T, D_in) → (B, T, d_context)
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
            # ch: (B, Tctx, skv*gkv*d_head)
            if self.skv == 1:
                k = ch  # (B, Tctx, gkv*d_head)
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

    # shift for causal LM
    logits = logits[:, :-1]      # (B, T-1, V)
    labels = labels[:, 1:]       # (B, T-1)

    loss_mask = (labels != pad_id) & (labels != bos_id)

    # token-level CE
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none"
    )

    loss = loss.view(B, -1)
    loss = (loss * loss_mask).sum() / loss_mask.sum()

    return loss


class LazyDecoderBlock(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads_q,
                 gkv,
                 d_ff,
                 use_moe: bool = False,   
                 attn_drop=0.0,
                 resid_drop=0.0):
        super().__init__()

        self.cross_attn = LazyCrossAttentionGQA(
            d_model, n_heads_q, gkv,
            attn_drop=attn_drop, proj_drop=resid_drop
        )

        self.self_attn = MultiHeadSelfAttention(
            d_model, n_heads_q,
            attn_drop=attn_drop, proj_drop=resid_drop
        )

        if use_moe:
            self.ffn = MoEFeedForward(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=8,
                drop=resid_drop
            )
        else:
            self.ffn = FeedForward(  
                d_model=d_model,
                d_ff=d_ff,
                drop=resid_drop
            )

    def forward(self, x, k_ctx, v_ctx, causal=True):
        x = x + self.cross_attn(x, k_ctx, v_ctx)
        x = x + self.self_attn(x, causal=causal)
        x = x + self.ffn(x)
        return x



class LazyDecoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model = 768,
                 n_layers = 12,
                 moe_layers = 4,
                 n_heads_q = 12,
                 gkv = 3,
                 d_ff = 2048,
                 # Context Processor
                 d_ctx_in = 256,
                 lkv = 1,
                 skv = 1,
                 pad_id = 0,
                 bos_id = 1,
                 attn_drop = 0.0,
                 resid_drop = 0.0):
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
            use_moe = (l >= n_layers - moe_layers)

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

    def _kv_index_for_layer(self, l):
        return (l * self.lkv) // self.n_layers

    def forward(self,
                target_ids,  # (B, T_gen)，[BOS, s1, s2]
                user_static,  # (B, Ns, d_ctx_in)
                short_term,  # (B, Ts, d_ctx_in)
                long_term,   # (B, Tl, d_ctx_in)
                return_hidden = False):
        kv_list = self.ctx_proc(user_static, short_term, long_term)

        x = self.tok_emb(target_ids)  # (B, T, D)

        for l, blk in enumerate(self.blocks):
            idx = self._kv_index_for_layer(l)
            k_ctx, v_ctx = kv_list[idx]
            x = blk(x, k_ctx, v_ctx, causal=True)

        h = self.out_norm(x)
        logits = self.out_proj(h)  # (B, T, vocab_size)
        out = {"logits": logits}
        if return_hidden:
            out["hidden"] = h
        return out

    @torch.no_grad()
    def step(self,
             prev_ids: torch.Tensor,     # (B, T_prev)
             user_static: Optional[torch.Tensor] = None,
             short_term: Optional[torch.Tensor] = None,
             long_term: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.forward(prev_ids, user_static, short_term, long_term, return_hidden=False)
        logits_last = out["logits"][:, -1, :]  # (B, vocab)
        return logits_last

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        user_static=None,
        short_term=None,
        long_term=None,
        max_new_tokens=20,
        eos_id=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        logits_processor=None,
    ):
        """
        greedy、temperature、top-k、top-p。
        """
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = self.step(
                generated,
                user_static=user_static,
                short_term=short_term,
                long_term=long_term,
            )  # (B, vocab)

            # logits processor pipeline
            if logits_processor is not None:
                for proc in logits_processor:
                    logits = proc(generated, logits)

            # sampling or greedy
            if temperature is None or temperature == 1.0:
                temp_logits = logits
            else:
                temp_logits = logits / temperature

            # top-k
            if top_k is not None:
                kth = torch.topk(temp_logits, top_k)[0][:, -1].unsqueeze(-1)
                temp_logits[temp_logits < kth] = -float("inf")

            # top-p / nucleus
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(temp_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = -float("inf")
                temp_logits = torch.zeros_like(temp_logits).scatter(1, sorted_idx, sorted_logits)

            # final sampling
            probs = torch.softmax(temp_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # EOS break
            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated

    @torch.no_grad()
    def generate_with_logprobs(
        self,
        input_ids,
        user_static=None,
        short_term=None,
        long_term=None,
        max_new_tokens=20,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_id=None,
        logits_processor=None,
    ):
        generated = input_ids
        logprobs_list = []
        entropy_list = []

        for _ in range(max_new_tokens):
            logits = self.step(
                generated,
                user_static=user_static,
                short_term=short_term,
                long_term=long_term,
            )  # (B, V)

            if logits_processor is not None:
                for proc in logits_processor:
                    logits = proc(generated, logits)

            # temperature
            if temperature is not None and temperature != 1.0:
                logits = logits / temperature

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # sample
            next_token = torch.multinomial(probs, num_samples=1)

            # logprob / entropy (aligned with rollout policy)
            logprob = log_probs.gather(1, next_token)
            entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

            logprobs_list.append(logprob)
            entropy_list.append(entropy)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        logprobs_list = torch.cat(logprobs_list, dim=1)
        entropy_list = torch.cat(entropy_list, dim=1)

        return generated, logprobs_list, entropy_list

    
    @torch.no_grad()
    def generate_multiple(
        self,
        input_ids,
        num_return_sequences=4,
        **kwargs
    ):
        """
        return：
            sequences: (B * num_return_sequences, T)
            logprobs
            entropy
        """
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
        )
        return SimpleNamespace(logits=out["logits"], labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids, **kwargs)

    @torch.no_grad()
    def generate_rl(self, input_ids, **kwargs):
        return self.model.generate_with_logprobs(input_ids, **kwargs)

    @torch.no_grad()
    def generate_multiple(self, input_ids, **kwargs):
        return self.model.generate_multiple(input_ids, **kwargs)



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
            max_new_tokens=max_new_tokens,
            temperature=getattr(generation_config, "temperature", 1.0),
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
        bos_id: int = 1,                 # 
        gbpo_level: str = "sequence",    # "sequence" or "token"
        gbpo_use_clip: bool = False,     # 
        gbpo_mask_bos: bool = True,      # no cacualate BOS
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
        # prefix = max_prefix_len * exp(-k * progress)
        return max_prefix_len * math.exp(-k * progress)

    def _compute_token_log_probs(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)          # (B, T, V)
        log_probs_token = torch.gather(
            log_probs, dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)                                      # (B, T)
        return log_probs_token

    def _sequence_mask(self, target_ids: torch.Tensor) -> torch.Tensor:
        mask = (target_ids != self.pad_id)
        if self.gbpo_mask_bos:
            mask = mask & (target_ids != self.bos_id)
        return mask.float()  # (B, T)

    def _normalize_advantage(
        self,
        rewards: torch.Tensor,          # (B,)
        group_ids: Optional[torch.Tensor] = None,  # (B,) 或 None
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
        logp_new: torch.Tensor,   # (B, T)
        logp_old: torch.Tensor,   # (B, T)
        rewards: torch.Tensor,    # (B,) or (B,T)
        mask: torch.Tensor,       # (B, T)
        group_ids: Optional[torch.Tensor] = None,
    ):
        if rewards.dim() == 2:
            seq_reward = (rewards * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        else:
            seq_reward = rewards

        A = self._normalize_advantage(seq_reward, group_ids)  # (B,)

        if self.gbpo_level == "sequence":
            logp_new_seq = (logp_new * mask).sum(dim=1)  # (B,)
            logp_old_seq = (logp_old * mask).sum(dim=1)  # (B,)
            ratio = torch.exp(logp_new_seq - logp_old_seq)

            if self.gbpo_use_clip:
                ratio_clip = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = -(ratio * A)
                loss_clipped   = -(ratio_clip * A)
                loss = torch.max(loss_unclipped, loss_clipped).mean()
            else:
                loss = -(ratio * A).mean()
            return loss

        else:  # token-level
            # (B,T)
            log_ratio = logp_new - logp_old
            ratio_tok = torch.exp(log_ratio)
            adv_tok = A.unsqueeze(1).expand_as(mask)

            if self.gbpo_use_clip:
                ratio_clip = torch.clamp(ratio_tok, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                loss_unclipped = -(ratio_tok * adv_tok)
                loss_clipped   = -(ratio_clip * adv_tok)
                loss_tok = torch.max(loss_unclipped, loss_clipped)
            else:
                loss_tok = -(ratio_tok * adv_tok)

            loss_tok = loss_tok * mask
            return loss_tok.sum() / mask.sum().clamp_min(1.0)


    # def compute_gbpo_loss(
    #     self,
    #     new_logits: torch.Tensor,      # (B, T, V)
    #     old_logits: torch.Tensor,      # (B, T, V)
    #     target_ids: torch.Tensor,      # (B, T)
    #     rewards: torch.Tensor,         # (B,)， (B, T)
    #     group_ids: Optional[torch.Tensor] = None,
    # ) -> torch.Tensor:
    #     mask = self._sequence_mask(target_ids)  # (B, T)

    #     logp_new_tok = self._compute_token_log_probs(new_logits, target_ids)  # (B, T)
    #     logp_old_tok = self._compute_token_log_probs(old_logits, target_ids)  # (B, T)

    #     if rewards.dim() == 1:
    #         seq_reward = rewards  # (B,)
    #     else:
    #         seq_reward = (rewards * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-5)

    #     A_seq = self._normalize_advantage(seq_reward, group_ids)  # (B,)

    #     if self.gbpo_level == "sequence":
    #         logp_new_seq = (logp_new_tok * mask).sum(dim=1)   # (B,)
    #         logp_old_seq = (logp_old_tok * mask).sum(dim=1)   # (B,)
    #         ratio = torch.exp(logp_new_seq - logp_old_seq)    # (B,)

    #         if self.gbpo_use_clip:
    #             ratio_clip = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
    #             loss_unclipped = -ratio * A_seq
    #             loss_clipped   = -ratio_clip * A_seq
    #             loss = torch.max(loss_unclipped, loss_clipped).mean()
    #         else:
    #             loss = -(ratio * A_seq).mean()

    #         return loss

    #     else:
    #         log_ratio_tok = logp_new_tok - logp_old_tok        # (B, T)
    #         ratio_tok = torch.exp(log_ratio_tok)               # (B, T)

    #         adv_tok = A_seq.unsqueeze(1).expand_as(mask)       # (B, T)

    #         if self.gbpo_use_clip:
    #             ratio_clip = torch.clamp(ratio_tok, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
    #             loss_unclipped = -ratio_tok * adv_tok
    #             loss_clipped   = -ratio_clip * adv_tok
    #             loss_tok = torch.max(loss_unclipped, loss_clipped)
    #         else:
    #             loss_tok = -(ratio_tok * adv_tok)

    #         loss_tok = loss_tok * mask
    #         loss = loss_tok.sum() / mask.sum().clamp_min(1.0)
    #         return loss

    def compute_auto_prefix_len(self, step, total_steps, max_prefix_len):
        progress = step / max(total_steps, 1)

        if self.auto_prefix_mode == "linear":
            val = self._decay_linear(progress, max_prefix_len)
        elif self.auto_prefix_mode == "cosine":
            val = self._decay_cosine(progress, max_prefix_len)
        elif self.auto_prefix_mode == "exp":
            val = self._decay_exponential(progress, max_prefix_len)
        else:
            val = max_prefix_len  # fallback

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

        target_ids = batch["target_ids"].to(self.device)          # (B, T)
        user_static = batch.get("user_static", None)
        short_term  = batch.get("short_term", None)
        long_term   = batch.get("long_term", None)

        if user_static is not None:
            user_static = user_static.to(self.device)
        if short_term is not None:
            short_term = short_term.to(self.device)
        if long_term is not None:
            long_term = long_term.to(self.device)

        out_new = self.model(target_ids, user_static, short_term, long_term)
        logits_new = out_new["logits"]        # (B, T, V)

        loss_ce = self.compute_supervised_loss(logits_new, target_ids)

        if not use_rl:
            loss = loss_ce
            loss_gbpo = torch.tensor(0.0, device=self.device)

        else:
            B = target_ids.size(0)

            # start_ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=self.device)
            prefix_len = self.fixed_prefix_len

            if self.use_auto_prefix:
                step = batch.get("step", 0)
                total_steps = batch.get("total_steps", 1000)

                # max_prefix_len = target_ids.size(1) - 1  
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
            action_ids = gen_ids[:, 1:1+T]               # (B, T)
            mask = self._sequence_mask(action_ids)       # (B, T)

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




