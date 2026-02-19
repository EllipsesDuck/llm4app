import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from types import SimpleNamespace
from transformers import PreTrainedModel, PretrainedConfig, LogitsProcessor

from utils import HierarchicalSemanticIDProcessor, RMSNorm, MultiHeadSelfAttentionSDPA, LazyCrossAttentionGQASDPA, FeedForward, MoEFeedForward, ContextProcessor

_major_minor = tuple(map(int, torch.__version__.split(".")[:2]))
assert _major_minor >= (2, 0), f"Need torch>=2.0 for SDPA, got torch=={torch.__version__}"


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




