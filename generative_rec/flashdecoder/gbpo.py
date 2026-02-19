import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from types import SimpleNamespace
from transformers import PreTrainedModel, PretrainedConfig, LogitsProcessor

from loss import compute_sft_loss

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