# -*- coding: utf-8 -*-
# token mixing + per-token FFN（可选 MoE） - PyTorch version

import torch
import torch.nn as nn

from per_token_ffn import PerTokenFFN
from sparse_moe import PerTokenSparseMoE
from token_mixing import ParameterFreeTokenMixer


class RankMixerBlock(nn.Module):
    """
    RankMixer block: TokenMixing + Per-token FFN/MoE with residuals.

    Keeps TF behavior:
      - ln_style: "pre" (default) or "post"
      - if use_moe: per_token_ffn returns (z, aux_loss)
      - else: per_token_ffn returns z
      - saves moe_loss into self.moe_loss (scalar tensor)
    """

    def __init__(
        self,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        name=None,
    ):
        super().__init__()
        self.name = name
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)

        # TF layer_norm -> PyTorch LayerNorm (over last dim D)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.token_mixer = ParameterFreeTokenMixer(
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            dropout=token_dp,
            name="token_mixer",
        )

        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                num_experts=moe_experts,
                dropout=ffn_dp,
                l1_coef=moe_l1_coef,
                sparsity_ratio=moe_sparsity_ratio,
                use_dtsi=moe_use_dtsi,
                routing_type=moe_routing_type,
                name="per_token_moe",
            )
        else:
            self.per_token_ffn = PerTokenFFN(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                dropout=ffn_dp,
                name="per_token_ffn",
            )

        # will be set on forward; keep as tensor
        self.moe_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) -> out: (B, T, D)
        """
        # keep moe_loss on correct device/dtype
        moe_loss = x.new_zeros(())

        if self.ln_style == "post":
            # Post-LN: mix/ffn -> residual -> LN
            y = self.token_mixer(x)          # dropout handled inside token_mixer via self.training
            x = self.ln1(x + y)

            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x)  # returns (z, aux_loss)
            else:
                z = self.per_token_ffn(x)

            out = self.ln2(x + z)

        else:
            # Pre-LN: LN -> mix -> residual; LN -> ffn -> residual
            y = self.ln1(x)
            y = self.token_mixer(y)
            x = x + y

            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z)
            else:
                z = self.per_token_ffn(z)

            out = x + z

        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(nn.Module):
    """
    Stack RankMixer blocks.

    Behavior:
      - out = block_i(out)
      - moe_loss = sum(block_i.moe_loss)
      - optional final LayerNorm
      - stores moe_loss in self.moe_loss
    """

    def __init__(
        self,
        num_layers,
        num_tokens,
        d_model,
        num_heads,
        ffn_mult,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi",
        use_final_ln=True,
        name=None,
    ):
        super().__init__()
        self.name = name
        self.use_final_ln = bool(use_final_ln)

        self.blocks = nn.ModuleList(
            [
                RankMixerBlock(
                    num_tokens=num_tokens,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    token_dp=token_dp,
                    ffn_dp=ffn_dp,
                    ln_style=ln_style,
                    use_moe=use_moe,
                    moe_experts=moe_experts,
                    moe_l1_coef=moe_l1_coef,
                    moe_sparsity_ratio=moe_sparsity_ratio,
                    moe_use_dtsi=moe_use_dtsi,
                    moe_routing_type=moe_routing_type,
                    name=f"block_{i}",
                )
                for i in range(int(num_layers))
            ]
        )

        self.final_ln = nn.LayerNorm(d_model)
        self.moe_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        moe_losses = []

        for blk in self.blocks:
            out = blk(out)
            moe_losses.append(blk.moe_loss)

        if moe_losses:
            # sum as scalar tensor on correct device
            self.moe_loss = torch.stack(moe_losses).sum()
        else:
            self.moe_loss = x.new_zeros(())

        return self.final_ln(out) if self.use_final_ln else out
