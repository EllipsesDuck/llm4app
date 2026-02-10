# -*- coding: utf-8 -*-
# token mixing + per-token FFN（可选 MoE）。

from collections import OrderedDict

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from modules.transformerv1 import layer_norm
from modules.rankmixer.per_token_ffn import PerTokenFFN
from modules.rankmixer.sparse_moe import PerTokenSparseMoE
from modules.rankmixer.token_mixing import ParameterFreeTokenMixer



class RankMixerBlock(tf.layers.Layer):
    """
    RankMixer block: TokenMixing + Per-token FFN with residuals.
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
        super(RankMixerBlock, self).__init__(name=name)

        # self.ln1 = layer_norm(name="ln1")
        # self.ln2 = layer_norm(name="ln2")
        ln1_scope = (self.name + "/ln1") if self.name else "ln1"
        ln2_scope = (self.name + "/ln2") if self.name else "ln2"
        self.ln1 = lambda x: layer_norm(x, scope=ln1_scope)
        self.ln2 = lambda x: layer_norm(x, scope=ln2_scope)

        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)

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
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        moe_loss = tf.constant(0.0)
        if self.ln_style == "post":
            # Post-LN：先 mix/ffn，再残差相加并归一化。
            y = self.token_mixer(x, training=training)
            x = self.ln1(x + y)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x, training=training)
            else:
                z = self.per_token_ffn(x, training=training)
            out = self.ln2(x + z)
        else:
            # Pre-LN：先归一化，再 mix/ffn，最后残差相加。
            y = self.ln1(x)
            y = self.token_mixer(y, training=training)
            x = x + y
            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z, training=training)
            else:
                z = self.per_token_ffn(z, training=training)
            out = x + z
        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(tf.layers.Layer):
    """Stack RankMixer blocks."""

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
        super(RankMixerEncoder, self).__init__(name=name)
        self.use_final_ln = bool(use_final_ln)
        self.blocks = [
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
                name="block_%d" % i,
            )
            for i in range(num_layers)
        ]
        # self.final_ln = layer_norm(name="encoder_ln")
        final_ln_scope = (self.name + "/final_ln") if self.name else "final_ln"
        self.final_ln = lambda x: layer_norm(x, scope=final_ln_scope)
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        out = x
        moe_losses = []
        for blk in self.blocks:
            out = blk(out, training=training)
            moe_losses.append(blk.moe_loss)
        self.moe_loss = tf.add_n(moe_losses) if moe_losses else tf.constant(0.0)
        return self.final_ln(out) if self.use_final_ln else out