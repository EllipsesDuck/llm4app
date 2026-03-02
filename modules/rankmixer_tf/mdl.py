# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from per_token_ffn import PerTokenFFN
from rankmixer import RankMixerEncoder


# ============================================================
# LayerNorm (TF1.x)  —— 你要的，放最上面
# ============================================================
def layer_norm(x, scope, eps=1e-6):
    """
    TF1 LayerNorm for [B, T, D] or [B, D]
    - scope: string, used as variable_scope name
    - creates gamma/beta with shape [D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d = x.get_shape().as_list()[-1]
        if d is None:
            raise ValueError("layer_norm requires static last dimension")
        gamma = tf.get_variable("gamma", shape=[d], initializer=tf.ones_initializer())
        beta  = tf.get_variable("beta",  shape=[d], initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        y = (x - mean) / tf.sqrt(var + eps)
        return y * gamma + beta


# ============================================================
# Multi-Head Cross Attention (TF1.x)
# ============================================================
class MultiHeadCrossAttention(tf.layers.Layer):
    def __init__(self, d_model, num_heads, attn_dp=0.0, out_dp=None, name=None):
        super(MultiHeadCrossAttention, self).__init__(name=name)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.attn_dp = float(attn_dp)
        self.out_dp = float(out_dp) if out_dp is not None else float(attn_dp)
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_head = self.d_model // self.num_heads

    def build(self, input_shape):
        init = tf.variance_scaling_initializer(scale=1.0)
        self.Wq = self.add_weight("Wq", [self.d_model, self.d_model], initializer=init)
        self.Wk = self.add_weight("Wk", [self.d_model, self.d_model], initializer=init)
        self.Wv = self.add_weight("Wv", [self.d_model, self.d_model], initializer=init)
        self.Wo = self.add_weight("Wo", [self.d_model, self.d_model], initializer=init)
        super(MultiHeadCrossAttention, self).build(input_shape)

    def call(self, q, kv, training=False, attn_mask=None):
        """
        q:  [B, Nq, D]
        kv: [B, Nk, D]
        attn_mask: optional, broadcastable to [B, H, Nq, Nk], 1 for keep, 0 for mask
        """
        B = tf.shape(q)[0]
        Nq = tf.shape(q)[1]
        Nk = tf.shape(kv)[1]
        D  = self.d_model
        H  = self.num_heads
        Dh = self.d_head

        q_proj = tf.tensordot(q,  self.Wq, axes=1)  # [B,Nq,D]
        k_proj = tf.tensordot(kv, self.Wk, axes=1)  # [B,Nk,D]
        v_proj = tf.tensordot(kv, self.Wv, axes=1)  # [B,Nk,D]

        qh = tf.transpose(tf.reshape(q_proj, [B, Nq, H, Dh]), [0, 2, 1, 3])  # [B,H,Nq,Dh]
        kh = tf.transpose(tf.reshape(k_proj, [B, Nk, H, Dh]), [0, 2, 1, 3])  # [B,H,Nk,Dh]
        vh = tf.transpose(tf.reshape(v_proj, [B, Nk, H, Dh]), [0, 2, 1, 3])  # [B,H,Nk,Dh]

        logits = tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(tf.cast(Dh, tf.float32))  # [B,H,Nq,Nk]
        if attn_mask is not None:
            logits = logits + (1.0 - tf.cast(attn_mask, tf.float32)) * (-1e9)

        w = tf.nn.softmax(logits, axis=-1)
        if self.attn_dp and training:
            w = tf.nn.dropout(w, keep_prob=1.0 - self.attn_dp)

        out = tf.matmul(w, vh)  # [B,H,Nq,Dh]
        out = tf.transpose(out, [0, 2, 1, 3])            # [B,Nq,H,Dh]
        out = tf.reshape(out, [B, Nq, D])                # [B,Nq,D]
        out = tf.tensordot(out, self.Wo, axes=1)         # [B,Nq,D]
        if self.out_dp and training:
            out = tf.nn.dropout(out, keep_prob=1.0 - self.out_dp)
        return out


# ============================================================
# Domain Token Builder (Scenario/Task)
# ============================================================
class DomainTokenBuilder(tf.layers.Layer):
    """
    Build domain tokens (scenario or task):
      token = ReLU( PerTokenFFN( learnable_id_emb + optional_prior ) )
    """
    def __init__(self, num_tokens, d_model, ffn_mult=4, dp=0.0, name=None):
        super(DomainTokenBuilder, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.dp = float(dp)
        self.ffn = PerTokenFFN(num_tokens=num_tokens, d_model=d_model, mult=ffn_mult, dropout=dp, name="pt_ffn")

    def build(self, input_shape):
        init = tf.random_normal_initializer(stddev=0.02)
        self.id_emb = self.add_weight("id_emb", [self.num_tokens, self.d_model], initializer=init)
        super(DomainTokenBuilder, self).build(input_shape)

    def call(self, B, prior=None, training=False):
        # base: [B,T,D]
        base = tf.tile(tf.expand_dims(self.id_emb, 0), [B, 1, 1])

        if prior is not None:
            Din = prior.get_shape().as_list()[-1]
            if Din is None:
                raise ValueError("prior last dim must be static")
            if Din != self.d_model:
                prior = tf.layers.dense(prior, self.d_model, activation=None, name="prior_proj", reuse=tf.AUTO_REUSE)
            x = base + prior
        else:
            x = base

        y = self.ffn(x, training=training)
        y = tf.nn.relu(y)
        if self.dp and training:
            y = tf.nn.dropout(y, keep_prob=1.0 - self.dp)
        return y


# ============================================================
# MDL Block (RankMixer backbone is outside; this block does
#  scenario/task propagation + domain fused + per-token FFN)
# ============================================================
class MDLBlock(tf.layers.Layer):
    def __init__(
        self,
        num_feature_tokens,
        num_scenario_tokens,   # Ns+1 (last is global)
        num_task_tokens,
        d_model,
        num_heads,
        ffn_mult_domain=4,
        attn_dp=0.0,
        ffn_dp=0.0,
        name=None,
    ):
        super(MDLBlock, self).__init__(name=name)
        self.Nf = int(num_feature_tokens)
        self.Ns1 = int(num_scenario_tokens)
        self.Nt = int(num_task_tokens)
        self.d_model = int(d_model)

        self.scen_attn = MultiHeadCrossAttention(d_model, num_heads, attn_dp=attn_dp, name="scen_xattn")
        self.task_attn = MultiHeadCrossAttention(d_model, num_heads, attn_dp=attn_dp, name="task_xattn")

        self.scen_ffn = PerTokenFFN(num_tokens=self.Ns1, d_model=d_model, mult=ffn_mult_domain, dropout=ffn_dp, name="scen_ptffn")
        self.task_ffn = PerTokenFFN(num_tokens=self.Nt,  d_model=d_model, mult=ffn_mult_domain, dropout=ffn_dp, name="task_ptffn")

        self._ln = lambda x, scope: layer_norm(x, scope=scope)

    def call(self, Tf, Ts, Tt, scenario_ids, training=False, attn_mask_f=None):
        """
        Tf: [B,Nf,D]   feature tokens
        Ts: [B,Ns+1,D] scenario tokens (last is global)
        Tt: [B,Nt,D]   task tokens
        scenario_ids: [B] int32 in [0, Ns-1]
        attn_mask_f: optional attention mask for feature tokens, broadcastable to [B,H,Nq,Nf]
        """
        B = tf.shape(Tf)[0]
        Ns = self.Ns1 - 1  # global index

        # Safety asserts: scenario_ids in [0, Ns)
        with tf.control_dependencies([
            tf.assert_greater_equal(tf.reduce_min(scenario_ids), 0),
            tf.assert_less(tf.reduce_max(scenario_ids), Ns)
        ]):
            scenario_ids = tf.identity(scenario_ids)

        # ---- Scenario propagation ----
        Ts_ln  = self._ln(Ts, scope=(self.name + "/ln_s_q") if self.name else "ln_s_q")
        Tf_ln1 = self._ln(Tf, scope=(self.name + "/ln_f_kv_s") if self.name else "ln_f_kv_s")
        s_attn = self.scen_attn(Ts_ln, Tf_ln1, training=training, attn_mask=attn_mask_f)
        Ts = Ts + s_attn
        Ts = Ts + self.scen_ffn(self._ln(Ts, scope=(self.name + "/ln_s_ffn") if self.name else "ln_s_ffn"),
                                training=training)

        # ---- Task propagation ----
        Tt_ln  = self._ln(Tt, scope=(self.name + "/ln_t_q") if self.name else "ln_t_q")
        Tf_ln2 = self._ln(Tf, scope=(self.name + "/ln_f_kv_t") if self.name else "ln_f_kv_t")
        t_attn = self.task_attn(Tt_ln, Tf_ln2, training=training, attn_mask=attn_mask_f)
        Tt = Tt + t_attn

        # ---- Domain-fused module ----
        idx = tf.stack([tf.range(B), tf.cast(scenario_ids, tf.int32)], axis=1)
        s_sel = tf.gather_nd(Ts, idx)   # [B,D]
        s_glb = Ts[:, Ns, :]            # [B,D]
        s_avg = (s_sel + s_glb) * 0.5   # [B,D]
        Tt = Tt + tf.expand_dims(s_avg, 1)

        # ---- Per-task FFN ----
        Tt = Tt + self.task_ffn(self._ln(Tt, scope=(self.name + "/ln_t_ffn") if self.name else "ln_t_ffn"),
                                training=training)

        return Tf, Ts, Tt


# ============================================================
# MDL Encoder: SemanticTokenizer -> (RankMixer + MDLBlock)*L
# ============================================================
class MDLEncoder(tf.layers.Layer):
    def __init__(
        self,
        feature_tokenizer,      # your SemanticTokenizer instance
        Ns, Nt,
        d_model,
        feature_num_layers,
        feature_num_heads,
        feature_ffn_mult,
        mdl_num_layers,
        mdl_num_heads,
        domain_ffn_mult=4,
        token_dp=0.0,
        ffn_dp=0.0,
        attn_dp=0.0,
        use_moe=False,
        strict_one_backbone_per_layer=True,  # True: backbone called mdl_num_layers times (paper-like)
        name=None,
    ):
        super(MDLEncoder, self).__init__(name=name)
        self.tok = feature_tokenizer
        self.Ns = int(Ns)
        self.Nt = int(Nt)
        self.d_model = int(d_model)
        self.strict_one_backbone_per_layer = bool(strict_one_backbone_per_layer)

        # Domain token builders
        self.scen_builder = DomainTokenBuilder(
            num_tokens=self.Ns + 1,
            d_model=d_model,
            ffn_mult=domain_ffn_mult,
            dp=ffn_dp,
            name="scen_builder"
        )
        self.task_builder = DomainTokenBuilder(
            num_tokens=self.Nt,
            d_model=d_model,
            ffn_mult=domain_ffn_mult,
            dp=ffn_dp,
            name="task_builder"
        )

        # Feature backbone (RankMixer)
        self.Nf = int(feature_tokenizer.total_tokens)
        self.feat_backbone = RankMixerEncoder(
            num_layers=feature_num_layers,
            num_tokens=self.Nf,
            d_model=d_model,
            num_heads=feature_num_heads,  # NOTE: your ParameterFreeTokenMixer requires num_heads == num_tokens
            ffn_mult=feature_ffn_mult,
            token_dp=token_dp,
            ffn_dp=ffn_dp,
            ln_style="pre",
            use_moe=use_moe,
            use_final_ln=True,
            name="rankmixer_backbone",
        )

        # MDL blocks stack
        self.blocks = [
            MDLBlock(
                num_feature_tokens=self.Nf,
                num_scenario_tokens=self.Ns + 1,
                num_task_tokens=self.Nt,
                d_model=d_model,
                num_heads=mdl_num_heads,
                ffn_mult_domain=domain_ffn_mult,
                attn_dp=attn_dp,
                ffn_dp=ffn_dp,
                name="mdl_block_%d" % i,
            )
            for i in range(mdl_num_layers)
        ]

        # Optional final LN for task tokens (often stabilizes)
        self._final_ln = lambda x: layer_norm(x, scope=(self.name + "/final_ln_t") if self.name else "final_ln_t")

    def call(
        self,
        user_emb_list, aud_emb_list, shop_emb_list,
        scenario_ids,                 # [B]
        scenario_prior=None,          # [B, Ns+1, D'] or [B, Ns, D'] (if Ns, you can pad global outside)
        task_prior=None,              # [B, Nt, D'] optional
        training=False,
        attn_mask_f=None,             # optional attention mask for feature tokens
    ):
        # 1) feature tokens from SemanticTokenizer
        Tf, total_T, _ = self.tok.tokenize(
            user_emb_list, aud_emb_list, shop_emb_list,
            enforce_static_signature=True
        )

        # (Optional) if you want strictly one backbone per MDL layer, don't run backbone here.
        if not self.strict_one_backbone_per_layer:
            Tf = self.feat_backbone(Tf, training=training)

        B = tf.shape(Tf)[0]

        # 2) domain tokens
        Ts = self.scen_builder(B, prior=scenario_prior, training=training)  # [B, Ns+1, D]
        Tt = self.task_builder(B, prior=task_prior, training=training)      # [B, Nt, D]

        # 3) MDL layers: backbone + block
        for blk in self.blocks:
            Tf = self.feat_backbone(Tf, training=training)
            Tf, Ts, Tt = blk(Tf, Ts, Tt, scenario_ids=scenario_ids, training=training, attn_mask_f=attn_mask_f)

        Tt = self._final_ln(Tt)
        return Tf, Ts, Tt


# ============================================================
# Optional: multi-task logits head from task tokens
# ============================================================
def task_logits_from_tokens(Tt, name="task_logits_head"):
    """
    Tt: [B, Nt, D] -> logits: [B, Nt, 1]
    Shared head (one Dense applied to each task token).
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(Tt, 1, activation=None, name="logits", reuse=tf.AUTO_REUSE)
        return logits