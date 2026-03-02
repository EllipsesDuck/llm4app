# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3))))


class PerTokenFFN(tf.layers.Layer):
    """
    Per-token FFN with independent parameters for each token.
    """

    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0, name=None):
        super(PerTokenFFN, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = float(dropout)

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        # 每个 token 独立参数：各自的 FFN 权重。
        self.W1 = self.add_weight("W1", [self.num_tokens, self.d_model, hidden_dim], initializer=init)
        self.b1 = self.add_weight("b1", [self.num_tokens, hidden_dim], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight("W2", [self.num_tokens, hidden_dim, self.d_model], initializer=init)
        self.b2 = self.add_weight("b2", [self.num_tokens, self.d_model], initializer=tf.zeros_initializer())
        super(PerTokenFFN, self).build(input_shape)

    def call(self, x, training=False):
        # btd x tdh -> bth，逐 token 线性变换。
        h = tf.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        # bth x thd -> btd，恢复 token 维度
        y = tf.einsum("bth,thd->btd", h, self.W2) + self.b2
        if self.dropout and training:
            y = tf.nn.dropout(y, keep_prob=1.0 - self.dropout)
        return y
    
# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from modules.rankmixer.per_token_ffn import gelu


class PerTokenSparseMoE(tf.layers.Layer):
    """
    Per-token Sparse MoE with ReLU routing + optional DTSI.
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
        super(PerTokenSparseMoE, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight(
            "W1",
            [self.num_tokens, self.num_experts, self.d_model, hidden_dim],
            initializer=init,
        )
        self.b1 = self.add_weight(
            "b1",
            [self.num_tokens, self.num_experts, hidden_dim],
            initializer=tf.zeros_initializer(),
        )
        self.W2 = self.add_weight(
            "W2",
            [self.num_tokens, self.num_experts, hidden_dim, self.d_model],
            initializer=init,
        )
        self.b2 = self.add_weight(
            "b2",
            [self.num_tokens, self.num_experts, self.d_model],
            initializer=tf.zeros_initializer(),
        )
        self.gate_w_train = self.add_weight(
            "gate_w_train",
            [self.num_tokens, self.d_model, self.num_experts],
            initializer=init,
        )
        self.gate_b_train = self.add_weight(
            "gate_b_train",
            [self.num_tokens, self.num_experts],
            initializer=tf.zeros_initializer(),
        )
        if self.use_dtsi:
            self.gate_w_infer = self.add_weight(
                "gate_w_infer",
                [self.num_tokens, self.d_model, self.num_experts],
                initializer=init,
            )
            self.gate_b_infer = self.add_weight(
                "gate_b_infer",
                [self.num_tokens, self.num_experts],
                initializer=tf.zeros_initializer(),
            )
        super(PerTokenSparseMoE, self).build(input_shape)

    def _router_logits(self, x, w, b):
        # 每个 token 的路由 logits，用于专家选择。
        return tf.einsum("btd,tde->bte", x, w) + b

    def call(self, x, training=False):
        # x 形状: [B, T, D]
        # 计算每个 token 的专家输出。
        h = tf.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        expert_out = tf.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout and training:
            expert_out = tf.nn.dropout(expert_out, keep_prob=1.0 - self.dropout)

        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            # 训练阶段使用 soft 路由以提高专家覆盖。
            gate_train = tf.nn.softmax(gate_train_logits, axis=-1)
        elif self.routing_type == "relu":
            gate_train = tf.nn.relu(gate_train_logits)
        else:
            raise ValueError("Unsupported routing_type: %s" % self.routing_type)

        if self.use_dtsi:
            # 推理阶段使用 ReLU gate 以获得稀疏激活。
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = tf.nn.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # 训练/推理选择不同 gate。
        gate = gate_train if training else gate_infer
        y = tf.reduce_sum(expert_out * tf.expand_dims(gate, -1), axis=2)

        if self.l1_coef > 0.0:
            # L1 惩罚鼓励稀疏专家激活。
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            l1_loss = self.l1_coef * scale * tf.reduce_mean(tf.reduce_sum(gate_infer, axis=-1))
        else:
            l1_loss = tf.constant(0.0)
        return y, l1_loss

# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class ParameterFreeTokenMixer(tf.layers.Layer):
    """
    Paper-style parameter-free token mixing with strict H = T.
    """

    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0, name=None):
        super(ParameterFreeTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads, got d_model=%d num_heads=%d"
                % (self.d_model, self.num_heads)
            )
        self.d_head = self.d_model // self.num_heads
        super(ParameterFreeTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        # x 形状: [B, T, D]
        batch_size = tf.shape(x)[0]
        t_count = self.num_tokens
        h_count = self.num_heads
        d_head = self.d_head

        # 先拆分 head，再交换 token/head 轴完成混合。
        split = tf.reshape(x, [batch_size, t_count, h_count, d_head])
        shuffled = tf.transpose(split, [0, 2, 1, 3])  # 形状: [B, H, T, D/H]
        merged = tf.reshape(shuffled, [batch_size, h_count, t_count * d_head])  # 形状: [B, H, D]
        mixed = tf.reshape(merged, [batch_size, t_count, self.d_model])  # 形状: [B, T, D]

        if self.dropout and training:
            mixed = tf.nn.dropout(mixed, keep_prob=1.0 - self.dropout)
        return mixed

# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class SemanticTokenizer(object):
    """
    - token 分配：floor + 余数，严格保证 sum == total_tokens
    - scope：支持传 experiment_tag / signature 自动拼进 name，避免同进程多实验 AUTO_REUSE 误复用
    """

    def __init__(
        self,
        d_model,
        embedding_dim,
        total_tokens=64,
        min_tokens_per_group=1,
        name="semantic_tokenizer",
        experiment_tag=None,          # 实验标识，如 "T64_v1" / "ab_20260128"
        isolate_by_config=True,       # 自动把关键超参拼进 scope，避免冲突
    ):
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

        # 生成一个更安全的 scope 名称
        self.name = self._make_safe_scope_name()

        # 可以把 signature 存起来，后续多次调用强制一致
        self._built_signature = None

    def _make_safe_scope_name(self):
        parts = [self.base_name]
        if self.experiment_tag:
            parts.append(self.experiment_tag)
        if self.isolate_by_config:
            # 变量 shape 变化的关键配置拼进去
            parts.append("T%d" % self.total_tokens)
            parts.append("Dm%d" % self.d_model)
            parts.append("De%d" % self.embedding_dim)
            parts.append("Min%d" % self.min_tokens_per_group)
        return "/".join(parts)


    def _alloc_3_floor_remainder(self, n1, n2, n3):
        """
        按 size 比例分配 total_tokens，采用 floor + 余数分配，严格保证 sum == T。
        约束：每组至少 m（若 m*3 > T，则降级为尽量均分但不保证 >=m）
        """
        T = self.total_tokens
        m = self.min_tokens_per_group

        if m * 3 > T:
            # 无法满足每组至少 m，降级为均分（严格 sum==T）
            base = T // 3
            rem = T - 3 * base
            a = [base, base, base]
            for i in range(rem):
                a[i] += 1
            return a[0], a[1], a[2]

        sizes = [max(0, int(n1)), max(0, int(n2)), max(0, int(n3))]
        s = sum(sizes)

        if s == 0:
            # 均分，严格 sum==T，且 >=m（能满足）
            base = T // 3
            rem = T - 3 * base
            a = [base, base, base]
            for i in range(rem):
                a[i] += 1
            # 强制 >=m（一定能修正成功，因为 m*3<=T）
            for i in range(3):
                if a[i] < m:
                    a[i] = m
            # 修正总和
            while sum(a) > T:
                k = max(range(3), key=lambda i: a[i])
                if a[k] > m:
                    a[k] -= 1
            while sum(a) < T:
                k = min(range(3), key=lambda i: a[i])
                a[k] += 1
            return a[0], a[1], a[2]

        # 先给每组 m 个
        T_left = T - 3 * m

        # 按比例分配剩余
        raw = [sizes[i] * float(T_left) / float(s) for i in range(3)]
        base = [int(x) for x in raw]  # floor
        rem = T_left - sum(base)      # 余数

        # 余数按 fractional part 从大到小分
        frac = [(i, raw[i] - float(base[i])) for i in range(3)]
        frac.sort(key=lambda x: x[1], reverse=True)
        for k in range(rem):
            base[frac[k % 3][0]] += 1

        a = [m + base[i] for i in range(3)]
        # 严格保证
        assert sum(a) == T, "allocation bug: sum != total_tokens"
        return a[0], a[1], a[2]

    # group: chunk -> project
    def _chunk_project(self, x_3d, Tg, group_name):
        """
        x_3d: [B, Ng, D] -> [B, Tg, d_model]
        b-like chunk: token_size=ceil(Ng/Tg), pad, reshape, dense
        """
        Tg = max(1, int(Tg))

        Ng = x_3d.shape.as_list()[1]
        if Ng is None:
            raise ValueError("Ng must be static (python list length fixed).")
        if Ng == 0:
            return tf.zeros([tf.shape(x_3d)[0], Tg, self.d_model])

        token_size = int((Ng + Tg - 1) / Tg)
        pad_needed = Tg * token_size - Ng
        if pad_needed > 0:
            pad = tf.zeros([tf.shape(x_3d)[0], pad_needed, self.embedding_dim])
            x_3d = tf.concat([x_3d, pad], axis=1)

        flat = tf.reshape(x_3d, [tf.shape(x_3d)[0], Tg, token_size * self.embedding_dim])

        # dense name 含输入维度签名，防止 AUTO_REUSE 下 shape mismatch
        with tf.variable_scope(group_name, reuse=tf.AUTO_REUSE):
            proj_name = "proj_in%d_out%d" % (token_size * self.embedding_dim, self.d_model)
            y = tf.layers.dense(
                flat,
                self.d_model,
                activation=None,
                name=proj_name,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.36, dtype=tf.float32),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001),
                # kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                reuse=tf.AUTO_REUSE
            )
            y.set_shape([None, Tg, self.d_model])
            return y


    def tokenize(self, user_emb_list, aud_emb_list, shop_emb_list, enforce_static_signature=True):
        """
        enforce_static_signature:
          - True: 同一个 tokenizer 实例多次 tokenize 时要求 Nu/Na/Ns 不变（避免变量集合变化）
        """

        def _stack_list(lst, list_name):
            if not lst:
                return None
            # rank2 check
            deps = [tf.assert_rank(lst[0], 2, message="%s elements must be [B, D]" % list_name)]
            with tf.control_dependencies(deps):
                x0 = tf.identity(lst[0])
            x = tf.stack([x0] + lst[1:], axis=1)  # [B, Ng, D]
            x.set_shape([None, len(lst), self.embedding_dim])
            return x

        U = _stack_list(user_emb_list, "user_emb_list")
        A = _stack_list(aud_emb_list, "aud_emb_list")
        S = _stack_list(shop_emb_list, "shop_emb_list")

        ref = U if U is not None else (A if A is not None else S)
        if ref is None:
            raise ValueError("All groups are empty; need at least one embedding list non-empty.")
        B = tf.shape(ref)[0]

        nU = len(user_emb_list) if user_emb_list else 0
        nA = len(aud_emb_list) if aud_emb_list else 0
        nS = len(shop_emb_list) if shop_emb_list else 0

        # 可选：强制每次调用 Nu/Na/Ns 不变（更工程安全）
        if enforce_static_signature:
            sig = (nU, nA, nS)
            if self._built_signature is None:
                self._built_signature = sig
            elif self._built_signature != sig:
                raise ValueError(
                    "Tokenizer group sizes changed across calls: prev=%s now=%s. "
                    "This can create different variables under AUTO_REUSE. "
                    "Use a different tokenizer name/experiment_tag or disable enforce_static_signature."
                    % (str(self._built_signature), str(sig))
                )

        Tu, Ta, Ts = self._alloc_3_floor_remainder(nU, nA, nS)

        # 带实验签名，避免同进程不同配置互相复用/冲突
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            u_tok = self._chunk_project(U, Tu, "user") if U is not None else tf.zeros([B, Tu, self.d_model])
            a_tok = self._chunk_project(A, Ta, "audience") if A is not None else tf.zeros([B, Ta, self.d_model])
            s_tok = self._chunk_project(S, Ts, "shop") if S is not None else tf.zeros([B, Ts, self.d_model])

            tokens = tf.concat([u_tok, a_tok, s_tok], axis=1)
            total_T = Tu + Ta + Ts
            tokens.set_shape([None, total_T, self.d_model])
            return tokens, total_T, (Tu, Ta, Ts)

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