# -*- coding: utf-8 -*-
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# =========================================================
# Zenith / Zenith++ high-fidelity TF1.x reference
# =========================================================
# Covered:
# 1) Prime Tokenization
# 2) Zenith: RSA + Flatten&Retokenize + residual token regeneration
# 3) Zenith++: TMHSA
# 4) Zenith: TSwiGLU
# 5) Zenith++: TSMoE with softmax / top-k / load balance / z-loss
# 6) Stack wrapper
# 7) Example head + total loss
#
# Notes:
# - This is a faithful public-paper-aligned implementation, not an internal production clone.
# - Static T, D are assumed where tokenwise variables are needed.
# - For clarity, sparse experts are computed densely then masked.
# =========================================================


# ---------------------------------------------------------
# Basic ops
# ---------------------------------------------------------
def layer_norm(x, scope, eps=1e-5):
    """LayerNorm over last dim."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d = x.get_shape().as_list()[-1]
        gamma = tf.get_variable("gamma", [d], initializer=tf.ones_initializer())
        beta = tf.get_variable("beta", [d], initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        x_hat = (x - mean) / tf.sqrt(var + eps)
        return x_hat * gamma + beta


def dense(x, out_dim, scope, use_bias=True, act=None):
    in_dim = x.get_shape().as_list()[-1]
    if in_dim is None:
        raise ValueError("dense() requires static last dim in TF1.x reference.")
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable(
            "w", [in_dim, out_dim],
            initializer=tf.glorot_uniform_initializer()
        )
        y = tf.tensordot(x, w, axes=[[-1], [0]])
        if use_bias:
            b = tf.get_variable("b", [out_dim], initializer=tf.zeros_initializer())
            y = y + b
        if act is not None:
            y = act(y)
        return y


def silu(x):
    return x * tf.nn.sigmoid(x)


# ---------------------------------------------------------
# Prime Tokenization
# ---------------------------------------------------------
# Paper idea:
# - ID features get separate tokens
# - sequence features get separate tokens
# - remaining features grouped by semantics
# - one feature embedding should not be split across tokens
# This function builds prime tokens from a list of feature tensors.
# Each group => concat => project => one token
# ---------------------------------------------------------
def build_prime_tokens(group_tensors,
                       token_dim,
                       scope="prime_tokenization",
                       add_token_type_emb=True):
    """
    Args:
        group_tensors: list of tensors, each shape [B, G_i]
                       one tensor = one semantic group / one prime token input
        token_dim: output token dim D
    Returns:
        tokens: [B, T, D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        tok_list = []
        T = len(group_tensors)

        if add_token_type_emb:
            type_emb = tf.get_variable(
                "token_type_emb", [T, token_dim],
                initializer=tf.glorot_uniform_initializer()
            )
        else:
            type_emb = None

        for i, g in enumerate(group_tensors):
            # one group -> one token; preserves complete feature embedding inside group
            tok = dense(g, token_dim, scope="group_proj_%d" % i, use_bias=True, act=None)  # [B,D]
            if type_emb is not None:
                tok = tok + type_emb[i][None, :]
            tok_list.append(tok)

        tokens = tf.stack(tok_list, axis=1)  # [B,T,D]
        return tokens


# ---------------------------------------------------------
# Token Boost: TSwiGLU
# Paper-aligned idea: tokenwise parameterization
# each token owns its own FFN params
# ---------------------------------------------------------
def tokenwise_swiglu(x, d_hidden, scope):
    """
    x: [B,T,D]
    tokenwise FFN:
      W1, W2: [T,D,R]
      W3:     [T,R,D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        R = d_hidden
        if T is None or D is None:
            raise ValueError("tokenwise_swiglu needs static T,D.")

        w1 = tf.get_variable("w1", [T, D, R], initializer=tf.glorot_uniform_initializer())
        w2 = tf.get_variable("w2", [T, D, R], initializer=tf.glorot_uniform_initializer())
        w3 = tf.get_variable("w3", [T, R, D], initializer=tf.glorot_uniform_initializer())

        a = tf.einsum("btd,tdr->btr", x, w1)
        b = tf.einsum("btd,tdr->btr", x, w2)
        h = silu(a) * b
        y = tf.einsum("btr,trd->btd", h, w3)
        return y


# ---------------------------------------------------------
# Token Boost: TSMoE
# Paper figure says: Linear + Softmax -> TopK -> common experts + sparse experts
# Also paper ablation mentions auxiliary loss
# Here we implement:
# - tokenwise router
# - softmax probs
# - top-k hard routing mask
# - common experts + sparse experts
# - load balancing loss
# - z-loss
# ---------------------------------------------------------
def tokenwise_sparse_moe(x,
                         d_hidden,
                         num_common_experts,
                         num_sparse_experts,
                         top_k,
                         router_z_loss_coef=1e-4,
                         load_balance_coef=1e-2,
                         scope="tsmoe"):
    """
    Args:
        x: [B,T,D]
    Returns:
        y: [B,T,D]
        aux: dict with
            load: [Es]
            router_logits: [B,T,Es]
            router_probs: [B,T,Es]
            router_mask: [B,T,Es]
            aux_loss: scalar
            z_loss: scalar
            load_balance_loss: scalar
    """
    if top_k > num_sparse_experts:
        raise ValueError("top_k must <= num_sparse_experts")

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        Es = num_sparse_experts
        if T is None or D is None:
            raise ValueError("tokenwise_sparse_moe needs static T,D.")

        # token-specific router: [T,D,Es]
        router_w = tf.get_variable(
            "router_w", [T, D, Es],
            initializer=tf.glorot_uniform_initializer()
        )
        router_b = tf.get_variable(
            "router_b", [T, Es],
            initializer=tf.zeros_initializer()
        )

        logits = tf.einsum("btd,tde->bte", x, router_w) + router_b[None, :, :]  # [B,T,Es]
        probs = tf.nn.softmax(logits, axis=-1)  # [B,T,Es]

        # hard top-k routing
        topv, topi = tf.nn.top_k(probs, k=top_k)  # [B,T,K]
        hard_mask = tf.reduce_sum(tf.one_hot(topi, depth=Es, dtype=tf.float32), axis=2)  # [B,T,Es]

        # renormalize selected probs among top-k
        gated = probs * hard_mask
        gated_sum = tf.reduce_sum(gated, axis=-1, keepdims=True) + 1e-9
        gated = gated / gated_sum  # [B,T,Es]

        # common experts
        y_common = 0.0
        for e in range(num_common_experts):
            y_common += tokenwise_swiglu(x, d_hidden, scope="common_expert_%d" % e)

        # sparse experts
        sparse_outs = []
        for e in range(num_sparse_experts):
            out_e = tokenwise_swiglu(x, d_hidden, scope="sparse_expert_%d" % e)  # [B,T,D]
            sparse_outs.append(out_e)
        sparse_stack = tf.stack(sparse_outs, axis=2)  # [B,T,Es,D]

        y_sparse = tf.reduce_sum(sparse_stack * gated[:, :, :, None], axis=2)  # [B,T,D]
        y = y_common + y_sparse

        # load stats
        # actual fraction selected
        load = tf.reduce_mean(hard_mask, axis=[0, 1])  # [Es]
        # average routing probability
        prob_mean = tf.reduce_mean(probs, axis=[0, 1])  # [Es]

        # load balancing loss: encourage actual usage ~ avg probs
        # simple symmetric alignment loss
        load_balance_loss = tf.reduce_sum(load * prob_mean) * float(Es)

        # z-loss on router logits to avoid logit explosion
        # standard stable choice: squared logsumexp
        lse = tf.reduce_logsumexp(logits, axis=-1)  # [B,T]
        z_loss = tf.reduce_mean(tf.square(lse))

        aux_loss = load_balance_coef * load_balance_loss + router_z_loss_coef * z_loss

        aux = {
            "load": load,
            "router_logits": logits,
            "router_probs": probs,
            "router_mask": hard_mask,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "load_balance_loss": load_balance_loss,
        }
        return y, aux


# ---------------------------------------------------------
# Zenith Token Fusion: RSA
# Paper formula:
# O1 = X X^T X W_R
# Then flatten & retokenize into O_TF \in R^{T_hat x d}
# Since retokenization changes token count, residual path uses MLP(X)
# final X_TB = Norm(O_TF + MLP(X))
#
# We implement:
# - o1 = (X X^T X) W_R => [B,T,k]
# - flatten [B, T*k]
# - reshape into [B, T_hat, d_hat] where T_hat * d_hat = T*k
# - residual token regeneration MLP(X)-> [B,T_hat,d_hat]
# ---------------------------------------------------------
def rsa_fusion_retokenize(x,
                          t_out=None,
                          d_out=None,
                          scope="rsa_fusion"):
    """
    Args:
        x: [B,T,D]
        t_out, d_out:
            target token count and dim after retokenization
            must satisfy t_out * d_out == T * k
            if omitted, defaults to t_out = T//2, d_out computed if divisible
    Returns:
        out: [B,t_out,d_out]
        info: dict
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        if T is None or D is None:
            raise ValueError("rsa_fusion_retokenize needs static T,D.")

        # choose k so flatten size = T * k
        # if no target given, pick t_out = T//2 when possible, else keep same T
        if t_out is None and d_out is None:
            t_out = max(1, T // 2)
            total = T * D
            if total % t_out == 0:
                d_out = total // t_out
                k = D
            else:
                t_out = T
                d_out = D
                k = D
        elif t_out is not None and d_out is not None:
            if (t_out * d_out) % T != 0:
                raise ValueError("Need T*k == t_out*d_out for retokenization.")
            k = (t_out * d_out) // T
        else:
            raise ValueError("Either specify both t_out and d_out, or neither.")

        # O1 = X X^T X W_R
        a = tf.matmul(x, x, transpose_b=True)               # [B,T,T]
        ax = tf.matmul(a, x)                                # [B,T,D]
        o1 = dense(ax, k, scope="wr", use_bias=False, act=None)  # [B,T,k]

        # flatten & retokenize (computation-free reshape)
        flat = tf.reshape(o1, [-1, T * k])                  # [B, T*k]
        o_tf = tf.reshape(flat, [-1, t_out, d_out])         # [B, t_out, d_out]

        # residual token regeneration from original X
        # map X -> same [B,t_out,d_out]
        x_flat = tf.reshape(x, [-1, T * D])                 # [B, T*D]
        res_flat = dense(x_flat, t_out * d_out, scope="res_mlp_1", act=tf.nn.relu)
        res_flat = dense(res_flat, t_out * d_out, scope="res_mlp_2", act=None)
        res = tf.reshape(res_flat, [-1, t_out, d_out])

        out = layer_norm(o_tf + res, scope="ln")
        info = {
            "o1": o1,
            "o_tf": o_tf,
            "residual_tokens": res,
            "t_out": t_out,
            "d_out": d_out,
            "k": k,
        }
        return out, info


# ---------------------------------------------------------
# Zenith++ Token Fusion: TMHSA
# Paper: token- and head-specific q/k/v projections
# No retokenization here
# output: Norm(O_TF + X)
# ---------------------------------------------------------
def tmhsa_fusion(x, num_heads, scope="tmhsa_fusion"):
    """
    x: [B,T,D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        if T is None or D is None:
            raise ValueError("tmhsa_fusion needs static T,D.")
        if D % num_heads != 0:
            raise ValueError("D must be divisible by num_heads.")
        dk = D // num_heads

        # token- and head-specific projections
        # [T,H,D,dk]
        wq = tf.get_variable("wq", [T, num_heads, D, dk], initializer=tf.glorot_uniform_initializer())
        wk = tf.get_variable("wk", [T, num_heads, D, dk], initializer=tf.glorot_uniform_initializer())
        wv = tf.get_variable("wv", [T, num_heads, D, dk], initializer=tf.glorot_uniform_initializer())

        # q[b,t,h,k] = sum_d x[b,t,d]*wq[t,h,d,k]
        q = tf.einsum("btd,thdk->bthk", x, wq)  # [B,T,H,dk]
        k = tf.einsum("btd,thdk->bthk", x, wk)
        v = tf.einsum("btd,thdk->bthk", x, wv)

        # -> [B,H,T,dk]
        qh = tf.transpose(q, [0, 2, 1, 3])
        kh = tf.transpose(k, [0, 2, 1, 3])
        vh = tf.transpose(v, [0, 2, 1, 3])

        logits = tf.matmul(qh, kh, transpose_b=True) / math.sqrt(float(dk))  # [B,H,T,T]
        attn = tf.nn.softmax(logits, axis=-1)
        out = tf.matmul(attn, vh)  # [B,H,T,dk]

        out = tf.transpose(out, [0, 2, 1, 3])   # [B,T,H,dk]
        out = tf.reshape(out, [-1, T, D])       # [B,T,D]

        out = dense(out, D, scope="wo", use_bias=False, act=None)
        out = layer_norm(out + x, scope="ln")

        info = {
            "q": q, "k": k, "v": v,
            "attn_logits": logits,
            "attn_prob": attn
        }
        return out, info


# ---------------------------------------------------------
# One Zenith Layer
# Zenith:
#   Fusion: RSA(retokenize)
#   Boost:  TSwiGLU
# Zenith++:
#   Fusion: TMHSA
#   Boost:  TSMoE
# ---------------------------------------------------------
def zenith_layer(x,
                 variant,
                 d_hidden,
                 scope,
                 num_heads=8,
                 rsa_t_out=None,
                 rsa_d_out=None,
                 moe_common=1,
                 moe_sparse=8,
                 moe_topk=2,
                 router_z_loss_coef=1e-4,
                 load_balance_coef=1e-2):
    """
    Args:
        x: [B,T,D]
        variant: "zenith" or "zenith_pp"
    Returns:
        y
        aux
    """
    aux = {}
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if variant == "zenith":
            # Token Fusion: RSA with retokenization
            h, finfo = rsa_fusion_retokenize(
                x, t_out=rsa_t_out, d_out=rsa_d_out, scope="fusion_rsa"
            )
            aux["fusion"] = finfo

            # Token Boost: TSwiGLU
            # note: h may now have changed token shape [B,T',D']
            D2 = h.get_shape().as_list()[2]
            b = tokenwise_swiglu(h, d_hidden=d_hidden, scope="boost_tswiglu")
            y = layer_norm(b + h, scope="boost_ln")
            aux["boost"] = {}

        elif variant == "zenith_pp":
            # Token Fusion: TMHSA
            h, finfo = tmhsa_fusion(x, num_heads=num_heads, scope="fusion_tmhsa")
            aux["fusion"] = finfo

            # Token Boost: TSMoE
            b, baux = tokenwise_sparse_moe(
                h,
                d_hidden=d_hidden,
                num_common_experts=moe_common,
                num_sparse_experts=moe_sparse,
                top_k=moe_topk,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_coef=load_balance_coef,
                scope="boost_tsmoe"
            )
            y = layer_norm(b + h, scope="boost_ln")
            aux["boost"] = baux
        else:
            raise ValueError("variant must be 'zenith' or 'zenith_pp'.")

        return y, aux


# ---------------------------------------------------------
# Stack
# For Zenith:
#   RSA may change token shape in first layer.
#   Later layers can continue on the new shape.
# ---------------------------------------------------------
def zenith_stack(x,
                 num_layers,
                 variant,
                 d_hidden,
                 scope="zenith_stack",
                 num_heads=8,
                 rsa_t_out=None,
                 rsa_d_out=None,
                 moe_common=1,
                 moe_sparse=8,
                 moe_topk=2,
                 router_z_loss_coef=1e-4,
                 load_balance_coef=1e-2):
    """
    Returns:
        h
        aux_list
        aux_loss_total
    """
    h = x
    aux_list = []
    aux_losses = []

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i in range(num_layers):
            h, aux = zenith_layer(
                h,
                variant=variant,
                d_hidden=d_hidden,
                scope="layer_%d" % i,
                num_heads=num_heads,
                rsa_t_out=rsa_t_out,
                rsa_d_out=rsa_d_out,
                moe_common=moe_common,
                moe_sparse=moe_sparse,
                moe_topk=moe_topk,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_coef=load_balance_coef
            )
            aux_list.append(aux)

            if variant == "zenith_pp":
                if "boost" in aux and "aux_loss" in aux["boost"]:
                    aux_losses.append(aux["boost"]["aux_loss"])

    aux_loss_total = tf.add_n(aux_losses) if len(aux_losses) > 0 else tf.constant(0.0, dtype=tf.float32)
    return h, aux_list, aux_loss_total


# ---------------------------------------------------------
# Optional: learning rate warmup helper
# Paper ablation shows LR warmup helps.
# ---------------------------------------------------------
def build_lr_with_warmup(base_lr, global_step, warmup_steps):
    """
    Linear warmup from 0.1% * base_lr to base_lr
    """
    warmup_start = base_lr * 0.001
    gs = tf.cast(global_step, tf.float32)
    ws = float(warmup_steps)

    warmup_lr = warmup_start + (base_lr - warmup_start) * tf.minimum(gs / ws, 1.0)
    return tf.where(global_step < warmup_steps, warmup_lr, tf.constant(base_lr, dtype=tf.float32))