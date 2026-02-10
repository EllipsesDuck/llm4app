# -*- coding: utf-8 -*-
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# TF1.x graph mode
# tf.disable_eager_execution()  # 若你的环境默认启用 eager，可打开

# --------------------------
# Basic ops
# --------------------------
def layer_norm(x, scope, eps=1e-5):
    """LayerNorm over last dim."""
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d = x.get_shape().as_list()[-1]
        gamma = tf.get_variable("gamma", [d], initializer=tf.ones_initializer())
        beta  = tf.get_variable("beta",  [d], initializer=tf.zeros_initializer())
        mean, var = tf.nn.moments(x, axes=[-1], keep_dims=True)
        xhat = (x - mean) / tf.sqrt(var + eps)
        return xhat * gamma + beta

def dense(x, out_dim, scope, use_bias=True, act=None):
    in_dim = x.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable("w", [in_dim, out_dim],
                            initializer=tf.glorot_uniform_initializer())
        y = tf.tensordot(x, w, axes=[[-1], [0]])
        if use_bias:
            b = tf.get_variable("b", [out_dim], initializer=tf.zeros_initializer())
            y = y + b
        if act is not None:
            y = act(y)
        return y

def silu(x):
    # TF1.x: swish/silu
    return x * tf.nn.sigmoid(x)

# --------------------------
# Token Boost: TSwiGLU (tokenwise)
# --------------------------
def tokenwise_swiglu(x, d_hidden, scope):
    """
    x: [B,T,D]
    per-token weights:
      W1,W2: [T,D,R]
      W3:    [T,R,D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        B = tf.shape(x)[0]
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        R = d_hidden
        assert T is not None and D is not None, "Need static T,D in TF1.x for variables."

        w1 = tf.get_variable("w1", [T, D, R], initializer=tf.glorot_uniform_initializer())
        w2 = tf.get_variable("w2", [T, D, R], initializer=tf.glorot_uniform_initializer())
        w3 = tf.get_variable("w3", [T, R, D], initializer=tf.glorot_uniform_initializer())

        # xw1[b,t,r] = sum_d x[b,t,d]*w1[t,d,r]
        xw1 = tf.einsum("btd,tdr->btr", x, w1)
        xw2 = tf.einsum("btd,tdr->btr", x, w2)
        gate = silu(xw1) * xw2
        y = tf.einsum("btr,trd->btd", gate, w3)
        return y

# --------------------------
# Token Boost: TSMoE (tokenwise router + shared + sparse experts)
# --------------------------
def tokenwise_sparse_moe(x,
                         d_hidden,
                         num_shared_experts,
                         num_sparse_experts,
                         top_k,
                         scope):
    """
    x: [B,T,D]
    Router is token-specific: W0: [T,D,Es] => logits [B,T,Es]
    Experts: each expert is a TSwiGLU (tokenwise)
    Output:
      y: [B,T,D]
      load: [Es] (average routing mask per expert for monitoring/loss)
      logits: [B,T,Es] (for z-loss if you want)
      mask: [B,T,Es] (top-k 0/1)
    """
    assert top_k <= num_sparse_experts
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        Es = num_sparse_experts
        assert T is not None and D is not None

        # token-specific router weights
        router_w = tf.get_variable("router_w", [T, D, Es], initializer=tf.glorot_uniform_initializer())
        logits = tf.einsum("btd,tde->bte", x, router_w)  # [B,T,Es]

        # top-k mask (0/1)
        topv, topi = tf.nn.top_k(logits, k=top_k)  # [B,T,K]
        mask = tf.reduce_sum(tf.one_hot(topi, depth=Es, dtype=tf.float32), axis=2)  # [B,T,Es]

        # shared experts sum
        y = 0.0
        for e in range(num_shared_experts):
            y += tokenwise_swiglu(x, d_hidden, scope="shared_expert_%d" % e)

        # sparse experts: compute all then mask-sum (simple, slower)
        sparse_outs = []
        for e in range(num_sparse_experts):
            out_e = tokenwise_swiglu(x, d_hidden, scope="sparse_expert_%d" % e)  # [B,T,D]
            sparse_outs.append(out_e)
        stack = tf.stack(sparse_outs, axis=2)            # [B,T,Es,D]
        y_sparse = tf.reduce_sum(stack * mask[:, :, :, None], axis=2)  # [B,T,D]
        y += y_sparse

        # load statistic
        load = tf.reduce_mean(mask, axis=[0, 1])  # [Es]
        return y, load, logits, mask

# --------------------------
# Token Fusion: RSA (Zenith)
# --------------------------
def rsa_fusion(x, scope, k=None):
    """
    RSA per paper (simplified):
      A = X X^T  -> [B,T,T]
      O = (A X) W_R -> [B,T,k]
    Then "retokenize" (reshape) in paper. For simplicity, we keep k==D so it's a no-op.
    Residual uses MLP(X).
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        assert T is not None and D is not None
        if k is None:
            k = D
        if k != D:
            raise ValueError("This TF1.x reference keeps k==D to avoid retokenize reshapes.")

        # A = X X^T
        a = tf.matmul(x, x, transpose_b=True)     # [B,T,T]
        ax = tf.matmul(a, x)                      # [B,T,D]
        o = dense(ax, k, scope="wr", use_bias=False, act=None)  # [B,T,D]

        # residual MLP(X)
        h = dense(x, D, scope="mlp_1", act=tf.nn.relu)
        h = dense(h, D, scope="mlp_2", act=None)

        out = layer_norm(o + h, scope="ln")
        return out

# --------------------------
# Token Fusion: TMHSA (Zenith++)
# --------------------------
def tmhsa_fusion(x, num_heads, scope):
    """
    Tokenwise Multi-Head Self-Attention:
      token-specific Wq/Wk/Wv: [T,D,D]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        T = x.get_shape().as_list()[1]
        D = x.get_shape().as_list()[2]
        assert T is not None and D is not None
        assert D % num_heads == 0
        dk = D // num_heads

        wq = tf.get_variable("wq", [T, D, D], initializer=tf.glorot_uniform_initializer())
        wk = tf.get_variable("wk", [T, D, D], initializer=tf.glorot_uniform_initializer())
        wv = tf.get_variable("wv", [T, D, D], initializer=tf.glorot_uniform_initializer())

        q = tf.einsum("btd,tde->bte", x, wq)  # [B,T,D]
        k = tf.einsum("btd,tde->bte", x, wk)
        v = tf.einsum("btd,tde->bte", x, wv)

        # [B,H,T,dk]
        def split_heads(z):
            z = tf.reshape(z, [-1, T, num_heads, dk])
            return tf.transpose(z, [0, 2, 1, 3])

        qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
        attn_logits = tf.matmul(qh, kh, transpose_b=True) / math.sqrt(float(dk))  # [B,H,T,T]
        attn = tf.nn.softmax(attn_logits, axis=-1)
        out = tf.matmul(attn, vh)  # [B,H,T,dk]

        out = tf.transpose(out, [0, 2, 1, 3])                 # [B,T,H,dk]
        out = tf.reshape(out, [-1, T, D])                     # [B,T,D]
        out = dense(out, D, scope="wo", use_bias=False, act=None)

        out = layer_norm(out + x, scope="ln")
        return out

# --------------------------
# Zenith Layer: Fusion + Boost
# --------------------------
def zenith_layer(x,
                 fusion_type,          # "rsa" or "tmhsa"
                 boost_type,           # "tswiglu" or "tsmoe"
                 d_hidden,
                 scope,
                 num_heads=8,
                 moe_shared=1,
                 moe_sparse=8,
                 moe_topk=2):
    """
    x: [B,T,D]
    returns:
      y: [B,T,D]
      aux: dict of tensors (load/logits/mask) if moe else empty
    """
    aux = {}
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Token Fusion
        if fusion_type == "rsa":
            h = rsa_fusion(x, scope="fusion_rsa")
        elif fusion_type == "tmhsa":
            h = tmhsa_fusion(x, num_heads=num_heads, scope="fusion_tmhsa")
        else:
            raise ValueError("Unknown fusion_type: %s" % fusion_type)

        # Token Boost
        if boost_type == "tswiglu":
            b = tokenwise_swiglu(h, d_hidden=d_hidden, scope="boost_tswiglu")
            y = layer_norm(b + h, scope="boost_ln")
        elif boost_type == "tsmoe":
            b, load, logits, mask = tokenwise_sparse_moe(
                h, d_hidden=d_hidden,
                num_shared_experts=moe_shared,
                num_sparse_experts=moe_sparse,
                top_k=moe_topk,
                scope="boost_tsmoe"
            )
            y = layer_norm(b + h, scope="boost_ln")
            aux.update({"load": load, "router_logits": logits, "router_mask": mask})
        else:
            raise ValueError("Unknown boost_type: %s" % boost_type)

        return y, aux

def zenith_stack(x, num_layers, layer_kwargs, scope="zenith"):
    """
    x: [B,T,D]
    layer_kwargs: dict passed to zenith_layer (fusion_type, boost_type, d_hidden, etc.)
    returns:
      y: [B,T,D]
      aux_list: list of aux dicts per layer
    """
    aux_list = []
    h = x
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i in range(num_layers):
            h, aux = zenith_layer(h, scope="layer_%d" % i, **layer_kwargs)
            aux_list.append(aux)
    return h, aux_list


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    tf.reset_default_graph()

    # Suppose you already have tokens: [B,T,D]
    B, T, D = 32, 32, 256
    tokens = tf.placeholder(tf.float32, shape=[None, T, D], name="tokens")

    # Zenith (RSA + TSwiGLU)
    y1, aux1 = zenith_stack(
        tokens, num_layers=2,
        layer_kwargs=dict(
            fusion_type="rsa",
            boost_type="tswiglu",
            d_hidden=1024
        ),
        scope="Zenith"
    )

    # Zenith++ (TMHSA + TSMoE)
    y2, aux2 = zenith_stack(
        tokens, num_layers=2,
        layer_kwargs=dict(
            fusion_type="tmhsa",
            boost_type="tsmoe",
            d_hidden=1024,
            num_heads=8,
            moe_shared=1,
            moe_sparse=8,
            moe_topk=2
        ),
        scope="ZenithPP"
    )

    # A simple pooling + head (你可替换成自己的 tower)
    pooled = tf.reduce_mean(y2, axis=1)  # [B,D]
    logit = dense(pooled, 1, scope="head", act=None)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import numpy as np
        feed = {tokens: np.random.randn(B, T, D).astype(np.float32)}
        out = sess.run([logit], feed_dict=feed)
        print(out[0].shape)