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
 
   

