import argparse
import json
import os
from collections import defaultdict
from typing import Optional, Tuple

import faiss
import numpy as np
from tqdm import tqdm


def pairwise_sq_dists_batch(
    X: np.ndarray,
    C: np.ndarray,
    C_norm2: Optional[np.ndarray] = None,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32, order="C")
    C = np.asarray(C, dtype=np.float32, order="C")

    if X.ndim != 2 or C.ndim != 2:
        raise ValueError(f"X and C must be 2D, got X.shape={X.shape}, C.shape={C.shape}")
    if X.shape[1] != C.shape[1]:
        raise ValueError(f"Dim mismatch: X.shape[1]={X.shape[1]} vs C.shape[1]={C.shape[1]}")

    if C_norm2 is None:
        C_norm2 = np.sum(C * C, axis=1)  # (K,)

    X_norm2 = np.sum(X * X, axis=1, keepdims=True)  # (B, 1)
    dots = X @ C.T  # (B, K)
    D = X_norm2 + C_norm2[None, :] - 2.0 * dots
    return D


def train_faiss_rq(
    data: np.ndarray,
    num_levels: int = 3,
    codebook_size: int = 256,
    verbose: bool = True,
) -> faiss.ResidualQuantizer:
    data = np.asarray(data, dtype=np.float32, order="C")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    N, d = data.shape

    if codebook_size <= 0:
        raise ValueError(f"codebook_size must be > 0, got {codebook_size}")
    nbits = int(np.log2(codebook_size))
    if (1 << nbits) != codebook_size:
        raise ValueError(f"codebook_size must be a power of 2, got {codebook_size}")

    if verbose:
        print("Training FAISS ResidualQuantizer")
        print(
            f"  data={N}  dim={d}  levels={num_levels}  "
            f"codebook={codebook_size}  total_codes={codebook_size ** num_levels:,}"
        )

    rq = faiss.ResidualQuantizer(d, num_levels, nbits)
    rq.train_type = faiss.ResidualQuantizer.Train_default
    rq.max_beam_size = 1

    rq.train(data)
    if verbose:
        print("  training completed\n")
    return rq


def unpack_rq_codes(
    codes: np.ndarray,
    nbits: int,
    num_levels: int,
) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8, order="C")
    if codes.ndim == 1:
        # single byte string per vector
        codes = codes.reshape(-1, codes.shape[0])

    N, M_bytes = codes.shape
    if N == 0:
        return np.empty((0, num_levels), dtype=np.int32)

    packed_ints = np.zeros(N, dtype=np.int64)
    for i in range(M_bytes):
        packed_ints |= codes[:, i].astype(np.int64) << (8 * i)

    unpacked_codes = np.zeros((N, num_levels), dtype=np.int32)
    mask = (1 << nbits) - 1  

    for i in range(num_levels):
        unpacked_codes[:, i] = (packed_ints >> (i * nbits)) & mask

    return unpacked_codes


def encode_with_rq(
    rq: faiss.ResidualQuantizer,
    data: np.ndarray,
    codebook_size: int,
    verbose: bool = True,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32, order="C")
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")

    nbits = int(np.log2(codebook_size))
    if verbose:
        print(f"Encoding {data.shape[0]} vectors ...")

    codes_packed = rq.compute_codes(data)

    if nbits % 8 == 0:
        codes = codes_packed.astype(np.int32)
    else:
        # handle bit-packed codes
        if codes_packed.ndim == 1:
            n_bytes = (rq.M * nbits + 7) // 8
            codes_packed = codes_packed.reshape(-1, n_bytes)
        codes = unpack_rq_codes(codes_packed, nbits=nbits, num_levels=rq.M)

    codes = np.asarray(codes, dtype=np.int32)
    if codes.ndim != 2 or codes.shape[1] != rq.M:
        raise ValueError(
            f"Unexpected codes shape {codes.shape}, expected (N, {rq.M})"
        )

    if verbose:
        print(f"  done, codes.shape={codes.shape}\n")
    return codes


def get_rq_codebooks(rq: faiss.ResidualQuantizer) -> np.ndarray:
    M, d = rq.M, rq.d
    cb_flat = faiss.vector_to_array(rq.codebooks).astype(np.float32)
    if cb_flat.size % (M * d) != 0:
        raise ValueError(
            f"Codebooks size {cb_flat.size} is not divisible by M*d={M*d}."
        )

    total_centers = cb_flat.size // d
    K = total_centers // M  # centers per level
    if K * M * d != cb_flat.size:
        raise ValueError(
            f"Cannot reshape codebooks into (M={M}, K=?, d={d}), "
            f"computed K={K}, cb_flat.size={cb_flat.size}"
        )

    return cb_flat.reshape(M, K, d)  # (M, K, d)


def compute_residuals_upto_level(
    rq: faiss.ResidualQuantizer,
    data: np.ndarray,
    codes: np.ndarray,
    upto_level: int,
    codebooks: Optional[np.ndarray] = None,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32, order="C")
    codes = np.asarray(codes, dtype=np.int32, order="C")

    if codebooks is None:
        codebooks = get_rq_codebooks(rq)

    if data.ndim != 2 or codes.ndim != 2:
        raise ValueError(f"data and codes must be 2D, got {data.shape}, {codes.shape}")
    N, d = data.shape
    if codes.shape[0] != N:
        raise ValueError(f"Mismatch: data N={N} vs codes N={codes.shape[0]}")
    M = codes.shape[1]
    if not (0 <= upto_level <= M):
        raise ValueError(f"upto_level must be in [0, {M}], got {upto_level}")
    if codebooks.shape[0] < upto_level:
        raise ValueError(
            f"codebooks has only {codebooks.shape[0]} levels, "
            f"but upto_level={upto_level}"
        )

    residuals = data.copy()
    for l in range(upto_level):
        residuals -= codebooks[l][codes[:, l]]

    return residuals


def estimate_tau(
    residuals: np.ndarray,
    centroids: np.ndarray,
    sample_size: int = 4000,
    percentile: float = 90,
    min_tau: float = 1e-6,
) -> float:
    residuals = np.asarray(residuals, dtype=np.float32, order="C")
    centroids = np.asarray(centroids, dtype=np.float32, order="C")

    N = residuals.shape[0]
    if N == 0:
        return min_tau

    idx = np.random.choice(N, size=min(sample_size, N), replace=False)
    X = residuals[idx]
    Cn2 = np.sum(centroids * centroids, axis=1)
    D = pairwise_sq_dists_batch(X, centroids, Cn2)  # (n_s, K)

    # distance spread relative to the best centroid per sample
    D_min = D.min(axis=1, keepdims=True)
    spread = np.percentile(D - D_min, percentile, axis=1)
    tau = float(np.median(spread) * 0.1)
    return max(tau, min_tau)


def sinkhorn_balance_level(
    residuals: np.ndarray,
    centroids: np.ndarray,
    capacities: Optional[np.ndarray] = None,
    *,
    batch_size: int = 8192,
    iters: int = 30,
    tau: Optional[float] = None,
    verbose: bool = True,
    topk: int = 32,
    seed: int = 42,
) -> np.ndarray:
    import ot  # POT

    residuals = np.asarray(residuals, dtype=np.float32, order="C")
    centroids = np.asarray(centroids, dtype=np.float32, order="C")

    N, d = residuals.shape
    K = centroids.shape[0]

    if K <= 0:
        raise ValueError("K must be > 0")
    if capacities is None:
        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1
    else:
        capacities = np.asarray(capacities, dtype=np.int64)
    if capacities.shape != (K,):
        raise ValueError(f"capacities must be shape (K,), got {capacities.shape}")
    if capacities.sum() != N:
        raise ValueError(
            f"capacities must sum to N={N}, got sum={capacities.sum()}"
        )

    if tau is None:
        tau = estimate_tau(residuals, centroids)

    if verbose:
        print(
            f"  Sinkhorn level: N={N}  K={K}  tau={tau:.5g}  "
            f"iters={iters}  batch={batch_size}"
        )

    # uniform source distribution
    a = np.full(N, 1.0 / N, dtype=np.float64)
    # capacities normalized to 1
    b = capacities.astype(np.float64) / float(N)
    Cn2 = np.sum(centroids * centroids, axis=1).astype(np.float32)

    # full cost matrix (might be heavy for very large N,K)
    D_full = pairwise_sq_dists_batch(residuals, centroids, Cn2).astype(np.float64)

    # sinkhorn transport matrix P
    P = ot.sinkhorn(a, b, D_full, reg=tau, numItermax=iters)

    rng = np.random.RandomState(seed)
    remaining = capacities.copy()
    assign = np.empty(N, dtype=np.int32)
    order = np.arange(N)
    rng.shuffle(order)

    for i in order:
        probs = P[i]
        if topk and topk < K:
            # consider only top-k by probability
            cand = np.argpartition(-probs, topk - 1)[:topk]
            cand = cand[np.argsort(-probs[cand])]
        else:
            cand = np.argsort(-probs)

        chosen = -1
        for c in cand:
            if remaining[c] > 0:
                chosen = c
                break

        if chosen < 0:
            # fall back: choose argmax, if also exhausted, pick smallest remaining
            c = int(np.argmax(probs))
            if remaining[c] == 0:
                c = int(np.argmin(remaining))
            chosen = c

        remaining[chosen] -= 1
        assign[i] = chosen

    if verbose:
        used = capacities - remaining
        print(f"    level balanced: min={used.min()}  max={used.max()}")

    return assign


def sinkhorn_uniform_mapping(
    rq: faiss.ResidualQuantizer,
    data: np.ndarray,
    codes: np.ndarray,
    *,
    batch_size: int = 8192,
    iters: int = 30,
    tau: Optional[float] = None,
    verbose: bool = True,
    topk: int = 32,
    seed: int = 42,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32, order="C")
    codes = np.asarray(codes, dtype=np.int32, order="C")

    if data.ndim != 2 or codes.ndim != 2:
        raise ValueError(f"data and codes must be 2D, got {data.shape}, {codes.shape}")
    N, d = data.shape
    if codes.shape[0] != N:
        raise ValueError(f"N mismatch: data={N}, codes={codes.shape[0]}")
    M = codes.shape[1]

    codebooks = get_rq_codebooks(rq)
    K = codebooks.shape[1]

    codes_bal = codes.copy()
    for l in range(M):
        if verbose:
            print(f"\n=== Sinkhorn uniform mapping  level {l+1}/{M} ===")

        # residuals *before* level l
        residuals = compute_residuals_upto_level(
            rq, data, codes_bal, upto_level=l, codebooks=codebooks
        )

        capacities = np.full(K, N // K, dtype=np.int64)
        capacities[: (N % K)] += 1

        new_ids = sinkhorn_balance_level(
            residuals,
            codebooks[l],
            capacities=capacities,
            batch_size=batch_size,
            iters=iters,
            tau=tau,
            verbose=verbose,
            topk=topk,
            seed=seed + l,
        )

        codes_bal[:, l] = new_ids

    return codes_bal


def analyze_codes(
    codes: np.ndarray,
    title: str = "",
    verbose: bool = True,
) -> None:
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape {codes.shape}")

    N, M = codes.shape
    if not verbose:
        return

    if title:
        print(title)
    print(f"  total={N}")

    for l in range(M):
        uniq = np.unique(codes[:, l]).size
        print(f"  L{l+1}: unique={uniq}")

    # number of unique code paths
    unique_paths = np.unique(codes, axis=0).shape[0]
    collision_rate = 1.0 - unique_paths / float(N)
    print(
        f"  unique full-paths={unique_paths}  "
        f"collision_rate={collision_rate:.4f}"
    )


def save_indices_json(
    codes: np.ndarray,
    path: str,
    use_prefix: bool = True,
) -> None:
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2D, got shape {codes.shape}")
    N, M = codes.shape

    tpl = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
    if use_prefix and M > len(tpl):
        raise ValueError(
            f"use_prefix=True only supports up to {len(tpl)} levels, "
            f"but got M={M}"
        )

    idx = {}
    for i, code in enumerate(codes):
        if use_prefix:
            idx[i] = [tpl[j].format(int(c)) for j, c in enumerate(code)]
        else:
            idx[i] = [int(c) for c in code]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx, f, indent=2)
    print("Saved indices:", path)

import numpy as np
import os


def main():
    print("====== Demo: FAISS RQ + Sinkhorn Uniform Mapping ======\n")

    # ------------------------------------------------------
    # 1. 伪造 embedding 数据
    # ------------------------------------------------------
    N = 5000         # 样本数（够大才能看到良好分布效果）
    D = 128          # 向量维度
    LEVELS = 3       # 残差层数
    K = 256          # 每层 codebook 大小（必须是 2^n）

    print(f"Generating fake embeddings: N={N}, D={D}")
    data = np.random.randn(N, D).astype(np.float32)

    # ------------------------------------------------------
    # 2. 训练 FAISS RQ
    # ------------------------------------------------------
    print("\n--- Training FAISS ResidualQuantizer ---")
    rq = train_faiss_rq(
        data,
        num_levels=LEVELS,
        codebook_size=K,
        verbose=True
    )

    # ------------------------------------------------------
    # 3. 编码为 RQ codes
    # ------------------------------------------------------
    print("\n--- Encoding with RQ ---")
    codes = encode_with_rq(
        rq,
        data,
        codebook_size=K,
        verbose=True
    )
    print("codes shape:", codes.shape)    # (N, LEVELS)

    analyze_codes(codes, title="\n[Before Sinkhorn]")

    # ------------------------------------------------------
    # 4. Sinkhorn 多层均匀映射
    # ------------------------------------------------------
    print("\n--- Applying Sinkhorn Uniform Mapping ---")
    balanced_codes = sinkhorn_uniform_mapping(
        rq,
        data,
        codes,
        batch_size=8192,
        iters=30,
        tau=None,         # 自动估计温度
        verbose=True,
        topk=32,
        seed=42
    )

    analyze_codes(balanced_codes, title="\n[After Sinkhorn]")

    # ------------------------------------------------------
    # 5. 保存 JSON index
    # ------------------------------------------------------
    save_path = "./demo_index.json"
    save_indices_json(balanced_codes, save_path)

    print("\nDemo completed! JSON saved:", save_path)


if __name__ == "__main__":
    main()
