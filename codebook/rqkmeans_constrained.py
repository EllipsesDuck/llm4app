import time
import numpy as np
import polars as pl
from typing import Tuple, List, Union, Optional

try:
    from k_means_constrained import KMeansConstrained
    HAS_CONSTRAINED = True
except ImportError:
    HAS_CONSTRAINED = False
    print("Warning: k-means-constrained not available.")
    print("Install with: pip install k-means-constrained")


def _normalize_random_state(random_state: Optional[Union[int, np.random.Generator]]) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def balanced_kmeans_level_constrained(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-7,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_CONSTRAINED:
        raise ImportError(
            "k-means-constrained is required for balanced_kmeans_level_constrained. "
            "Install via: pip install k-means-constrained"
        )

    start_time = time.time()

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {X.shape}")
    n, d = X.shape

    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    if K > n:
        raise ValueError(f"K must be <= number of samples: K={K}, N={n}")

    X = X.astype(np.float32, copy=False)

    avg_size = n / K
    imbalance_ratio = 0.05  
    min_size = max(1, int(avg_size * (1.0 - imbalance_ratio)))
    max_size = max(min_size, int(np.ceil(avg_size * (1.0 + imbalance_ratio))))

    if verbose:
        print(f"    Starting constrained K-means with K={K}, N={n}, D={d}")
        print(f"    Cluster size constraints: [{min_size}, {max_size}]")

    rng = _normalize_random_state(random_state)
    seed = int(rng.integers(0, 2**31 - 1))

    try:
        import multiprocessing
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    except Exception:
        n_jobs = 1

    kmeans = KMeansConstrained(
        n_clusters=K,
        size_min=min_size,
        size_max=max_size,
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        n_init=3,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    labels = kmeans.fit_predict(X)        # (N,)
    centroids = kmeans.cluster_centers_   # (K, D)

    elapsed = time.time() - start_time
    print(f"[Time] balanced_kmeans_level_constrained (K={K}): {elapsed:.2f}s")

    if verbose:
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"    #clusters = {len(unique_labels)}")
        print(f"    Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")

    return labels, centroids


def residual_kmeans_constrained(
    X: np.ndarray,
    K: Union[int, List[int]],
    L: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[Union[int, np.random.Generator]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    total_start = time.time()

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D), got shape {X.shape}")
    n, d = X.shape

    Ks = [K] * L if isinstance(K, int) else list(K)
    if len(Ks) != L:
        raise ValueError(f"Length of K list {len(Ks)} must equal L={L}")

    X32 = X.astype(np.float32, copy=False)
    R = X32.copy()  # residual
    codes_all = np.empty((L, n), dtype=np.int32)
    codebooks: List[np.ndarray] = []

    base_rng = _normalize_random_state(random_state)

    for l in range(L):
        level_start = time.time()
        k_l = Ks[l]

        if verbose:
            mse_before = float(np.mean(R ** 2))
            print(f"\n=== Level {l+1}/{L} | K={k_l} ===")
            print(f"  Residual MSE before clustering: {mse_before:.6f}")

        seed_l = int(base_rng.integers(0, 2**31 - 1))

        codes_l, C_l = balanced_kmeans_level_constrained(
            R,
            K=k_l,
            max_iter=max_iter,
            tol=tol,
            random_state=seed_l,
            verbose=verbose,
        )

        if C_l.dtype != R.dtype:
            C_l = C_l.astype(R.dtype, copy=False)

        codes_all[l] = codes_l  # (N,)
        codebooks.append(C_l)   # (K_l, D)

        R -= C_l[codes_l]

        elapsed_l = time.time() - level_start
        print(f"[Time] Level {l+1}: {elapsed_l:.2f}s")

        if verbose:
            mse_after = float(np.mean(R ** 2))
            print(f"  Residual MSE after Level {l+1}: {mse_after:.6f}")

    recon32 = X32 - R
    recon = recon32.astype(X.dtype, copy=False)

    total_elapsed = time.time() - total_start
    print(f"[Time] residual_kmeans_constrained total: {total_elapsed:.2f}s")

    if verbose:
        total_mse = float(np.mean((X32 - recon32) ** 2))
        print(f"\nFinal reconstruction MSE (float32 space): {total_mse:.6f}")

    return codes_all, codebooks, recon


def deal_with_deduplicate(df: pl.DataFrame) -> pl.DataFrame:
    if "codes" not in df.columns:
        raise ValueError("DataFrame must contain a 'codes' column")

    df_with_index = df.with_row_index("row_id")

    df_with_index = df_with_index.with_columns(
        occurrence=pl.col("row_id").rank("dense").over("codes") - 1
    )

    df_with_index = df_with_index.with_columns(
        group_size=pl.len().over("codes")
    )

    result_df = df_with_index.with_columns(
        pl.when(pl.col("group_size") > 1)
        .then(pl.col("codes") + pl.col("occurrence").cast(pl.Int64))
        .otherwise(pl.col("codes"))
        .alias("codes")
    ).drop(["row_id", "occurrence", "group_size"])

    return result_df



def analyze_codes(
    codes: np.ndarray,
    title: str = "",
    verbose: bool = True,
) -> None:
    codes = np.asarray(codes)

    if codes.ndim != 2:
        raise ValueError(f"`codes` must be 2D, got shape {codes.shape}")

    if codes.shape[0] < codes.shape[1]:
        N, M = codes.shape[1], codes.shape[0]
        codes_for_stats = codes.T
    else:
        N, M = codes.shape
        codes_for_stats = codes

    if verbose:
        if title:
            print(f"\n{title}")
        print(f"  Total items: {N}")
        for l in range(M):
            unique_count = np.unique(codes_for_stats[:, l]).size
            print(f"  Level {l+1}: unique codes = {unique_count}")

        unique_paths = np.unique(codes_for_stats, axis=0).shape[0]
        collision_rate = 1.0 - unique_paths / float(N)
        print(f"  Unique full-paths: {unique_paths}")
        print(f"  Collision rate: {collision_rate:.4f}")

