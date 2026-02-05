from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

LabelCountKey = Union[int, str]


def sample_indices(n: int, max_n: Optional[int], seed: int = 42) -> np.ndarray:
    if (max_n is None) or (max_n <= 0) or (max_n >= n):
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_n, replace=False).astype(np.int64))


def group_indices_by_disease(Y: np.ndarray) -> Dict[int, np.ndarray]:
    Yb = (np.asarray(Y) > 0).astype(np.int8)
    out: Dict[int, np.ndarray] = {}
    for d in range(int(Yb.shape[1])):
        idx = np.where(Yb[:, d] == 1)[0]
        if idx.size > 0:
            out[d] = idx
    return out


def group_indices_by_label_count(Y: np.ndarray) -> Dict[LabelCountKey, np.ndarray]:
    Yb = (np.asarray(Y) > 0).astype(np.int8)
    lc = Yb.sum(axis=1)
    out: Dict[LabelCountKey, np.ndarray] = {}
    for k in (1, 2, 3):
        idx = np.where(lc == k)[0]
        if idx.size > 0:
            out[k] = idx
    idx4 = np.where(lc >= 4)[0]
    if idx4.size > 0:
        out["4+"] = idx4
    return out


def l2_normalize(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(E, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def ndcg_at_k(rels: np.ndarray, k: int) -> float:
    r = np.asarray(rels[:k], dtype=np.float32)
    if r.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, r.size + 2, dtype=np.float32))
    dcg = float((r * discounts).sum())
    ideal = np.sort(r)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _merge_topk(
    prev_scores: np.ndarray,
    prev_idx: np.ndarray,
    new_scores: np.ndarray,
    new_idx: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.concatenate([prev_scores, new_scores], axis=1)
    idxs = np.concatenate([prev_idx, new_idx], axis=1)
    kk = min(int(k), int(scores.shape[1]))
    part = np.argpartition(-scores, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(scores.shape[0])[:, None]
    top_scores = scores[row, part]
    top_idx = idxs[row, part]
    order = np.argsort(-top_scores, axis=1)
    top_scores = top_scores[row, order]
    top_idx = top_idx[row, order]
    if kk < int(k):
        pad = int(k) - kk
        top_scores = np.concatenate(
            [top_scores, np.full((top_scores.shape[0], pad), -np.inf, dtype=top_scores.dtype)],
            axis=1,
        )
        top_idx = np.concatenate(
            [top_idx, np.full((top_idx.shape[0], pad), -1, dtype=top_idx.dtype)],
            axis=1,
        )
    return top_scores, top_idx


def topk_cosine_indices_chunked(
    E: np.ndarray,
    q_idx: np.ndarray,
    k: int = 10,
    chunk_size: int = 200_000,
    exclude_self: bool = True,
) -> np.ndarray:
    k = int(k)
    chunk_size = int(chunk_size)
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    X = np.asarray(E, dtype=np.float32)
    N, _ = X.shape
    q = np.asarray(q_idx, dtype=np.int64)
    Q = int(q.size)
    if Q == 0:
        return np.empty((0, k), dtype=np.int64)

    kk0 = min(k, max(N - (1 if exclude_self else 0), 0))
    if kk0 <= 0:
        return np.full((Q, k), -1, dtype=np.int64)

    Qemb = X[q]
    top_scores = np.full((Q, k), -np.inf, dtype=np.float32)
    top_idx = np.full((Q, k), -1, dtype=np.int64)

    for s in range(0, N, chunk_size):
        t = min(N, s + chunk_size)
        chunk = X[s:t]
        sims = Qemb @ chunk.T

        if exclude_self:
            mask = (q >= s) & (q < t)
            if mask.any():
                rows = np.where(mask)[0]
                cols = (q[rows] - s).astype(np.int64)
                sims[rows, cols] = -1e9

        kk = min(k, int(sims.shape[1]))
        part = np.argpartition(-sims, kth=kk - 1, axis=1)[:, :kk]
        row = np.arange(Q)[:, None]
        cand_scores = sims[row, part]
        cand_idx = part + s
        top_scores, top_idx = _merge_topk(top_scores, top_idx, cand_scores, cand_idx, k)

    return top_idx


def eval_retrieval_multilabel_fast(
    E: np.ndarray,
    Y: np.ndarray,
    k: int = 10,
    q_idx: Optional[np.ndarray] = None,
    pos_mode: str = "overlap",
    overlap_t: int = 2,
    jaccard_t: float = 0.5,
    chunk_size: int = 200_000,
    exclude_self: bool = True,
) -> Dict[str, Any]:
    k = int(k)
    chunk_size = int(chunk_size)
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    pm = (pos_mode or "").lower()
    use_overlap = pm == "overlap"
    use_jaccard = pm == "jaccard"
    if (not use_overlap) and (not use_jaccard):
        raise ValueError("pos_mode must be 'overlap' or 'jaccard'")

    X = l2_normalize(E)
    Yb = (np.asarray(Y) > 0).astype(np.int8)
    N = int(X.shape[0])

    q = np.arange(N, dtype=np.int64) if q_idx is None else np.asarray(q_idx, dtype=np.int64)
    y_sum = Yb.sum(axis=1)
    q = q[y_sum[q] > 0]
    if q.size == 0:
        return {"hit@k": 0.0, "ndcg@k": 0.0, "valid_queries": 0}

    topk_idx = topk_cosine_indices_chunked(
        E=X,
        q_idx=q,
        k=k,
        chunk_size=chunk_size,
        exclude_self=bool(exclude_self),
    )

    hit = 0
    ndcg_sum = 0.0

    for qi, i in enumerate(q):
        nbr = topk_idx[qi]
        nbr = nbr[nbr >= 0]
        if nbr.size == 0:
            continue

        overlap = (Yb[nbr] & Yb[i]).sum(axis=1).astype(np.float32)

        if use_overlap:
            rel = overlap
            is_pos = overlap >= float(overlap_t)
        else:
            union = (Yb[nbr] | Yb[i]).sum(axis=1).astype(np.float32)
            jac = overlap / np.maximum(union, 1.0)
            rel = jac
            is_pos = jac >= float(jaccard_t)

        if is_pos.any():
            hit += 1
        ndcg_sum += ndcg_at_k(rel, k)

    Q = int(q.size)
    return {"hit@k": float(hit / Q), "ndcg@k": float(ndcg_sum / Q), "valid_queries": Q}


def eval_random_baseline_hit(
    Y: np.ndarray,
    k: int,
    q_idx: np.ndarray,
    pos_mode: str = "overlap",
    overlap_t: int = 2,
    jaccard_t: float = 0.5,
    exclude_self: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")

    pm = (pos_mode or "").lower()
    use_overlap = pm == "overlap"
    use_jaccard = pm == "jaccard"
    if (not use_overlap) and (not use_jaccard):
        raise ValueError("pos_mode must be 'overlap' or 'jaccard'")

    rng = np.random.default_rng(int(seed))
    Yb = (np.asarray(Y) > 0).astype(np.int8)
    N = int(Yb.shape[0])

    q = np.asarray(q_idx, dtype=np.int64)
    q = q[Yb.sum(axis=1)[q] > 0]
    if q.size == 0:
        return {"hit@k_rand": 0.0, "valid_queries": 0}

    hit = 0
    for i in q:
        if exclude_self:
            if N <= 1:
                continue
            cand = rng.integers(0, N - 1, size=k, dtype=np.int64)
            cand = np.where(cand >= i, cand + 1, cand)
        else:
            cand = rng.integers(0, N, size=k, dtype=np.int64)

        overlap = (Yb[cand] & Yb[i]).sum(axis=1).astype(np.float32)

        if use_overlap:
            is_pos = overlap >= float(overlap_t)
        else:
            union = (Yb[cand] | Yb[i]).sum(axis=1).astype(np.float32)
            jac = overlap / np.maximum(union, 1.0)
            is_pos = jac >= float(jaccard_t)

        if is_pos.any():
            hit += 1

    Q = int(q.size)
    return {"hit@k_rand": float(hit / Q), "valid_queries": Q}


def eval_retrieval_with_macro_and_lift(
    E: np.ndarray,
    Y: np.ndarray,
    cfg: "RetrievalConfig",
    max_queries: Optional[int] = 20_000,
    max_corpus: Optional[int] = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    Y = np.asarray(Y)
    N = int(Y.shape[0])

    corpus_idx = sample_indices(N, max_corpus, seed=int(seed))
    q_sub = sample_indices(int(corpus_idx.size), max_queries, seed=int(seed) + 1)

    E_sub = E[corpus_idx]
    Y_sub = Y[corpus_idx]

    overall = eval_retrieval_multilabel_fast(
        E=E_sub,
        Y=Y_sub,
        k=cfg.k,
        q_idx=q_sub,
        pos_mode=cfg.pos_mode,
        overlap_t=cfg.overlap_t,
        jaccard_t=cfg.jaccard_t,
        chunk_size=cfg.chunk_size,
        exclude_self=cfg.exclude_self,
    )

    rand = eval_random_baseline_hit(
        Y=Y_sub,
        k=cfg.k,
        q_idx=q_sub,
        pos_mode=cfg.pos_mode,
        overlap_t=cfg.overlap_t,
        jaccard_t=cfg.jaccard_t,
        exclude_self=cfg.exclude_self,
        seed=int(seed) + 2,
    )

    lift = float(overall["hit@k"] / max(float(rand["hit@k_rand"]), 1e-12)) if overall["valid_queries"] > 0 else 0.0

    by_disease: Dict[int, Any] = {}
    for d, idx in group_indices_by_disease(Y_sub).items():
        qd = q_sub[np.isin(q_sub, idx)]
        if qd.size == 0:
            continue
        by_disease[d] = eval_retrieval_multilabel_fast(
            E=E_sub,
            Y=Y_sub,
            k=cfg.k,
            q_idx=qd,
            pos_mode=cfg.pos_mode,
            overlap_t=cfg.overlap_t,
            jaccard_t=cfg.jaccard_t,
            chunk_size=cfg.chunk_size,
            exclude_self=cfg.exclude_self,
        )

    by_lc: Dict[LabelCountKey, Any] = {}
    for g, idx in group_indices_by_label_count(Y_sub).items():
        qg = q_sub[np.isin(q_sub, idx)]
        if qg.size == 0:
            continue
        by_lc[g] = eval_retrieval_multilabel_fast(
            E=E_sub,
            Y=Y_sub,
            k=cfg.k,
            q_idx=qg,
            pos_mode=cfg.pos_mode,
            overlap_t=cfg.overlap_t,
            jaccard_t=cfg.jaccard_t,
            chunk_size=cfg.chunk_size,
            exclude_self=cfg.exclude_self,
        )

    return {
        "subset": {"corpus_n": int(corpus_idx.size), "query_n": int(q_sub.size)},
        "overall": overall,
        "random": rand,
        "lift": lift,
        "macro_by_disease": by_disease,
        "macro_by_label_count": by_lc,
    }


def gini_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(counts, dtype=np.float64)
    x = x[x >= 0]
    if x.size == 0:
        return 0.0
    x.sort()
    total = float(x.sum())
    if total <= eps:
        return 0.0
    n = int(x.size)
    i = np.arange(1, n + 1, dtype=np.float64)
    g = (2.0 * (i * x).sum()) / (n * total) - (n + 1.0) / n
    return float(np.clip(g, 0.0, 1.0))


def perplexity_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(counts, dtype=np.float64)
    total = float(x.sum())
    if total <= eps:
        return 0.0
    p = x / total
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H))


def sid_level_stats(sid_codes: np.ndarray, V: int) -> Dict[str, Any]:
    raw = np.asarray(sid_codes, dtype=np.int64).reshape(-1)
    raw_n = int(raw.size)

    x = raw[(raw >= 0) & (raw < int(V))]
    valid_n = int(x.size)
    drop_ratio = float(1.0 - valid_n / max(raw_n, 1))

    counts = np.bincount(x, minlength=int(V))[: int(V)].astype(np.int64)

    total = int(counts.sum())
    util = float((counts > 0).sum() / max(int(V), 1))
    gini = gini_from_counts(counts)
    gini_nonzero = gini_from_counts(counts[counts > 0]) if (counts > 0).any() else 0.0
    perp = perplexity_from_counts(counts)
    top1 = float(counts.max() / max(total, 1)) if total > 0 else 0.0
    top10 = float(np.sort(counts)[::-1][:10].sum() / max(total, 1)) if total > 0 else 0.0

    return {
        "V": int(V),
        "n": int(total),
        "raw_n": int(raw_n),
        "valid_n": int(valid_n),
        "drop_ratio": float(drop_ratio),
        "gini": float(gini),
        "gini_nonzero": float(gini_nonzero),
        "util": float(util),
        "perplexity": float(perp),
        "perplexity_norm": float(perp / max(int(V), 1)),
        "top1_share": float(top1),
        "top10_share": float(top10),
        "counts": counts,
    }


def sid_joint_stats(sids_3: np.ndarray, V_list: Tuple[int, int, int] = (512, 512, 512)) -> Dict[str, Any]:
    x = np.asarray(sids_3, dtype=np.int64)
    V1, V2, V3 = (int(V_list[0]), int(V_list[1]), int(V_list[2]))

    if not (x.ndim == 2 and x.shape[1] == 3):
        raise ValueError("sids_3 must have shape [N,3]")

    m = (
        (x[:, 0] >= 0) & (x[:, 0] < V1)
        & (x[:, 1] >= 0) & (x[:, 1] < V2)
        & (x[:, 2] >= 0) & (x[:, 2] < V3)
    )
    x = x[m]
    n = int(x.shape[0])
    if n == 0:
        return {"n": 0, "unique": 0, "perplexity": 0.0, "top1_share": 0.0, "top10_share": 0.0, "unique_ratio_over_n": 0.0}

    key = x[:, 0].astype(np.int64) * (V2 * V3) + x[:, 1].astype(np.int64) * V3 + x[:, 2].astype(np.int64)
    uniq, counts = np.unique(key, return_counts=True)
    counts = counts.astype(np.int64)

    perp = perplexity_from_counts(counts)
    top1 = float(counts.max() / max(n, 1))
    top10 = float(np.sort(counts)[::-1][:10].sum() / max(n, 1))

    return {
        "n": int(n),
        "unique": int(uniq.size),
        "perplexity": float(perp),
        "top1_share": float(top1),
        "top10_share": float(top10),
        "unique_ratio_over_n": float(uniq.size / max(n, 1)),
    }


def sid_3level_report(sids_3: np.ndarray, V_list: Tuple[int, int, int] = (512, 512, 512)) -> Dict[str, Any]:
    x = np.asarray(sids_3, dtype=np.int64)
    if not (x.ndim == 2 and x.shape[1] == 3):
        raise ValueError("sids_3 must have shape [N,3]")
    V1, V2, V3 = (int(V_list[0]), int(V_list[1]), int(V_list[2]))
    return {
        "level1": sid_level_stats(x[:, 0], V1),
        "level2": sid_level_stats(x[:, 1], V2),
        "level3": sid_level_stats(x[:, 2], V3),
        "joint": sid_joint_stats(x, V_list=(V1, V2, V3)),
    }


def conditional_sid_report(
    sids_3: np.ndarray,
    Y: np.ndarray,
    V_list: Tuple[int, int, int] = (512, 512, 512),
) -> Dict[str, Any]:
    x = np.asarray(sids_3, dtype=np.int64)
    Yb = (np.asarray(Y) > 0).astype(np.int8)
    if x.shape[0] != Yb.shape[0]:
        raise ValueError("sids_3 and Y must have the same number of rows")

    out: Dict[str, Any] = {"by_disease": {}, "by_label_count": {}}

    for d, idx in group_indices_by_disease(Yb).items():
        out["by_disease"][f"d{d}"] = {"n": int(idx.size), "report": sid_3level_report(x[idx], V_list=V_list)}

    for g, idx in group_indices_by_label_count(Yb).items():
        out["by_label_count"][f"count{g}"] = {"n": int(idx.size), "report": sid_3level_report(x[idx], V_list=V_list)}

    return out


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 10
    pos_mode: str = "overlap"
    overlap_t: int = 2
    jaccard_t: float = 0.5
    chunk_size: int = 200_000
    exclude_self: bool = True


def eval_retrieval(E: np.ndarray, Y: np.ndarray, cfg: RetrievalConfig, q_idx: Optional[np.ndarray] = None) -> Dict[str, Any]:
    return eval_retrieval_multilabel_fast(
        E=E,
        Y=Y,
        k=cfg.k,
        q_idx=q_idx,
        pos_mode=cfg.pos_mode,
        overlap_t=cfg.overlap_t,
        jaccard_t=cfg.jaccard_t,
        chunk_size=cfg.chunk_size,
        exclude_self=cfg.exclude_self,
    )


def eval_retrieval_full_report(
    E: np.ndarray,
    Y: np.ndarray,
    cfg: RetrievalConfig,
    max_queries: Optional[int] = 20_000,
    max_corpus: Optional[int] = 50_000,
    seed: int = 42,
) -> Dict[str, Any]:
    return eval_retrieval_with_macro_and_lift(
        E=E,
        Y=Y,
        cfg=cfg,
        max_queries=max_queries,
        max_corpus=max_corpus,
        seed=seed,
    )

