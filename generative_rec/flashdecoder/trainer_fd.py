import os
import json
import math
import random
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from flashdecoderv3 import LazyDecoder, compute_sft_loss


# Utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_bucket_split(num_buckets=256, seed=42, train_ratio=0.70, test_ratio=0.20):
    buckets = list(range(num_buckets))
    rng = random.Random(seed)
    rng.shuffle(buckets)

    n_train = round(num_buckets * train_ratio)
    n_test = round(num_buckets * test_ratio)
    n_eval = num_buckets - n_train - n_test

    train_b = sorted(buckets[:n_train])
    test_b = sorted(buckets[n_train:n_train + n_test])
    eval_b = sorted(buckets[n_train + n_test:])
    return train_b, test_b, eval_b


def load_sample_index_labels(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sid2y = {}
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)

        if head == "[":
            arr = json.load(f)
            for r in arr:
                sid = str(r.get("sample_id", "")).strip()
                y = r.get("label_multi", None)
                if sid and y is not None:
                    sid2y[sid] = np.asarray(y, dtype=np.float32)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = str(r.get("sample_id", "")).strip()
                y = r.get("label_multi", None)
                if sid and y is not None:
                    sid2y[sid] = np.asarray(y, dtype=np.float32)

    return sid2y


def try_import_sklearn_metrics():
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        return roc_auc_score, f1_score
    except Exception:
        return None, None


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_auc_f1(
    y_true: np.ndarray,
    y_logit: np.ndarray,
    thr: float = 0.5,
    ignore_value: float = -1.0,
) -> Dict[str, float]:
    roc_auc_score, f1_score = try_import_sklearn_metrics()
    y_prob = sigmoid_np(y_logit)

    out = {"macro_auc": float("nan"), "micro_auc": float("nan"),
           "macro_f1": float("nan"), "micro_f1": float("nan")}

    # ignore -1, map {1 -> 1, 0/-1 -> 0} but mask -1 out
    mask = (y_true != ignore_value)
    y01 = (y_true > 0).astype(np.int32)

    # -------- AUC (ignore -1 per class) --------
    if roc_auc_score is not None:
        aucs = []
        for c in range(y_true.shape[1]):
            m = mask[:, c]
            if m.sum() == 0:
                continue
            yt = y01[m, c]
            yp = y_prob[m, c]
            if yt.max() == yt.min():
                continue
            try:
                aucs.append(float(roc_auc_score(yt, yp)))
            except Exception:
                pass
        out["macro_auc"] = float(np.mean(aucs)) if len(aucs) else float("nan")

        try:
            yt = y01[mask].reshape(-1)
            yp = y_prob[mask].reshape(-1)
            if yt.size > 0 and yt.max() != yt.min():
                out["micro_auc"] = float(roc_auc_score(yt, yp))
        except Exception:
            pass

    # -------- F1 @ thr (ignore -1) --------
    y_pred = (y_prob >= float(thr)).astype(np.int32)

    if f1_score is not None:
        # micro f1
        try:
            yt = y01[mask].reshape(-1)
            yp = y_pred[mask].reshape(-1)
            out["micro_f1"] = float(f1_score(yt, yp, average="binary", zero_division=0))
        except Exception:
            pass

        # macro f1 (mean over classes)
        try:
            f1s = []
            for c in range(y_true.shape[1]):
                m = mask[:, c]
                if m.sum() == 0:
                    continue
                yt = y01[m, c]
                yp = y_pred[m, c]
                f1s.append(float(f1_score(yt, yp, average="binary", zero_division=0)))
            out["macro_f1"] = float(np.mean(f1s)) if len(f1s) else float("nan")
        except Exception:
            pass

    return out



# Bucket I/O
def bucket_dir(tab_root: str, bid: int) -> str:
    return os.path.join(tab_root, f"bucket_{bid:03d}")


def _decode_sid_list(arr) -> List[str]:
    arr = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8", errors="ignore").strip())
        else:
            out.append(str(x).strip())
    return out


def load_tab_bucket(tab_root: str, bid: int) -> Tuple[List[str], np.ndarray]:
    bdir = bucket_dir(tab_root, bid)
    sid_path = os.path.join(bdir, "sample_id.npy")
    emb_path = os.path.join(bdir, "tab_embed.npy")
    sids = _decode_sid_list(np.load(sid_path, allow_pickle=True))
    emb = np.load(emb_path)
    return sids, emb


def load_ts_bucket(ts_root: str, bid: int) -> Tuple[List[str], torch.Tensor]:
    path = os.path.join(ts_root, f"bucket_{bid:03d}_ts_emb.pt")
    obj = torch.load(path, map_location="cpu")
    sids = [str(x) for x in obj["sample_id"]]
    ts_emb = obj["ts_emb"].float()
    return sids, ts_emb


def load_code_bucket(code_root: str, bid: int) -> Tuple[List[str], np.ndarray]:
    codes_path = os.path.join(code_root, f"bucket_{bid:03d}_codes.npy")
    sid_path = os.path.join(code_root, f"bucket_{bid:03d}_sample_id.npy")
    sids = _decode_sid_list(np.load(sid_path, allow_pickle=True))
    codes = np.load(codes_path)
    if codes.ndim == 1:
        codes = codes[:, None]
    return sids, codes.astype(np.int64)


def intersect_by_sample_id(
    tab: Tuple[List[str], np.ndarray],
    ts: Tuple[List[str], torch.Tensor],
    code: Tuple[List[str], np.ndarray],
    sid2y: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tab_sids, tab_emb = tab
    ts_sids, ts_emb = ts
    code_sids, codes = code

    tab_map = {sid: i for i, sid in enumerate(tab_sids)}
    ts_map = {sid: i for i, sid in enumerate(ts_sids)}
    code_map = {sid: i for i, sid in enumerate(code_sids)}

    common = []
    for sid in tab_map.keys():
        if sid in ts_map and sid in code_map and sid in sid2y:
            common.append(sid)

    if len(common) == 0:
        label_dim = len(next(iter(sid2y.values())))
        return (np.zeros((0, tab_emb.shape[1]), dtype=np.float32),
                np.zeros((0, ts_emb.shape[1]), dtype=np.float32),
                np.zeros((0, codes.shape[1]), dtype=np.int64),
                np.zeros((0, label_dim), dtype=np.float32))

    tab_out = np.stack([tab_emb[tab_map[sid]] for sid in common], axis=0).astype(np.float32)
    ts_out = torch.stack([ts_emb[ts_map[sid]] for sid in common], dim=0).numpy().astype(np.float32)
    code_out = np.stack([codes[code_map[sid]] for sid in common], axis=0).astype(np.int64)
    y_out = np.stack([sid2y[sid] for sid in common], axis=0).astype(np.float32)
    return tab_out, ts_out, code_out, y_out


# Dataset
class BucketMixDataset(Dataset):
    def __init__(
        self,
        bucket_ids: List[int],
        tab_root: str,
        ts_root: str,
        code_root: str,
        sid2y: Dict[str, np.ndarray],
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        code_offset: int = 3,        
        cache_size: int = 2,
        verbose: bool = True,
    ):
        self.bucket_ids = bucket_ids
        self.tab_root = tab_root
        self.ts_root = ts_root
        self.code_root = code_root
        self.sid2y = sid2y

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.code_offset = int(code_offset)

        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []

        self.index = []
        self.bucket_sizes = {}

        it = bucket_ids
        if verbose:
            it = tqdm(bucket_ids, desc="Index buckets", leave=True)

        for bid in it:
            tab = load_tab_bucket(self.tab_root, bid)
            ts = load_ts_bucket(self.ts_root, bid)
            code = load_code_bucket(self.code_root, bid)
            tab_out, ts_out, code_out, y_out = intersect_by_sample_id(tab, ts, code, self.sid2y)
            n = tab_out.shape[0]
            self.bucket_sizes[bid] = n
            for i in range(n):
                self.index.append((bid, i))

        random.shuffle(self.index)

        self.tab_dim = None
        self.ts_dim = None
        self.code_len = None
        self.label_dim = None
        for bid in bucket_ids:
            if self.bucket_sizes.get(bid, 0) > 0:
                tab = load_tab_bucket(self.tab_root, bid)
                ts = load_ts_bucket(self.ts_root, bid)
                code = load_code_bucket(self.code_root, bid)
                tab_out, ts_out, code_out, y_out = intersect_by_sample_id(tab, ts, code, self.sid2y)
                self.tab_dim = tab_out.shape[1]
                self.ts_dim = ts_out.shape[1]
                self.code_len = code_out.shape[1]
                self.label_dim = y_out.shape[1]
                break

    def __len__(self):
        return len(self.index)

    def _put_cache(self, bid, tab_out, ts_out, code_out, y_out):
        if bid in self._cache:
            return
        self._cache[bid] = {"tab": tab_out, "ts": ts_out, "codes": code_out, "y": y_out}
        self._cache_order.append(bid)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            if old in self._cache:
                del self._cache[old]

    def _get_bucket(self, bid):
        if bid in self._cache:
            if bid in self._cache_order:
                self._cache_order.remove(bid)
            self._cache_order.append(bid)
            return self._cache[bid]

        tab = load_tab_bucket(self.tab_root, bid)
        ts = load_ts_bucket(self.ts_root, bid)
        code = load_code_bucket(self.code_root, bid)
        tab_out, ts_out, code_out, y_out = intersect_by_sample_id(tab, ts, code, self.sid2y)
        self._put_cache(bid, tab_out, ts_out, code_out, y_out)
        return self._cache[bid]

    def __getitem__(self, idx):
        bid, i = self.index[idx]
        bucket = self._get_bucket(bid)

        tab = bucket["tab"][i]
        ts = bucket["ts"][i]
        codes = bucket["codes"][i]
        y = bucket["y"][i]

        # codes offset: avoid collision with pad/bos/eos
        codes_off = codes + self.code_offset

        # target_ids: [BOS] + codes + [EOS]
        seq = [self.bos_id] + codes_off.tolist() + [self.eos_id]
        target_ids = np.asarray(seq, dtype=np.int64)

        return {
            "target_ids": target_ids,
            "tab": tab.astype(np.float32),
            "ts": ts.astype(np.float32),
            "y": y.astype(np.float32),
        }


def collate_fn(batch: List[Dict[str, Any]]):
    target_ids = torch.tensor(np.stack([b["target_ids"] for b in batch], axis=0), dtype=torch.long)
    tab = torch.tensor(np.stack([b["tab"] for b in batch], axis=0), dtype=torch.float32)
    ts = torch.tensor(np.stack([b["ts"] for b in batch], axis=0), dtype=torch.float32)
    tab = torch.nan_to_num(tab, nan=0.0, posinf=0.0, neginf=0.0)
    ts  = torch.nan_to_num(ts,  nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.tensor(np.stack([b["y"] for b in batch], axis=0), dtype=torch.float32)

    user_static = tab.unsqueeze(1)  # (B,1,Dtab)
    short_term = ts.unsqueeze(1)    # (B,1,Dts)

    return {
        "target_ids": target_ids,
        "user_static": user_static,
        "short_term": short_term,
        "long_term": None,
        "y": y,
    }


# Model wrapper (adds adapters + label head)
class FlashDecoderSFT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tab_dim: int,
        ts_dim: int,
        label_dim: int,
        ctx_dim: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads_q: int = 4,
        gkv: int = 1,
        d_ff: int = 512,
        pad_id: int = 0,
        bos_id: int = 1,
        attn_drop: float = 0.0,
        resid_drop: float = 0.0,
        use_label_head: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.use_label_head = use_label_head

        self.tab_adapter = nn.Linear(tab_dim, ctx_dim, bias=False) if tab_dim != ctx_dim else nn.Identity()
        self.ts_adapter = nn.Linear(ts_dim, ctx_dim, bias=False) if ts_dim != ctx_dim else nn.Identity()

        self.decoder = LazyDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads_q=n_heads_q,
            gkv=gkv,
            d_ff=d_ff,
            d_ctx_in=ctx_dim,
            pad_id=pad_id,
            bos_id=bos_id,
            attn_drop=attn_drop,
            resid_drop=resid_drop,
        )

        if use_label_head:
            self.label_head = nn.Linear(d_model, label_dim, bias=True)

    def _adapt_ctx(self, user_static, short_term, long_term=None):
        user_static = self.tab_adapter(user_static)
        short_term = self.ts_adapter(short_term)
        return user_static, short_term, long_term

    def forward(self, target_ids, user_static, short_term, long_term=None):
        user_static, short_term, long_term = self._adapt_ctx(user_static, short_term, long_term)
        out = self.decoder(
            target_ids=target_ids,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            return_hidden=False,
        )
        return {"logits": out["logits"]}

    def forward_with_hidden(self, target_ids, user_static, short_term, long_term=None):
        user_static, short_term, long_term = self._adapt_ctx(user_static, short_term, long_term)
        out = self.decoder(
            target_ids=target_ids,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            return_hidden=True,
        )
        return out  # {"logits":..., "hidden":...}

    def logits_to_label(self, hidden: torch.Tensor, which: str = "last") -> Optional[torch.Tensor]:
        if not self.use_label_head:
            return None
        if which == "bos":
            h = hidden[:, 0, :]
        elif which == "last":
            h = hidden[:, -1, :]
        else:
            raise ValueError(f"which must be 'bos' or 'last', got {which}")
        return self.label_head(h)


# Greedy generation for codes (deterministic) — eos stop supported
@torch.no_grad()
def greedy_generate_fixed(
    model: FlashDecoderSFT,
    user_static: torch.Tensor,
    short_term: torch.Tensor,
    code_len: int,
    bos_id: int,
    eos_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    B = user_static.size(0)
    gen = torch.full((B, 1), int(bos_id), dtype=torch.long, device=device)

    total_new = int(code_len) + 1  # code_len tokens + EOS
    finished = torch.zeros((B, 1), dtype=torch.bool, device=device)

    for _ in range(total_new):
        out = model.forward_with_hidden(gen, user_static, short_term, None)
        logits_last = out["logits"][:, -1, :]  # (B,V)
        next_tok = torch.argmax(logits_last, dim=-1, keepdim=True)

        if eos_id is not None:
            eos_fill = torch.full_like(next_tok, int(eos_id))
            next_tok = torch.where(finished, eos_fill, next_tok)
            finished = finished | (next_tok == int(eos_id))

        gen = torch.cat([gen, next_tok], dim=1)

        if eos_id is not None and finished.all():
            break

    return gen  # (B, <= 1 + total_new)


# Eval
@torch.no_grad()
def eval_split(
    model: FlashDecoderSFT,
    dl: DataLoader,
    device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    code_len: int,
    name: str,
    thr: float = 0.5,             
    ignore_value: float = -1.0,
) -> Dict[str, float]:
    model.eval()

    total_lm = 0.0
    n = 0

    y_true_all = []
    ylog_teacher_all = []
    ylog_bos_all = []
    ylog_freerun_all = []

    corr = np.zeros((code_len,), dtype=np.int64)
    tot_tok = 0
    exact_match = 0
    total_seq = 0

    for batch in tqdm(dl, desc=name, leave=False):
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        user_static = batch["user_static"].to(device, non_blocking=True)
        short_term = batch["short_term"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        out_t = model.forward_with_hidden(target_ids, user_static, short_term, None)
        logits_t = out_t["logits"]
        hidden_t = out_t["hidden"]

        lm_loss = compute_sft_loss(logits_t, target_ids, pad_id=pad_id, bos_id=bos_id)

        bs = target_ids.size(0)
        total_lm += float(lm_loss.item()) * bs
        n += bs

        ylog_teacher = model.logits_to_label(hidden_t, which="last")
        ylog_bos = model.logits_to_label(hidden_t, which="bos")

        # free-run generate (with eos stop)
        gen_ids = greedy_generate_fixed(
            model=model,
            user_static=user_static,
            short_term=short_term,
            code_len=code_len,
            bos_id=bos_id,
            eos_id=eos_id,
            device=device,
        )

        # ensure length for metric slicing
        if gen_ids.size(1) < (1 + code_len):
            pad_more = (1 + code_len) - gen_ids.size(1)
            gen_ids = torch.cat([gen_ids, torch.full((bs, pad_more), pad_id, device=device, dtype=torch.long)], dim=1)

        true_codes = target_ids[:, 1:1 + code_len]
        pred_codes = gen_ids[:, 1:1 + code_len]

        eq = (pred_codes == true_codes).detach().cpu().numpy().astype(np.int32)
        corr += eq.sum(axis=0).astype(np.int64)
        tot_tok += eq.shape[0]
        exact_match += int((eq.sum(axis=1) == code_len).sum())
        total_seq += eq.shape[0]

        out_g = model.forward_with_hidden(gen_ids, user_static, short_term, None)
        hidden_g = out_g["hidden"]
        ylog_free = model.logits_to_label(hidden_g, which="last")

        if ylog_teacher is not None and ylog_bos is not None and ylog_free is not None:
            y_true_all.append(y.detach().cpu().numpy())
            ylog_teacher_all.append(ylog_teacher.detach().cpu().numpy())
            ylog_bos_all.append(ylog_bos.detach().cpu().numpy())
            ylog_freerun_all.append(ylog_free.detach().cpu().numpy())

    out: Dict[str, float] = {}
    out["lm_loss"] = total_lm / max(1, n)

    if tot_tok > 0:
        for i in range(code_len):
            out[f"acc_c{i+1}"] = float(corr[i] / tot_tok)
        out["token_acc_mean"] = float(np.mean([out[f"acc_c{i+1}"] for i in range(code_len)]))
    else:
        for i in range(code_len):
            out[f"acc_c{i+1}"] = float("nan")
        out["token_acc_mean"] = float("nan")

    out["exact_match"] = float(exact_match / max(1, total_seq))

    if len(y_true_all) > 0:
        y_true = np.concatenate(y_true_all, axis=0).astype(np.float32)
        y_teacher = np.concatenate(ylog_teacher_all, axis=0).astype(np.float32)
        y_bos = np.concatenate(ylog_bos_all, axis=0).astype(np.float32)
        y_free = np.concatenate(ylog_freerun_all, axis=0).astype(np.float32)

        m_teacher = compute_auc_f1(y_true, y_teacher, thr=thr, ignore_value=ignore_value)
        m_bos     = compute_auc_f1(y_true, y_bos,     thr=thr, ignore_value=ignore_value)
        m_free    = compute_auc_f1(y_true, y_free,    thr=thr, ignore_value=ignore_value)


        for k, v in m_bos.items():
            out[f"bos_{k}"] = float(v)
        for k, v in m_free.items():
            out[f"free_{k}"] = float(v)
        for k, v in m_teacher.items():
            out[f"teacher_{k}"] = float(v)
    else:
        for pref in ["bos", "free", "teacher"]:
            out[f"{pref}_macro_auc"] = float("nan")
            out[f"{pref}_micro_auc"] = float("nan")
            out[f"{pref}_macro_f1"] = float("nan")
            out[f"{pref}_micro_f1"] = float("nan")

    return out


def train_one_epoch(
    model: FlashDecoderSFT,
    dl_tr: DataLoader,
    opt: torch.optim.Optimizer,
    device,
    pad_id: int,
    bos_id: int,
    alpha_cls: float = 0.0,
    use_amp: bool = False,
    grad_clip: float = 1.0,
    writer: Optional[SummaryWriter] = None,
    log_every: int = 10,
    global_step: int = 0,
) -> Tuple[Dict[str, float], int]:
    model.train()

    total = 0.0
    total_lm = 0.0
    total_cls = 0.0
    n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    pbar = tqdm(dl_tr, desc="train", leave=False)
    for batch in pbar:
        global_step += 1
        target_ids = batch["target_ids"].to(device, non_blocking=True)
        user_static = batch["user_static"].to(device, non_blocking=True)
        short_term = batch["short_term"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model.forward_with_hidden(target_ids, user_static, short_term, None)
            logits = out["logits"]
            hidden = out["hidden"]

            if global_step <= 2:
                print("\n[DEBUG] target_ids[0]:", target_ids[0].tolist())
                print("[DEBUG] unique target_ids:", torch.unique(target_ids))
                print("[DEBUG] pad_id =", pad_id, "bos_id =", bos_id)

            if torch.isnan(logits).any():
                print("❌ [FATAL] logits has NaN")
                print("logits range:", logits.min().item(), logits.max().item())
                raise RuntimeError("NaN logits detected")

            if (target_ids < 0).any():
                raise RuntimeError("Invalid target_ids (<0)")

            if (target_ids >= logits.size(-1)).any():
                print("❌ [FATAL] target_ids >= vocab_size")
                print("max target:", target_ids.max().item(), "vocab:", logits.size(-1))
                raise RuntimeError("Invalid target_ids (>= vocab_size)")

            with torch.no_grad():
                loss_mask = (target_ids != pad_id)
                if bos_id is not None:
                    loss_mask = loss_mask & (target_ids != bos_id)
                if loss_mask.sum() == 0:
                    raise RuntimeError("Empty loss mask → lm_loss will be NaN")

            loss_lm = compute_sft_loss(logits, target_ids, pad_id=pad_id, bos_id=bos_id)

            # loss_cls = torch.tensor(0.0, device=device)
            # if (alpha_cls is not None) and (alpha_cls > 0) and model.use_label_head:
            #     y_logit = model.logits_to_label(hidden, which="last")
            #     loss_cls = F.binary_cross_entropy_with_logits(y_logit, y)
            loss_cls = torch.tensor(0.0, device=device)
            if (alpha_cls is not None) and (alpha_cls > 0) and model.use_label_head:
                y_logit = model.logits_to_label(hidden, which="last")  # (B,C)

                # y in {-1,0,1} -> mask & convert to {0,1}
                mask = (y != -1).float()
                y01 = (y > 0).float()

                bce = F.binary_cross_entropy_with_logits(y_logit, y01, reduction="none")  # (B,C)
                loss_cls = (bce * mask).sum() / mask.sum().clamp_min(1.0)


            loss = loss_lm + float(alpha_cls) * loss_cls

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(opt)
        scaler.update()

        bs = target_ids.size(0)
        total += float(loss.item()) * bs
        total_lm += float(loss_lm.item()) * bs
        total_cls += float(loss_cls.item()) * bs
        n += bs

        pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lm=f"{float(loss_lm.item()):.4f}", cls=f"{float(loss_cls.item()):.4f}")

        if writer is not None and (global_step % max(1, log_every) == 0):
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            writer.add_scalar("train/lm_loss_step", float(loss_lm.item()), global_step)
            writer.add_scalar("train/cls_loss_step", float(loss_cls.item()), global_step)
            writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

    stats = {
        "loss": total / max(1, n),
        "lm_loss": total_lm / max(1, n),
        "cls_loss": total_cls / max(1, n),
    }
    return stats, global_step


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--tab_root", type=str, default=r"E:\NUS\data\perdata\train_text_all_samples\tab_embed_v1_flow")
    ap.add_argument("--ts_root", type=str, default=r"E:\NUS\data\perdata\train_text_all_samples\ts_embed_v1_cmgrw")
    ap.add_argument("--code_root", type=str, default=r"E:\NUS\data\perdata\train_text_all_samples\codebook\rqvae_cxr_chex0_v1\codes_bucketed")
    ap.add_argument("--label_index", type=str, default=r"E:\NUS\data\perdata\train_text_all_samples\meta\sample_index.json")

    ap.add_argument("--num_buckets", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch_size", default=256, type=int)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--wd", default=0.01, type=float)
    ap.add_argument("--num_workers", default=2, type=int)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_clip", default=1.0, type=float)

    ap.add_argument("--ctx_dim", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads_q", type=int, default=4)
    ap.add_argument("--gkv", type=int, default=1)
    ap.add_argument("--d_ff", type=int, default=512)

    ap.add_argument("--pad_id", type=int, default=0)
    ap.add_argument("--bos_id", type=int, default=1)
    ap.add_argument("--eos_id", type=int, default=2)

    # NEW: avoid collision with special tokens
    ap.add_argument("--code_offset", type=int, default=3, help="add this offset to code tokens before training")

    ap.add_argument("--alpha_cls", type=float, default=0.0, help="cls loss weight; 0 means disable")
    ap.add_argument("--no_label_head", action="store_true")

    ap.add_argument("--out_ckpt", type=str, required=True)
    ap.add_argument("--log_dir", default="", type=str)
    ap.add_argument("--exp_name", default="", type=str)
    ap.add_argument("--run_id", default="", type=str)
    ap.add_argument("--log_every", default=10, type=int)

    ap.add_argument("--thr", type=float, default=0.5, help="threshold for F1 after sigmoid")
    ap.add_argument("--label_ignore", type=float, default=-1.0, help="ignore label value (e.g. -1)")


    args = ap.parse_args()

    roc_auc_score, _ = try_import_sklearn_metrics()
    if roc_auc_score is None:
        print("[Warn] sklearn not found. Install: pip install scikit-learn (AUC/F1 -> NaN)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.fp16 and (device == "cuda")
    print(f"[Info] device={device}, amp={use_amp}")

    set_seed(args.seed)

    sid2y = load_sample_index_labels(args.label_index)
    print(f"[Info] loaded labels: {len(sid2y)} from {args.label_index}")
    label_dim = len(next(iter(sid2y.values())))
    print(f"[Info] label_dim={label_dim}")

    train_b, test_b, eval_b = make_bucket_split(args.num_buckets, seed=args.seed)
    print(f"[Info] bucket split: train={len(train_b)} test={len(test_b)} eval={len(eval_b)} (seed={args.seed})")

    ckpt_dir = os.path.dirname(args.out_ckpt) if os.path.dirname(args.out_ckpt) else "."
    os.makedirs(ckpt_dir, exist_ok=True)

    exp = args.exp_name.strip()
    if not exp:
        exp = (
            f"flashdecoder_sft_ctx{args.ctx_dim}_dm{args.d_model}_L{args.n_layers}_H{args.n_heads_q}_g{args.gkv}_"
            f"bs{args.batch_size}_lr{args.lr}_wd{args.wd}_fp16{int(args.fp16)}_aCls{args.alpha_cls}_off{args.code_offset}"
        )
    run_id = args.run_id.strip() if args.run_id.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.log_dir.strip():
        log_dir = args.log_dir.strip()
    else:
        log_dir = os.path.join(ckpt_dir, "tb", exp, run_id)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[TB ] log_dir: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("hparams", f"exp={exp}, run_id={run_id}\nargs={vars(args)}")

    split_path = os.path.join(ckpt_dir, f"bucket_split_70_20_10_seed{args.seed}.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "train": train_b, "test": test_b, "eval": eval_b}, f, indent=2)
    print("[Info] split saved:", split_path)

    ds_tr = BucketMixDataset(
        train_b, args.tab_root, args.ts_root, args.code_root, sid2y,
        bos_id=args.bos_id, eos_id=args.eos_id, pad_id=args.pad_id,
        code_offset=args.code_offset,
        verbose=True
    )
    ds_te = BucketMixDataset(
        test_b, args.tab_root, args.ts_root, args.code_root, sid2y,
        bos_id=args.bos_id, eos_id=args.eos_id, pad_id=args.pad_id,
        code_offset=args.code_offset,
        verbose=False
    )
    ds_ev = BucketMixDataset(
        eval_b, args.tab_root, args.ts_root, args.code_root, sid2y,
        bos_id=args.bos_id, eos_id=args.eos_id, pad_id=args.pad_id,
        code_offset=args.code_offset,
        verbose=False
    )

    print(f"[Info] sizes (after intersect): train={len(ds_tr)} test={len(ds_te)} eval={len(ds_ev)}")
    assert ds_tr.tab_dim is not None and len(ds_tr) > 0, "train split has 0 usable samples (sample_id intersection empty)."

    code_len = int(ds_tr.code_len)
    print(f"[Info] dims: tab_dim={ds_tr.tab_dim}, ts_dim={ds_tr.ts_dim}, code_len={code_len}, label_dim={ds_tr.label_dim}")

    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        collate_fn=collate_fn
    )
    dl_te = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )
    dl_ev = DataLoader(
        ds_ev, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn
    )

    # infer vocab_size + scan code min/max
    max_code = -1
    min_code = 10**18
    for bid in tqdm(range(args.num_buckets), desc="Scan vocab (min/max code)", leave=False):
        p = os.path.join(args.code_root, f"bucket_{bid:03d}_codes.npy")
        if not os.path.exists(p):
            continue
        c = np.load(p)
        if c.size > 0:
            max_code = max(max_code, int(c.max()))
            min_code = min(min_code, int(c.min()))

    if max_code < 0:
        raise RuntimeError("No codes found when scanning code_root. Check path.")

    vocab_size = (max_code + args.code_offset) + 3  # +3 for [pad,bos,eos] space (safe upper bound)
    print(f"[Info] raw code range: min={min_code}, max={max_code}")
    print(f"[Info] code_offset={args.code_offset} => shifted code range: [{min_code+args.code_offset}, {max_code+args.code_offset}]")
    print(f"[Info] inferred vocab_size={vocab_size}")

    model = FlashDecoderSFT(
        vocab_size=vocab_size,
        tab_dim=ds_tr.tab_dim,
        ts_dim=ds_tr.ts_dim,
        label_dim=label_dim,
        ctx_dim=args.ctx_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads_q=args.n_heads_q,
        gkv=args.gkv,
        d_ff=args.d_ff,
        pad_id=args.pad_id,
        bos_id=args.bos_id,
        use_label_head=(not args.no_label_head),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_key = -1e9
    best_ep = 0
    best_test_snapshot: Dict[str, float] = {}
    global_step = 0

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    for ep in tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True):
        tr_stats, global_step = train_one_epoch(
            model, dl_tr, opt, device,
            pad_id=args.pad_id, bos_id=args.bos_id,
            alpha_cls=args.alpha_cls,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            writer=writer,
            log_every=args.log_every,
            global_step=global_step,
        )

        writer.add_scalar("loss/train_epoch", tr_stats["loss"], global_step)
        writer.add_scalar("loss/train_lm_epoch", tr_stats["lm_loss"], global_step)
        writer.add_scalar("loss/train_cls_epoch", tr_stats["cls_loss"], global_step)

        te = eval_split(
            model, dl_te, device,
            pad_id=args.pad_id, bos_id=args.bos_id, eos_id=args.eos_id,
            code_len=code_len,
            name="test",
            thr=args.thr,
            ignore_value=args.label_ignore,
        )


        writer.add_scalar("loss/test_lm", te["lm_loss"], global_step)
        writer.add_scalar("gen/exact_match_test", te["exact_match"], global_step)
        writer.add_scalar("gen/token_acc_mean_test", te["token_acc_mean"], global_step)
        for i in range(code_len):
            writer.add_scalar(f"gen/acc_c{i+1}_test", te.get(f"acc_c{i+1}", float("nan")), global_step)

        for pref in ["bos", "free", "teacher"]:
            writer.add_scalar(f"auc/{pref}_macro_test", te.get(f"{pref}_macro_auc", float("nan")), global_step)
            writer.add_scalar(f"auc/{pref}_micro_test", te.get(f"{pref}_micro_auc", float("nan")), global_step)
            writer.add_scalar(f"f1/{pref}_macro_test", te.get(f"{pref}_macro_f1", float("nan")), global_step)
            writer.add_scalar(f"f1/{pref}_micro_test", te.get(f"{pref}_micro_f1", float("nan")), global_step)

        key_now = float(te.get("free_macro_auc", float("nan")))
        improved = (not math.isnan(key_now)) and (key_now > best_key)

        acc_vals = []
        for i in range(code_len):
            k = f"acc_c{i+1}"
            acc_vals.append(f"{te.get(k, float('nan')):.3f}")
        acc_str = ", ".join(acc_vals)

        print(
            f"[Epoch {ep:02d}] "
            f"train_loss={tr_stats['loss']:.4f} (lm={tr_stats['lm_loss']:.4f}, cls={tr_stats['cls_loss']:.4f}) | "
            f"test_lm={te['lm_loss']:.4f} "
            f"free_macroAUC={te.get('free_macro_auc', float('nan')):.4f} free_microAUC={te.get('free_micro_auc', float('nan')):.4f} "
            f"free_macroF1={te.get('free_macro_f1', float('nan')):.4f} free_microF1={te.get('free_micro_f1', float('nan')):.4f} | "
            f"bos_macroAUC={te.get('bos_macro_auc', float('nan')):.4f} teacher_macroAUC={te.get('teacher_macro_auc', float('nan')):.4f} | "
            f"acc=[{acc_str}] exact={te['exact_match']:.3f} "
            f"| select_by=free_macro_auc key_now={key_now:.4f} best_key={best_key:.4f}"
        )

        if improved:
            best_key = float(key_now)
            best_ep = int(ep)
            best_test_snapshot = dict(te)

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_ep": int(best_ep),
                    "select_by": "free_macro_auc",
                    "best_key": float(best_key),
                    "best_test": best_test_snapshot,
                    "log_dir": log_dir,
                    "exp": exp,
                    "run_id": run_id,
                    "split_path": split_path,
                    "args": vars(args),
                },
                args.out_ckpt,
            )
            print(f"[Save] best ckpt -> {args.out_ckpt} (ep={best_ep:02d}, best_free_macroAUC={best_key:.4f})")

        writer.flush()

    writer.close()

    if not os.path.exists(args.out_ckpt):
        print("[Final] no ckpt saved (unexpected). Done.")
        return

    print("\n[Final] load BEST ckpt and evaluate on EVAL split ONCE ...")
    ckpt = torch.load(args.out_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    ev = eval_split(
        model, dl_ev, device,
        pad_id=args.pad_id, bos_id=args.bos_id, eos_id=args.eos_id,
        code_len=code_len,
        name="eval",
        thr=args.thr,
        ignore_value=args.label_ignore,
    )


    bt = ckpt.get("best_test", {})

    def _fmt_acc(snapshot: Dict[str, float]) -> str:
        vals = []
        for i in range(code_len):
            vals.append(f"{snapshot.get(f'acc_c{i+1}', float('nan')):.3f}")
        return ", ".join(vals)

    bt_acc = _fmt_acc(bt)
    ev_acc = _fmt_acc(ev)

    print(
        f"[BEST by TEST] ep={int(ckpt.get('best_ep', -1))} select_by={ckpt.get('select_by','free_macro_auc')} best_key={float(ckpt.get('best_key', float('nan'))):.4f}\n"
        f"  TEST: lm={float(bt.get('lm_loss', float('nan'))):.4f} "
        f"free_macroAUC={float(bt.get('free_macro_auc', float('nan'))):.4f} bos_macroAUC={float(bt.get('bos_macro_auc', float('nan'))):.4f} teacher_macroAUC={float(bt.get('teacher_macro_auc', float('nan'))):.4f} "
        f"free_macroF1={float(bt.get('free_macro_f1', float('nan'))):.4f} free_microF1={float(bt.get('free_micro_f1', float('nan'))):.4f} "
        f"acc=[{bt_acc}] exact={float(bt.get('exact_match', float('nan'))):.4f}\n"
        f"  EVAL: lm={float(ev.get('lm_loss', float('nan'))):.4f} "
        f"free_macroAUC={float(ev.get('free_macro_auc', float('nan'))):.4f} bos_macroAUC={float(ev.get('bos_macro_auc', float('nan'))):.4f} teacher_macroAUC={float(ev.get('teacher_macro_auc', float('nan'))):.4f} "
        f"free_macroF1={float(ev.get('free_macro_f1', float('nan'))):.4f} free_microF1={float(ev.get('free_micro_f1', float('nan'))):.4f} "
        f"acc=[{ev_acc}] exact={float(ev.get('exact_match', float('nan'))):.4f}"
    )

    print("[Done]")
    print(f"[TB ] for THIS run: tensorboard --logdir \"{log_dir}\"")
    print(f"[TB ] compare MANY runs under ckpt_dir: tensorboard --logdir \"{os.path.join(ckpt_dir, 'tb')}\"")


if __name__ == "__main__":
    main()



# python trainer_fd.py --fp16 --epochs 5 --batch_size 128 --lr 2e-4 --wd 0.01 --log_every 10 --alpha_cls 0.0 --num_workers 0 --code_offset 3 --out_ckpt "E:\NUS\data\perdata\train_text_all_samples\ckpts\flashdecoder_sft_v1\best.pt"



# python trainer_fd.py --epochs 5 --batch_size 128 --lr 2e-4 --wd 0.01 --log_every 10 --alpha_cls 0.1 --num_workers 0 --code_offset 3 --thr 0.2 --label_ignore -1 --out_ckpt "E:\NUS\data\perdata\train_text_all_samples\ckpts\flashdecoder_sft_v1\best.pt"
