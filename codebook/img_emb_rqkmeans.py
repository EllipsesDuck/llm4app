import os
import glob
import json
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_sample_index(path: str) -> Dict[str, List[int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sid2lab = {}
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)

        if head == "[":
            arr = json.load(f)
            for r in arr:
                sid = str(r.get("sample_id", "")).strip()
                lab = r.get("label_multi", None)
                if sid:
                    sid2lab[sid] = lab
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = str(r.get("sample_id", "")).strip()
                lab = r.get("label_multi", None)
                if sid:
                    sid2lab[sid] = lab
    return sid2lab


def parse_meta_dtype(dtype_raw):
    if isinstance(dtype_raw, np.ndarray):
        if dtype_raw.size == 0:
            raise ValueError("Empty dtype array in meta")
        dtype_raw = dtype_raw.item() if dtype_raw.size == 1 else dtype_raw.reshape(-1)[0].item()

    if isinstance(dtype_raw, (bytes, np.bytes_)):
        dtype_raw = dtype_raw.decode("utf-8", errors="ignore")

    try:
        return np.dtype(dtype_raw)
    except TypeError:
        s = str(dtype_raw)
        if "float16" in s:
            return np.dtype(np.float16)
        if "float32" in s:
            return np.dtype(np.float32)
        raise


def open_emb_memmap(emb_path: str):
    meta_path = emb_path.replace("_emb.npy", "_meta.npz")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    meta = np.load(meta_path)
    count = int(np.asarray(meta["count"]).item())
    emb_dim = int(np.asarray(meta["emb_dim"]).item())
    dtype = parse_meta_dtype(meta["dtype"])

    emb = np.memmap(
        emb_path,
        dtype=dtype,
        mode="r",
        shape=(count, emb_dim)
    )

    meta_dict = {"count": count, "emb_dim": emb_dim, "dtype": dtype}
    return emb, meta_dict


def list_bucket_files(emb_dir: str, tag: str = "") -> List[Tuple[str, str, int]]:
    tag = tag.strip()
    tag_suffix = f"_{tag}" if tag else ""

    emb_glob = os.path.join(emb_dir, f"bucket_*{tag_suffix}_emb.npy")
    emb_paths = sorted(glob.glob(emb_glob))
    if not emb_paths:
        raise ValueError(f"No embedding files found under: {emb_glob}")

    pairs = []
    for ep in emb_paths:
        base = os.path.basename(ep)
        try:
            b = int(base.split("_")[1])
        except Exception:
            continue

        sp = ep.replace("_emb.npy", "_sid.npy")
        if not os.path.exists(sp):
            raise ValueError(f"Missing sid file for {ep}: expected {sp}")
        pairs.append((ep, sp, b))

    pairs = sorted(pairs, key=lambda x: x[2])
    return pairs


class BucketedEmbDataset(Dataset):
    def __init__(self, bucket_pairs: List[Tuple[str, str, int]], sid2lab: Dict[str, List[int]], require_label: bool = False):
        self.bucket_pairs = bucket_pairs
        self.sid2lab = sid2lab
        self.require_label = require_label

        self.emb_mmaps = []
        self.sids_per_bucket = []
        self.bucket_id_per_bucket = []
        self.index = []  # global index -> (bucket_idx, row_idx)

        for bi, (emb_path, sid_path, bucket_id) in enumerate(bucket_pairs):
            emb, _ = open_emb_memmap(emb_path)
            sids = np.load(sid_path, allow_pickle=True)

            assert emb.shape[0] == len(sids), f"Mismatch rows: {emb_path} vs {sid_path}"
            self.emb_mmaps.append(emb)
            self.sids_per_bucket.append(sids)
            self.bucket_id_per_bucket.append(bucket_id)

            for ri in range(len(sids)):
                sid = str(sids[ri])
                if self.require_label and sid not in sid2lab:
                    continue
                self.index.append((bi, ri))

        if len(self.index) == 0:
            raise ValueError("No samples available after filtering. Check labels or paths.")

        self.in_dim = int(self.emb_mmaps[0].shape[1])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        bi, ri = self.index[i]
        x = self.emb_mmaps[bi][ri]
        sid = str(self.sids_per_bucket[bi][ri])
        lab = self.sid2lab.get(sid, None)
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return x, sid, lab


# RQ-KMeans (Residual KMeans Quantization)
def _cdist_squared(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # x: (B,D), c: (K,D) -> dist2: (B,K)
    # dist2 = ||x||^2 + ||c||^2 - 2 x c^T
    x2 = (x * x).sum(dim=1, keepdim=True)           # (B,1)
    c2 = (c * c).sum(dim=1, keepdim=True).T         # (1,K)
    xc = x @ c.T                                     # (B,K)
    return x2 + c2 - 2.0 * xc


@torch.no_grad()
def _init_centroids_from_loader(dl: DataLoader, k: int, device: str, max_batches: int = 10) -> torch.Tensor:
    buf = []
    for bi, (x, _, _) in enumerate(dl):
        buf.append(x)
        if bi + 1 >= max_batches:
            break
    x0 = torch.cat(buf, dim=0)
    if x0.size(0) < k:
        rep = (k + x0.size(0) - 1) // x0.size(0)
        x0 = x0.repeat(rep, 1)
    perm = torch.randperm(x0.size(0))[:k]
    c = x0[perm].to(device)
    return c.contiguous()


class RQKMeans(torch.nn.Module):
    def __init__(self, in_dim: int, num_emb_list: List[int], ema_decay: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.num_emb_list = list(num_emb_list)
        self.n_layers = len(num_emb_list)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)

        self.codebooks = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(k, in_dim), requires_grad=False)
            for k in self.num_emb_list
        ])

        self.register_buffer("_inited", torch.tensor(0, dtype=torch.int32))
        self.ema_counts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(k), requires_grad=False)
            for k in self.num_emb_list
        ])
        self.ema_sums = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(k, in_dim), requires_grad=False)
            for k in self.num_emb_list
        ])

    @torch.no_grad()
    def init_from_loader(self, dl: DataLoader, device: str, init_batches: int = 10):
        r_dl_cache = None  
        buf = []
        for bi, (x, _, _) in enumerate(dl):
            buf_confirm = x
            buf.append(buf_confirm)
            if bi + 1 >= init_batches:
                break
        x0 = torch.cat(buf, dim=0).to(device)
        if x0.size(0) < max(self.num_emb_list):
            rep = (max(self.num_emb_list) + x0.size(0) - 1) // x0.size(0)
            x0 = x0.repeat(rep, 1)

        r = x0
        for li, k in enumerate(self.num_emb_list):
            perm = torch.randperm(r.size(0), device=device)[:k]
            c = r[perm].contiguous()
            self.codebooks[li].data.copy_(c)
            # reset ema stats
            self.ema_counts[li].data.zero_()
            self.ema_sums[li].data.zero_()
            dist2 = _cdist_squared(r, self.codebooks[li].data)
            idx = dist2.argmin(dim=1)
            q = self.codebooks[li].data[idx]
            r = r - q

        self._inited.fill_(1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # returns indices: (B, n_layers)
        r = x
        idxs = []
        for li in range(self.n_layers):
            c = self.codebooks[li].data
            dist2 = _cdist_squared(r, c)
            idx = dist2.argmin(dim=1)  # (B,)
            q = c[idx]
            r = r - q
            idxs.append(idx)
        return torch.stack(idxs, dim=1)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        # returns recon: (B,D)
        r = x
        recon = torch.zeros_like(x)
        for li in range(self.n_layers):
            c = self.codebooks[li].data
            dist2 = _cdist_squared(r, c)
            idx = dist2.argmin(dim=1)
            q = c[idx]
            recon = recon + q
            r = r - q
        return recon

    @torch.no_grad()
    def update_layer_minibatch(self, r: torch.Tensor, layer_idx: int):
        c = self.codebooks[layer_idx].data
        dist2 = _cdist_squared(r, c)
        idx = dist2.argmin(dim=1)  # (B,)

        k = c.size(0)
        onehot = torch.zeros(r.size(0), k, device=r.device, dtype=r.dtype)
        onehot.scatter_(1, idx.view(-1, 1), 1.0)

        batch_counts = onehot.sum(dim=0)                       # (K,)
        batch_sums = onehot.T @ r                               # (K,D)

        # EMA
        decay = self.ema_decay
        self.ema_counts[layer_idx].data.mul_(decay).add_(batch_counts * (1.0 - decay))
        self.ema_sums[layer_idx].data.mul_(decay).add_(batch_sums * (1.0 - decay))

        denom = self.ema_counts[layer_idx].data.clamp_min(self.eps).unsqueeze(1)  # (K,1)
        new_c = self.ema_sums[layer_idx].data / denom
        self.codebooks[layer_idx].data.copy_(new_c)

        q = self.codebooks[layer_idx].data[idx]
        r_next = r - q
        return r_next, idx


@torch.no_grad()
def train_rqkmeans(
    model: RQKMeans,
    dl: DataLoader,
    device: str,
    epochs: int,
    log_every: int = 100,
):
    assert int(model._inited.item()) == 1, "Call model.init_from_loader(...) first!"

    model.eval() 
    step = 0
    for ep in range(1, epochs + 1):
        avg_res2 = 0.0
        n_seen = 0

        for bi, (x, _, _) in enumerate(dl):
            step += 1
            x = x.to(device, non_blocking=True)
            r = x

            for li in range(model.n_layers):
                r, _ = model.update_layer_minibatch(r, li)

            res2 = (r * r).sum(dim=1).mean().item()
            avg_res2 += res2 * x.size(0)
            n_seen += x.size(0)

            if log_every > 0 and (bi + 1) % log_every == 0:
                print(f"[Epoch {ep:03d}][iter {bi+1:05d}] mean_res2={res2:.6f}")

        avg_res2 = avg_res2 / max(n_seen, 1)
        print(f"[Epoch {ep:03d}] avg_mean_res2={avg_res2:.6f}")


@torch.no_grad()
def export_codes_by_bucket_rqkmeans(model, bucket_pairs, out_dir_codes, device, batch_size: int, tag: str):
    os.makedirs(out_dir_codes, exist_ok=True)
    model.eval()

    tag = tag.strip()
    tag_suffix = f"_{tag}" if tag else ""

    for emb_path, sid_path, b in bucket_pairs:
        emb, _ = open_emb_memmap(emb_path)
        n, d = emb.shape
        codes_list = []

        for s in range(0, n, batch_size):
            x = np.asarray(emb[s:s + batch_size], dtype=np.float32)
            x = torch.from_numpy(x).to(device)
            idx = model.encode(x)  # (B, n_layers)
            codes_list.append(idx.detach().cpu().numpy().astype(np.int32))

        codes = np.concatenate(codes_list, axis=0)
        assert codes.shape[0] == n

        out_codes = os.path.join(out_dir_codes, f"bucket_{b:03d}{tag_suffix}_codes.npy")
        np.save(out_codes, codes)
        print(f"[Export] bucket {b:03d}: codes {codes.shape} -> {out_codes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, type=str)
    ap.add_argument("--meta_index", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--tag", default="", type=str)

    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch_size", default=2048, type=int)
    ap.add_argument("--num_workers", default=0, type=int)

    # RQ-KMeans knobs
    ap.add_argument("--num_emb_list", default="512,512,512", type=str, help="per-layer K, comma sep")
    ap.add_argument("--ema_decay", default=0.9, type=float, help="EMA decay for online kmeans update, 0.9~0.99")
    ap.add_argument("--init_batches", default=10, type=int, help="how many batches used for init sampling")
    ap.add_argument("--log_every", default=100, type=int)

    ap.add_argument("--require_label", action="store_true")
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--export_batch_size", default=4096, type=int)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    sid2lab = load_sample_index(args.meta_index)
    print(f"[Info] loaded labels for {len(sid2lab)} sample_ids from {args.meta_index}")

    bucket_pairs = list_bucket_files(args.emb_dir, tag=args.tag)
    print(f"[Info] found {len(bucket_pairs)} bucket pairs under {args.emb_dir}")

    ds = BucketedEmbDataset(bucket_pairs, sid2lab=sid2lab, require_label=args.require_label)
    print(f"[Info] dataset size: {len(ds)} samples, in_dim={ds.in_dim}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_emb_list = [int(x) for x in args.num_emb_list.split(",") if x.strip()]
    assert len(num_emb_list) >= 1, "num_emb_list must have at least 1 layer"

    model = RQKMeans(
        in_dim=ds.in_dim,
        num_emb_list=num_emb_list,
        ema_decay=args.ema_decay,
    ).to(device)

    # init
    print("[Info] init codebooks from data ...")
    model.init_from_loader(dl, device=device, init_batches=args.init_batches)
    print("[Info] init done.")

    # train
    print("[Info] training RQ-KMeans ...")
    train_rqkmeans(
        model=model,
        dl=dl,
        device=device,
        epochs=args.epochs,
        log_every=args.log_every,
    )

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "rqkmeans_ckpt.pt")
    torch.save(
        {
            "in_dim": ds.in_dim,
            "num_emb_list": num_emb_list,
            "ema_decay": args.ema_decay,
            "codebooks": [cb.detach().cpu() for cb in model.codebooks],
        },
        ckpt_path
    )
    print(f"[Info] saved ckpt: {ckpt_path}")

    # export codes
    out_codes_dir = os.path.join(args.out_dir, "codes_bucketed")
    export_codes_by_bucket_rqkmeans(
        model=model,
        bucket_pairs=bucket_pairs,
        out_dir_codes=out_codes_dir,
        device=device,
        batch_size=args.export_batch_size,
        tag=args.tag,
    )
    print("[Done] exported codes to:", out_codes_dir)


if __name__ == "__main__":
    main()

#  python img_emb_rqkmeans.py --emb_dir "E:/NUS/data/perdata/train_text_all_samples/cxr_emb_chex0_bucketed" --meta_index "E:/NUS/data/perdata/train_text_all_samples/meta/sample_index.json" --out_dir "E:/NUS/data/perdata/train_text_all_samples/codebook/rqkmeans_cxr_chex0" --epochs 10 --batch_size 1024 --num_emb_list "1024,1024,1024" --ema_decay 0.9 --init_batches 10 --log_every 100 --export_batch_size 4096


