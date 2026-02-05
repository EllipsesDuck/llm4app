import os
import glob
import json
import random
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from layers import MLPLayers
from rq import ResidualVectorQuantizer


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
            # jsonl
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
    # ndarray -> item
    if isinstance(dtype_raw, np.ndarray):
        if dtype_raw.size == 0:
            raise ValueError("Empty dtype array in meta")
        dtype_raw = dtype_raw.item() if dtype_raw.size == 1 else dtype_raw.reshape(-1)[0].item()

    # bytes -> str
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
        # bucket_007_xxx_emb.npy or bucket_007_emb.npy
        # bucket id:
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

        # open mmaps + build global index
        for bi, (emb_path, sid_path, bucket_id) in enumerate(bucket_pairs):
            emb, meta_dict = open_emb_memmap(emb_path)
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

        # infer in_dim
        self.in_dim = int(self.emb_mmaps[0].shape[1])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        bi, ri = self.index[i]
        x = self.emb_mmaps[bi][ri]  # (D,)
        sid = str(self.sids_per_bucket[bi][ri])
        lab = self.sid2lab.get(sid, None)

        # to torch float32 (训练更稳)
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return x, sid, lab


import torch.nn as nn

class RQVAE(nn.Module):
    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
    ):
        super().__init__()

        assert num_emb_list is not None, "num_emb_list must be provided"
        assert sk_epsilons is not None, "sk_epsilons must be provided"
        assert len(num_emb_list) == len(sk_epsilons), "num_emb_list and sk_epsilons length must same"
        assert layers is not None and len(layers) > 0, "layers must be provided"

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters

        # Encoder
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims, dropout=self.dropout_prob, use_bn=self.bn)

        # Residual VQ
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        # Decoder
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims, dropout=self.dropout_prob, use_bn=self.bn)

    def forward(self, x, use_sk=True):
        x_e = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x_e, use_sk=use_sk)
        out = self.decoder(x_q)
        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, use_sk=use_sk)
        return indices

    def compute_loss(self, out, quant_loss, xs):
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("incompatible loss type")
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon


def train_one_epoch(model, dl, opt, device, use_amp: bool, use_sk: bool, grad_clip: float):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total, recon_total, q_total, n = 0.0, 0.0, 0.0, 0

    for x, _, _ in dl:
        x = x.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out, rq_loss, _ = model(x, use_sk=use_sk)
                loss_total, loss_recon = model.compute_loss(out, rq_loss, x)
        else:
            out, rq_loss, _ = model(x, use_sk=use_sk)
            loss_total, loss_recon = model.compute_loss(out, rq_loss, x)

        if use_amp:
            scaler.scale(loss_total).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss_total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        bsz = x.size(0)
        total += float(loss_total.item()) * bsz
        recon_total += float(loss_recon.item()) * bsz
        q_total += float(rq_loss.item()) * bsz
        n += bsz

    return total / max(n, 1), recon_total / max(n, 1), q_total / max(n, 1)


@torch.no_grad()
def export_codes_by_bucket(model, bucket_pairs, out_dir_codes, device, batch_size: int, tag: str, use_sk: bool):
    os.makedirs(out_dir_codes, exist_ok=True)
    model.eval()

    tag = tag.strip()
    tag_suffix = f"_{tag}" if tag else ""

    for emb_path, sid_path, b in bucket_pairs:
        emb, meta_dict = open_emb_memmap(emb_path)
        n, d = emb.shape
        codes_list = []

        for s in range(0, n, batch_size):
            x = np.asarray(emb[s:s+batch_size], dtype=np.float32)
            x = torch.from_numpy(x).to(device)
            idx = model.get_indices(x, use_sk=use_sk)  # (B, n_layers)
            codes_list.append(idx.detach().cpu().numpy().astype(np.int32))

        codes = np.concatenate(codes_list, axis=0)
        assert codes.shape[0] == n

        out_codes = os.path.join(out_dir_codes, f"bucket_{b:03d}{tag_suffix}_codes.npy")
        np.save(out_codes, codes)
        print(f"[Export] bucket {b:03d}: codes {codes.shape} -> {out_codes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, type=str, help="dir containing bucket_XXX*_emb.npy and *_sid.npy")
    ap.add_argument("--meta_index", required=True, type=str, help="sample_index.json or jsonl with sample_id & label_multi")
    ap.add_argument("--out_dir", required=True, type=str, help="output dir for rqvae ckpt + exported codes")

    ap.add_argument("--tag", default="", type=str, help="if your emb files are like bucket_007_TAG_emb.npy, set tag=TAG")

    ap.add_argument("--epochs", default=10, type=int)
    ap.add_argument("--batch_size", default=1024, type=int)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--grad_clip", default=1.0, type=float)

    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--use_sk", action="store_true", help="use Sinkhorn during training forward")
    ap.add_argument("--export_use_sk", action="store_true", help="use Sinkhorn when exporting indices")

    ap.add_argument("--require_label", action="store_true", help="drop samples that have no label in sample_index")

    # RQVAE knobs (defaults good for DenseNet 1024d)
    ap.add_argument("--e_dim", default=128, type=int)
    ap.add_argument("--layers", default="512,256", type=str, help="encoder hidden dims, comma sep")
    ap.add_argument("--num_emb_list", default="512,512,512", type=str, help="3-layer RVQ codebook sizes")
    ap.add_argument("--sk_epsilons", default="0.0,0.003,0.01", type=str)
    ap.add_argument("--sk_iters", default=50, type=int)

    ap.add_argument("--beta", default=0.25, type=float)
    ap.add_argument("--quant_loss_weight", default=1.0, type=float)
    ap.add_argument("--dropout", default=0.0, type=float)
    ap.add_argument("--bn", action="store_true")

    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load label map
    sid2lab = load_sample_index(args.meta_index)
    print(f"[Info] loaded labels for {len(sid2lab)} sample_ids from {args.meta_index}")

    # load bucket pairs
    bucket_pairs = list_bucket_files(args.emb_dir, tag=args.tag)
    print(f"[Info] found {len(bucket_pairs)} bucket pairs under {args.emb_dir}")

    # build dataset
    ds = BucketedEmbDataset(bucket_pairs, sid2lab=sid2lab, require_label=args.require_label)
    print(f"[Info] dataset size: {len(ds)} samples, in_dim={ds.in_dim}")

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,     
        pin_memory=True,
        drop_last=False,
    )

    # parse rqvae config
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    num_emb_list = [int(x) for x in args.num_emb_list.split(",") if x.strip()]
    sk_epsilons = [float(x) for x in args.sk_epsilons.split(",") if x.strip()]
    assert len(num_emb_list) == 3, "You asked for 3-layer RVQ; set --num_emb_list with 3 ints"
    assert len(sk_epsilons) == 3, "Set --sk_epsilons with 3 floats"

    # build model
    model = RQVAE(
        in_dim=ds.in_dim,
        num_emb_list=num_emb_list,
        e_dim=args.e_dim,
        layers=layers,
        dropout_prob=args.dropout,
        bn=args.bn,
        loss_type="mse",
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=False,
        kmeans_iters=50,
        sk_epsilons=sk_epsilons,
        sk_iters=args.sk_iters,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "rqvae_ckpt.pt")

    for ep in range(1, args.epochs + 1):
        loss_total, loss_recon, loss_q = train_one_epoch(
            model, dl, opt, device,
            use_amp=(args.use_amp and device == "cuda"),
            use_sk=args.use_sk,
            grad_clip=args.grad_clip
        )
        print(f"[Epoch {ep:03d}] total={loss_total:.6f} recon={loss_recon:.6f} q={loss_q:.6f}")

        # save checkpoint each epoch
        torch.save(
            {
                "epoch": ep,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "in_dim": ds.in_dim,
                "e_dim": args.e_dim,
                "layers": layers,
                "num_emb_list": num_emb_list,
                "sk_epsilons": sk_epsilons,
                "sk_iters": args.sk_iters,
            },
            ckpt_path
        )

    print(f"[Info] saved ckpt: {ckpt_path}")

    # export codes
    out_codes_dir = os.path.join(args.out_dir, "codes_bucketed")
    export_codes_by_bucket(
        model,
        bucket_pairs=bucket_pairs,
        out_dir_codes=out_codes_dir,
        device=device,
        batch_size=min(args.batch_size, 4096),
        tag=args.tag,
        use_sk=args.export_use_sk
    )
    print("[Done] exported codes to:", out_codes_dir)


if __name__ == "__main__":
    main()

# python img_emb_rqvae.py --emb_dir "E:/NUS/data/perdata/train_text_all_samples/cxr_emb_chex0_bucketed" --meta_index "E:/NUS/data/perdata/train_text_all_samples/meta/sample_index.json" --out_dir "E:/NUS/data/perdata/train_text_all_samples/codebook/rqvae_cxr_chex0" --epochs 10 --batch_size 1024 --lr 2e-4 --weight_decay 1e-4 --use_amp --bn --e_dim 128 --layers "512,256" --num_emb_list "1024,1024,1024" --sk_epsilons "0.0,0.003,0.01" --sk_iters 50
