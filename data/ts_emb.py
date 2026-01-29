import os
import glob
import json
import random
import math
import gc
import argparse
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_bucket_pts(ts_tensor_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(ts_tensor_dir, "bucket_[0-9][0-9][0-9].pt")))
    return [p for p in paths if "_emb" not in os.path.basename(p) and "ckpt" not in os.path.basename(p)]


def get_bucket_id_from_path(p: str) -> int:
    return int(os.path.basename(p).split("_")[1].split(".")[0])


def masked_mean_pool(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(H.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (H * mask_f).sum(dim=1) / denom


class BucketPtDataset(Dataset):
    def __init__(self, bucket_pt_path: str):
        self.data = torch.load(bucket_pt_path, map_location="cpu")
        self.sample_id = self.data["sample_id"]
        self.item = self.data["item"]
        self.src = self.data["src"]
        self.value = self.data["value"]
        self.dt = self.data["dt"]
        self.mask = self.data["mask"]
        self.lab_flag = self.data.get("lab_flag", None)
        self.meta = self.data.get("meta", {})

    def __len__(self):
        return len(self.sample_id)

    def __getitem__(self, idx):
        out = {
            "sample_id": self.sample_id[idx],
            "item": self.item[idx],
            "src": self.src[idx],
            "value": self.value[idx],
            "dt": self.dt[idx],
            "mask": self.mask[idx],
        }
        if self.lab_flag is not None:
            out["lab_flag"] = self.lab_flag[idx]
        return out


def collate_fn_basic(batch):
    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "item": torch.stack([b["item"] for b in batch], dim=0),
        "src": torch.stack([b["src"] for b in batch], dim=0),
        "value": torch.stack([b["value"] for b in batch], dim=0),
        "dt": torch.stack([b["dt"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
    }
    if "lab_flag" in batch[0]:
        out["lab_flag"] = torch.stack([b["lab_flag"] for b in batch], dim=0)
    return out


def transform_value(v: torch.Tensor, mode: str, clip: float, eps: float) -> torch.Tensor:
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return v
    out = torch.sign(v) * torch.log1p(torch.abs(v) + eps)
    if clip is not None and clip > 0:
        out = out.clamp(min=-clip, max=clip)
    return out


def make_collate_fn_value_compress(mode: str, clip: float, eps: float):
    def _collate(batch):
        out = {
            "sample_id": [b["sample_id"] for b in batch],
            "item": torch.stack([b["item"] for b in batch], dim=0),
            "src": torch.stack([b["src"] for b in batch], dim=0),
            "value": torch.stack([b["value"] for b in batch], dim=0).float(),
            "dt": torch.stack([b["dt"] for b in batch], dim=0).float(),
            "mask": torch.stack([b["mask"] for b in batch], dim=0),
        }
        if "lab_flag" in batch[0]:
            out["lab_flag"] = torch.stack([b["lab_flag"] for b in batch], dim=0)
        out["value"] = transform_value(out["value"], mode=mode, clip=clip, eps=eps)
        return out

    return _collate


class EventTransformerEncoder(nn.Module):
    def __init__(
        self,
        hash_vocab: int,
        n_src: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_lab_flag: bool = True,
    ):
        super().__init__()
        self.use_lab_flag = use_lab_flag
        self.item_emb = nn.Embedding(hash_vocab + 1, d_model, padding_idx=0)
        self.src_emb = nn.Embedding(n_src + 1, d_model, padding_idx=0)
        self.val_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dt_proj = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        if use_lab_flag:
            self.flag_emb = nn.Embedding(4, d_model, padding_idx=0)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, item, src, value, dt, mask, lab_flag=None):
        x = self.item_emb(item) + self.src_emb(src)
        x = x + self.val_proj(value.unsqueeze(-1).to(x.dtype))
        x = x + self.dt_proj(dt.unsqueeze(-1).to(x.dtype))
        if self.use_lab_flag and (lab_flag is not None):
            lf = lab_flag.to(torch.int64)
            mapped = torch.zeros_like(lf)
            mapped[lf < 0] = 1
            mapped[lf > 0] = 3
            x = x + self.flag_emb(mapped)
        H = self.encoder(x, src_key_padding_mask=(~mask))
        emb = masked_mean_pool(H, mask)
        emb = self.out_norm(emb)
        return emb


def build_ts_encoder_from_dir(
    ts_tensor_dir: str,
    d_model: int = 128,
    nhead: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    n_src: int = 8,
    hash_v_fallback: int = 100_000,
    device: Optional[str] = None,
) -> Tuple[EventTransformerEncoder, Dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bucket_paths = list_bucket_pts(ts_tensor_dir)
    if len(bucket_paths) == 0:
        raise FileNotFoundError(f"No bucket_XXX.pt found in: {ts_tensor_dir}")
    tmp = torch.load(bucket_paths[0], map_location="cpu")
    meta0 = tmp.get("meta", {})
    hash_v = int(meta0.get("HASH_V", hash_v_fallback))
    use_lab_flag = ("lab_flag" in tmp)
    enc = EventTransformerEncoder(
        hash_vocab=hash_v,
        n_src=n_src,
        d_model=d_model,
        nhead=nhead,
        num_layers=n_layers,
        dropout=dropout,
        use_lab_flag=use_lab_flag,
    ).to(device)
    meta = {
        "device": device,
        "hash_v": hash_v,
        "use_lab_flag": use_lab_flag,
        "d_model": d_model,
        "nhead": nhead,
        "n_layers": n_layers,
        "dropout": dropout,
        "n_src": n_src,
    }
    return enc, meta


def make_view(item, src, value, dt, mask, lab_flag=None, drop_prob=0.15, v_noise=0.01, dt_noise=0.01):
    keep = mask.clone()
    if drop_prob > 0:
        drop = (torch.rand_like(keep.float()) < drop_prob) & keep
        keep = keep & (~drop)
    item2 = item.clone()
    src2 = src.clone()
    value2 = value.clone()
    dt2 = dt.clone()
    mask2 = keep
    pad = ~mask2
    item2[pad] = 0
    src2[pad] = 0
    value2[pad] = 0.0
    dt2[pad] = 0.0
    lab2 = None
    if lab_flag is not None:
        lab2 = lab_flag.clone()
        lab2[pad] = 0
    if v_noise > 0:
        value2 = value2 + torch.randn_like(value2) * v_noise * mask2.to(value2.dtype)
    if dt_noise > 0:
        dt2 = dt2 + torch.randn_like(dt2) * dt_noise * mask2.to(dt2.dtype)
    return item2, src2, value2, dt2, mask2, lab2


def nt_xent_loss(z1, z2, temp=0.1):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0).float()
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)
    pos = torch.arange(B, device=z.device)
    labels = torch.cat([pos + B, pos], dim=0)
    return F.cross_entropy(sim, labels)


def pretrain_ts_encoder_simclr(
    encoder: nn.Module,
    ts_tensor_dir: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    temp: float = 0.1,
    drop_prob: float = 0.15,
    value_noise_std: float = 0.01,
    dt_noise_std: float = 0.01,
    use_amp: bool = True,
    seed: int = 42,
    num_workers: int = 0,
):
    set_seed(seed)
    device = next(encoder.parameters()).device
    bucket_paths = list_bucket_pts(ts_tensor_dir)
    if len(bucket_paths) == 0:
        raise FileNotFoundError(f"No bucket_XXX.pt found in: {ts_tensor_dir}")

    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    encoder.train()
    for ep in range(epochs):
        random.shuffle(bucket_paths)
        total = 0.0
        n = 0
        for bp in bucket_paths:
            ds = BucketPtDataset(bp)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn_basic,
                drop_last=True,
            )
            for batch in dl:
                item = batch["item"].to(device)
                src = batch["src"].to(device)
                value = batch["value"].to(device)
                dt = batch["dt"].to(device)
                mask = batch["mask"].to(device)
                lab_flag = batch.get("lab_flag", None)
                if lab_flag is not None:
                    lab_flag = lab_flag.to(device)

                v1 = make_view(item, src, value, dt, mask, lab_flag, drop_prob, value_noise_std, dt_noise_std)
                v2 = make_view(item, src, value, dt, mask, lab_flag, drop_prob, value_noise_std, dt_noise_std)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                    z1 = encoder(*v1)
                    z2 = encoder(*v2)
                    loss = nt_xent_loss(z1, z2, temp=temp)

                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

                total += float(loss.detach().cpu())
                n += 1

            del ds, dl
            gc.collect()

        print(f"[SIMCLR] epoch {ep+1}/{epochs} | loss={total/max(n,1):.6f}")


@torch.no_grad()
def export_embeddings_event_transformer(
    encoder: nn.Module,
    ts_tensor_dir: str,
    out_dir: str,
    batch_size: int = 256,
    use_amp: bool = True,
    save_fp16: bool = True,
    num_workers: int = 0,
    skip_if_exists: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    device = next(encoder.parameters()).device
    bucket_paths = list_bucket_pts(ts_tensor_dir)
    if len(bucket_paths) == 0:
        raise FileNotFoundError(f"No bucket_XXX.pt found in: {ts_tensor_dir}")

    encoder.eval()
    for bp in bucket_paths:
        bucket_id = get_bucket_id_from_path(bp)
        out_path = os.path.join(out_dir, f"bucket_{bucket_id:03d}_ts_emb.pt")
        if skip_if_exists and os.path.exists(out_path):
            print(f"[SKIP] bucket {bucket_id:03d} exists: {out_path}")
            continue

        ds = BucketPtDataset(bp)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_basic,
            drop_last=False,
        )

        all_ids: List[str] = []
        all_emb: List[torch.Tensor] = []

        for batch in dl:
            item = batch["item"].to(device)
            src = batch["src"].to(device)
            value = batch["value"].to(device)
            dt = batch["dt"].to(device)
            mask = batch["mask"].to(device)
            lab_flag = batch.get("lab_flag", None)
            if lab_flag is not None:
                lab_flag = lab_flag.to(device)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                emb = encoder(item, src, value, dt, mask, lab_flag)

            emb = emb.detach().cpu()
            if save_fp16:
                emb = emb.to(torch.float16)

            all_ids.extend(batch["sample_id"])
            all_emb.append(emb)

        ts_emb = torch.cat(all_emb, dim=0)
        torch.save(
            {
                "bucket_id": bucket_id,
                "sample_id": all_ids,
                "ts_emb": ts_emb,
                "meta": {"dim": int(ts_emb.shape[1]), "save_fp16": bool(save_fp16), "mode": "event_transformer"},
            },
            out_path,
        )
        with open(os.path.join(out_dir, f"bucket_{bucket_id:03d}_schema.json"), "w", encoding="utf-8") as f:
            json.dump({"bucket": bucket_id, "num_rows": int(ts_emb.shape[0]), "dim": int(ts_emb.shape[1])}, f, ensure_ascii=False, indent=2)

        print(f"[DUMP] bucket {bucket_id:03d}: {ts_emb.shape} -> {out_path}")

        del ds, dl, all_ids, all_emb, ts_emb
        gc.collect()


class EventTokenProjector(nn.Module):
    def __init__(self, hash_vocab: int, n_src: int = 8, d: int = 128, use_lab_flag: bool = True):
        super().__init__()
        self.use_lab_flag = use_lab_flag
        self.item_emb = nn.Embedding(hash_vocab + 1, d, padding_idx=0)
        self.src_emb = nn.Embedding(n_src + 1, d, padding_idx=0)
        self.val_proj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.dt_proj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        if use_lab_flag:
            self.flag_emb = nn.Embedding(4, d, padding_idx=0)
        self.mask_token = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)
        self.norm = nn.LayerNorm(d)

    def forward(self, item, src, value, dt, mask, lab_flag=None, mask_pos: Optional[torch.Tensor] = None):
        x = self.item_emb(item) + self.src_emb(src)
        x = x + self.val_proj(value.unsqueeze(-1).to(x.dtype))
        x = x + self.dt_proj(dt.unsqueeze(-1).to(x.dtype))
        if self.use_lab_flag and (lab_flag is not None):
            lf = lab_flag.to(torch.int64)
            mapped = torch.zeros_like(lf)
            mapped[lf < 0] = 1
            mapped[lf > 0] = 3
            x = x + self.flag_emb(mapped)
        if mask_pos is not None:
            x = x + mask_pos.to(x.dtype).unsqueeze(-1) * self.mask_token.view(1, 1, -1)
        x = x * mask.to(x.dtype).unsqueeze(-1)
        return self.norm(x)


class TemporalEncoderCMGRW_Flex(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        lambda_rec: float = 1.0,
        model_type: str = "gru",
        pooling: Literal["last", "mean"] = "last",
        sigma_schedule: Literal["uniform", "log_uniform"] = "log_uniform",
        consistency_mode: Literal["two_sample", "teacher_student"] = "two_sample",
        teacher_on: Literal["f2", "f1"] = "f2",
        detach_teacher: bool = True,
        mlp_mult: int = 2,
        fuse_f_theta: bool = True,
        nhead: int = 4,
        transformer_ff_mult: int = 2,
        transformer_dropout: float = 0.1,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_activation: Literal["relu", "silu", "gelu"] = "relu",
        tcn_causal: bool = True,
        downsample_stride: int = 1,
        downsample_mode: Literal["avg", "conv"] = "avg",
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lambda_rec = float(lambda_rec)
        self.model_type = model_type.lower()
        self.pooling = pooling
        self.sigma_schedule = sigma_schedule
        self.consistency_mode = consistency_mode
        self.teacher_on = teacher_on
        self.detach_teacher = detach_teacher
        self.fuse_f_theta = fuse_f_theta
        self.tcn_causal = tcn_causal
        self.downsample_stride = int(downsample_stride)

        if self.downsample_stride > 1:
            if downsample_mode == "avg":
                self.downsampler = nn.AvgPool1d(kernel_size=self.downsample_stride, stride=self.downsample_stride)
            elif downsample_mode == "conv":
                self.downsampler = nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=self.downsample_stride,
                    stride=self.downsample_stride,
                    padding=0,
                    groups=1,
                    bias=True,
                )
            else:
                raise ValueError(f"Unsupported downsample_mode={downsample_mode}")
        else:
            self.downsampler = None

        if self.model_type == "gru":
            self.encoder = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
            self.input_proj = None
        elif self.model_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * transformer_ff_mult,
                batch_first=True,
                dropout=transformer_dropout,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        elif self.model_type == "tcn":
            act_cls = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[tcn_activation]
            layers: List[nn.Module] = []
            in_ch = input_dim
            for i in range(n_layers):
                out_ch = hidden_dim
                dilation = 2**i
                pad = (tcn_kernel_size - 1) * dilation
                layers.append(nn.Conv1d(in_ch, out_ch, tcn_kernel_size, dilation=dilation, padding=pad))
                layers.append(act_cls())
                layers.append(nn.Dropout(tcn_dropout))
                in_ch = out_ch
            self.encoder = nn.Sequential(*layers)
            self.input_proj = None
        else:
            raise ValueError(f"Unsupported model_type={model_type}")

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.f_theta = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * mlp_mult),
            nn.SiLU(),
            nn.Linear(hidden_dim * mlp_mult, hidden_dim),
        )

    def _maybe_downsample(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.downsampler is None:
            return x_seq
        x = x_seq.transpose(1, 2)
        x = self.downsampler(x)
        x = x.transpose(1, 2)
        return x

    def _sample_sigma(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sigma_schedule == "uniform":
            u = torch.rand(B, 1, device=device, dtype=dtype)
            return u * (self.sigma_max - self.sigma_min) + self.sigma_min
        if self.sigma_schedule == "log_uniform":
            u = torch.rand(B, 1, device=device, dtype=dtype)
            log_min = math.log(self.sigma_min)
            log_max = math.log(self.sigma_max)
            return torch.exp(log_min + u * (log_max - log_min))
        raise ValueError(f"Unsupported sigma_schedule={self.sigma_schedule}")

    def _masked_mean(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(h.dtype).unsqueeze(-1)
        denom = w.sum(dim=1).clamp_min(1.0)
        return (h * w).sum(dim=1) / denom

    def _encode_tokens_and_z0(self, x_seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_type == "gru":
            z_enc, _ = self.encoder(x_seq)
            h_tok = z_enc
            if self.pooling == "last":
                last_idx = mask.to(torch.int64).sum(dim=1).clamp_min(1) - 1
                z0 = h_tok[torch.arange(h_tok.size(0), device=h_tok.device), last_idx]
            else:
                z0 = self._masked_mean(h_tok, mask)
            return h_tok, self.proj(z0)

        if self.model_type == "transformer":
            h = self.input_proj(x_seq)
            h_tok = self.encoder(h, src_key_padding_mask=(~mask))
            if self.pooling == "last":
                last_idx = mask.to(torch.int64).sum(dim=1).clamp_min(1) - 1
                z0 = h_tok[torch.arange(h_tok.size(0), device=h_tok.device), last_idx]
            else:
                z0 = self._masked_mean(h_tok, mask)
            return h_tok, self.proj(z0)

        h = x_seq.transpose(1, 2)
        z = self.encoder(h)
        if self.tcn_causal:
            L = x_seq.size(1)
            z = z[..., -L:]
        h_tok = z.transpose(1, 2)
        if self.pooling == "last":
            last_idx = mask.to(torch.int64).sum(dim=1).clamp_min(1) - 1
            z0 = h_tok[torch.arange(h_tok.size(0), device=h_tok.device), last_idx]
        else:
            z0 = self._masked_mean(h_tok, mask)
        return h_tok, self.proj(z0)

    def _f_theta_batched(self, zt1, s1, zt2, s2):
        if self.fuse_f_theta:
            zt = torch.cat([zt1, zt2], dim=0)
            s = torch.cat([s1, s2], dim=0)
            f = self.f_theta(torch.cat([zt, s], dim=-1))
            f1, f2 = f.chunk(2, dim=0)
            return f1, f2
        f1 = self.f_theta(torch.cat([zt1, s1], dim=-1))
        f2 = self.f_theta(torch.cat([zt2, s2], dim=-1))
        return f1, f2

    def forward(self, x_seq: torch.Tensor, mask: Optional[torch.Tensor] = None, return_tokens: bool = False):
        x_seq = self._maybe_downsample(x_seq)
        B = x_seq.size(0)
        device = x_seq.device
        dtype = x_seq.dtype
        if mask is None:
            mask = torch.ones(B, x_seq.size(1), device=device, dtype=torch.bool)

        h_tok, z0 = self._encode_tokens_and_z0(x_seq, mask)

        eps1 = torch.randn_like(z0)
        eps2 = torch.randn_like(z0)
        sigma_t1 = self._sample_sigma(B, device, dtype)
        sigma_t2 = self._sample_sigma(B, device, dtype)

        zt1 = z0 + sigma_t1 * eps1
        zt2 = z0 + sigma_t2 * eps2

        f1, f2 = self._f_theta_batched(zt1, sigma_t1, zt2, sigma_t2)

        if self.consistency_mode == "two_sample":
            loss_cons = F.mse_loss(f1, f2) + self.lambda_rec * F.mse_loss(f1, z0)
        elif self.consistency_mode == "teacher_student":
            if self.teacher_on == "f2":
                teacher = f2.detach() if self.detach_teacher else f2
                student = f1
            elif self.teacher_on == "f1":
                teacher = f1.detach() if self.detach_teacher else f1
                student = f2
            else:
                raise ValueError(f"Unsupported teacher_on={self.teacher_on}")
            loss_cons = F.mse_loss(student, teacher) + self.lambda_rec * F.mse_loss(student, z0)
        else:
            raise ValueError(f"Unsupported consistency_mode={self.consistency_mode}")

        if return_tokens:
            return z0, loss_cons, h_tok
        return z0, loss_cons


def sample_mask_positions(mask: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return torch.zeros_like(mask)
    r = torch.rand_like(mask.float())
    return (r < p) & mask


def apply_masking(item, value, dt, src, mask, lab_flag, mask_pos):
    item_in = item.clone()
    value_in = value.clone()
    dt_in = dt.clone()
    src_in = src.clone()
    lab_in = lab_flag.clone() if lab_flag is not None else None
    item_in[mask_pos] = 0
    src_in[mask_pos] = 0
    value_in[mask_pos] = 0.0
    dt_in[mask_pos] = 0.0
    if lab_in is not None:
        lab_in[mask_pos] = 0
    return item_in, value_in, dt_in, src_in, lab_in


def train_export_cmgrw(
    ts_tensor_dir: str,
    out_dir: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    use_amp: bool,
    seed: int,
    input_dim: int,
    hidden_dim: int,
    n_layers: int,
    nhead: int,
    ff_mult: int,
    dropout: float,
    sigma_min: float,
    sigma_max: float,
    sigma_schedule: str,
    consistency_mode: str,
    teacher_on: str,
    detach_teacher: bool,
    fuse_f_theta: bool,
    lambda_rec: float,
    pooling: str,
    mask_prob: float,
    w_cons: float,
    w_value: float,
    dump_batch: int,
    save_fp16: bool,
    skip_if_exists: bool,
    num_workers: int,
    value_compress: str,
    value_clip: float,
    value_eps: float,
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    bucket_paths = list_bucket_pts(ts_tensor_dir)
    if len(bucket_paths) == 0:
        raise RuntimeError(f"No bucket_XXX.pt found under: {ts_tensor_dir}")

    tmp = torch.load(bucket_paths[0], map_location="cpu")
    meta0 = tmp.get("meta", {})
    hash_v = int(meta0.get("HASH_V", 100_000))
    use_lab_flag = ("lab_flag" in tmp)
    del tmp

    collate_fn = make_collate_fn_value_compress(mode=value_compress, clip=value_clip, eps=value_eps)

    projector = EventTokenProjector(hash_vocab=hash_v, n_src=8, d=input_dim, use_lab_flag=use_lab_flag).to(device)

    temporal = TemporalEncoderCMGRW_Flex(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        lambda_rec=lambda_rec,
        model_type="transformer",
        pooling=pooling,  # type: ignore
        sigma_schedule=sigma_schedule,  # type: ignore
        consistency_mode=consistency_mode,  # type: ignore
        teacher_on=teacher_on,  # type: ignore
        detach_teacher=detach_teacher,
        fuse_f_theta=fuse_f_theta,
        nhead=nhead,
        transformer_ff_mult=ff_mult,
        transformer_dropout=dropout,
        downsample_stride=1,
    ).to(device)

    value_head = nn.Linear(hidden_dim, 1).to(device)

    params = list(projector.parameters()) + list(temporal.parameters()) + list(value_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    amp_on = (use_amp and device == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_on)

    projector.train()
    temporal.train()
    value_head.train()

    for ep in range(epochs):
        random.shuffle(bucket_paths)
        total = 0.0
        total_cons = 0.0
        total_val = 0.0
        n_batches = 0

        for bp in bucket_paths:
            ds = BucketPtDataset(bp)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )

            for batch in dl:
                item = batch["item"].to(device)
                src = batch["src"].to(device)
                value = batch["value"].to(device)
                dt = batch["dt"].to(device)
                mask = batch["mask"].to(device)
                lab_flag = batch.get("lab_flag", None)
                if lab_flag is not None:
                    lab_flag = lab_flag.to(device)

                mask_pos = sample_mask_positions(mask, mask_prob)
                if mask_pos.sum().item() == 0:
                    continue

                item_in, value_in, dt_in, src_in, lab_in = apply_masking(item, value, dt, src, mask, lab_flag, mask_pos)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=amp_on):
                    x_seq = projector(item_in, src_in, value_in, dt_in, mask, lab_in, mask_pos=mask_pos)
                    z0, loss_cons, h_tok = temporal(x_seq, mask=mask, return_tokens=True)
                    h_m = h_tok[mask_pos]
                    value_tgt = value[mask_pos].float()
                    value_pred = value_head(h_m).squeeze(-1)
                    loss_val = F.mse_loss(value_pred.float(), value_tgt)
                    loss = w_cons * loss_cons + w_value * loss_val

                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(opt)
                scaler.update()

                total += float(loss.detach().cpu())
                total_cons += float(loss_cons.detach().cpu())
                total_val += float(loss_val.detach().cpu())
                n_batches += 1

            del ds, dl
            gc.collect()

        print(
            f"[CMGRW] epoch {ep+1}/{epochs} | "
            f"loss={total/max(n_batches,1):.6f} | "
            f"cons={total_cons/max(n_batches,1):.6f} | "
            f"val={total_val/max(n_batches,1):.6f}"
        )

    ckpt_path = os.path.join(out_dir, "ts_cmgrw_maskrecon_valueonly_ckpt.pt")
    torch.save(
        {
            "projector": projector.state_dict(),
            "temporal": temporal.state_dict(),
            "value_head": value_head.state_dict(),
            "meta": {
                "hash_v": hash_v,
                "use_lab_flag": use_lab_flag,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "mask_prob": mask_prob,
                "w_cons": w_cons,
                "w_value": w_value,
                "temporal_model_type": "transformer",
                "pooling": pooling,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "sigma_schedule": sigma_schedule,
                "consistency_mode": consistency_mode,
                "lambda_rec": lambda_rec,
                "n_layers": n_layers,
                "nhead": nhead,
                "ff_mult": ff_mult,
                "dropout": dropout,
                "downsample_stride": 1,
                "value_compress": value_compress,
                "value_clip": value_clip,
            },
        },
        ckpt_path,
    )
    print("[SAVE] ckpt:", ckpt_path)

    projector.eval()
    temporal.eval()

    for bp in list_bucket_pts(ts_tensor_dir):
        bucket_id = get_bucket_id_from_path(bp)
        out_path = os.path.join(out_dir, f"bucket_{bucket_id:03d}_ts_emb.pt")
        if skip_if_exists and os.path.exists(out_path):
            print(f"[SKIP] bucket {bucket_id:03d} exists: {out_path}")
            continue

        ds = BucketPtDataset(bp)
        dl = DataLoader(
            ds,
            batch_size=dump_batch,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )

        all_ids: List[str] = []
        all_emb: List[torch.Tensor] = []

        with torch.inference_mode():
            for batch in dl:
                item = batch["item"].to(device)
                src = batch["src"].to(device)
                value = batch["value"].to(device)
                dt = batch["dt"].to(device)
                mask = batch["mask"].to(device)
                lab_flag = batch.get("lab_flag", None)
                if lab_flag is not None:
                    lab_flag = lab_flag.to(device)

                with torch.amp.autocast(device_type="cuda", enabled=amp_on):
                    x_seq = projector(item, src, value, dt, mask, lab_flag, mask_pos=None)
                    z0, _ = temporal(x_seq, mask=mask)

                z0 = z0.detach().cpu()
                if save_fp16:
                    z0 = z0.to(torch.float16)

                all_ids.extend(batch["sample_id"])
                all_emb.append(z0)

        ts_emb = torch.cat(all_emb, dim=0)

        torch.save(
            {
                "bucket_id": bucket_id,
                "sample_id": all_ids,
                "ts_emb": ts_emb,
                "meta": {
                    "dim": int(ts_emb.shape[1]),
                    "save_fp16": bool(save_fp16),
                    "ckpt": os.path.basename(ckpt_path),
                    "mode": "cmgrw_maskrecon_valueonly",
                    "value_compress": value_compress,
                    "value_clip": value_clip,
                },
            },
            out_path,
        )
        with open(os.path.join(out_dir, f"bucket_{bucket_id:03d}_schema.json"), "w", encoding="utf-8") as f:
            json.dump({"bucket": bucket_id, "num_rows": int(ts_emb.shape[0]), "dim": int(ts_emb.shape[1])}, f, ensure_ascii=False, indent=2)

        print(f"[DUMP] bucket {bucket_id:03d}: {ts_emb.shape} -> {out_path}")

        del ds, dl, all_ids, all_emb, ts_emb
        gc.collect()

    print("[DONE] embeddings saved to:", out_dir)


def run_simclr(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.root
    ts_dir = os.path.join(root, args.ts_subdir)
    out_dir = args.out_dir or os.path.join(root, args.out_subdir)

    enc, meta = build_ts_encoder_from_dir(
        ts_dir,
        d_model=args.d_model,
        nhead=args.nhead,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_src=args.n_src,
        device=device,
    )

    pretrain_ts_encoder_simclr(
        enc,
        ts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        temp=args.temp,
        drop_prob=args.drop_prob,
        value_noise_std=args.value_noise,
        dt_noise_std=args.dt_noise,
        use_amp=args.amp,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    ckpt_path = os.path.join(out_dir, "ts_simclr_event_transformer_ckpt.pt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"encoder": enc.state_dict(), "meta": meta}, ckpt_path)
    print("[SAVE] ckpt:", ckpt_path)

    export_embeddings_event_transformer(
        enc,
        ts_dir,
        out_dir=out_dir,
        batch_size=args.dump_batch,
        use_amp=args.amp,
        save_fp16=args.save_fp16,
        num_workers=args.num_workers,
        skip_if_exists=args.skip_export_if_exists,
    )


def run_cmgrw(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = args.root
    ts_dir = os.path.join(root, args.ts_subdir)
    out_dir = args.out_dir or os.path.join(root, args.out_subdir)

    train_export_cmgrw(
        ts_tensor_dir=ts_dir,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        seed=args.seed,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        nhead=args.nhead,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        sigma_schedule=args.sigma_schedule,
        consistency_mode=args.consistency_mode,
        teacher_on=args.teacher_on,
        detach_teacher=args.detach_teacher,
        fuse_f_theta=args.fuse_f_theta,
        lambda_rec=args.lambda_rec,
        pooling=args.pooling,
        mask_prob=args.mask_prob,
        w_cons=args.w_cons,
        w_value=args.w_value,
        dump_batch=args.dump_batch,
        save_fp16=args.save_fp16,
        skip_if_exists=args.skip_export_if_exists,
        num_workers=args.num_workers,
        value_compress=args.value_compress,
        value_clip=args.value_clip,
        value_eps=args.value_eps,
    )


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("simclr")
    ps.add_argument("--root", type=str, required=True)
    ps.add_argument("--ts-subdir", type=str, default="timeseries_tensor_v1")
    ps.add_argument("--out-dir", type=str, default=None)
    ps.add_argument("--out-subdir", type=str, default="ts_emb_simclr_event_transformer")
    ps.add_argument("--device", type=str, default=None)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--epochs", type=int, default=3)
    ps.add_argument("--batch-size", type=int, default=256)
    ps.add_argument("--lr", type=float, default=2e-4)
    ps.add_argument("--weight-decay", type=float, default=0.01)
    ps.add_argument("--grad-clip", type=float, default=1.0)
    ps.add_argument("--amp", action="store_true")
    ps.add_argument("--no-amp", dest="amp", action="store_false")
    ps.set_defaults(amp=True)
    ps.add_argument("--d-model", type=int, default=128)
    ps.add_argument("--nhead", type=int, default=4)
    ps.add_argument("--n-layers", type=int, default=2)
    ps.add_argument("--dropout", type=float, default=0.1)
    ps.add_argument("--n-src", type=int, default=8)
    ps.add_argument("--temp", type=float, default=0.1)
    ps.add_argument("--drop-prob", type=float, default=0.15)
    ps.add_argument("--value-noise", type=float, default=0.01)
    ps.add_argument("--dt-noise", type=float, default=0.01)
    ps.add_argument("--dump-batch", type=int, default=512)
    ps.add_argument("--save-fp16", action="store_true")
    ps.add_argument("--no-save-fp16", dest="save_fp16", action="store_false")
    ps.set_defaults(save_fp16=True)
    ps.add_argument("--skip-export-if-exists", action="store_true")
    ps.add_argument("--no-skip-export-if-exists", dest="skip_export_if_exists", action="store_false")
    ps.set_defaults(skip_export_if_exists=True)
    ps.add_argument("--num-workers", type=int, default=0)
    ps.set_defaults(func=run_simclr)

    pc = sub.add_parser("cmgrw")
    pc.add_argument("--root", type=str, required=True)
    pc.add_argument("--ts-subdir", type=str, default="timeseries_tensor_v1")
    pc.add_argument("--out-dir", type=str, default=None)
    pc.add_argument("--out-subdir", type=str, default="ts_emb_cmgrw_maskrecon_valueonly")
    pc.add_argument("--device", type=str, default=None)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--epochs", type=int, default=3)
    pc.add_argument("--batch-size", type=int, default=256)
    pc.add_argument("--lr", type=float, default=2e-4)
    pc.add_argument("--weight-decay", type=float, default=0.01)
    pc.add_argument("--grad-clip", type=float, default=1.0)
    pc.add_argument("--amp", action="store_true")
    pc.add_argument("--no-amp", dest="amp", action="store_false")
    pc.set_defaults(amp=True)
    pc.add_argument("--input-dim", type=int, default=128)
    pc.add_argument("--hidden-dim", type=int, default=128)
    pc.add_argument("--n-layers", type=int, default=2)
    pc.add_argument("--nhead", type=int, default=4)
    pc.add_argument("--ff-mult", type=int, default=2)
    pc.add_argument("--dropout", type=float, default=0.1)
    pc.add_argument("--sigma-min", type=float, default=0.05)
    pc.add_argument("--sigma-max", type=float, default=0.5)
    pc.add_argument("--sigma-schedule", type=str, default="log_uniform")
    pc.add_argument("--consistency-mode", type=str, default="two_sample")
    pc.add_argument("--teacher-on", type=str, default="f2")
    pc.add_argument("--detach-teacher", action="store_true")
    pc.add_argument("--no-detach-teacher", dest="detach_teacher", action="store_false")
    pc.set_defaults(detach_teacher=True)
    pc.add_argument("--fuse-f-theta", action="store_true")
    pc.add_argument("--no-fuse-f-theta", dest="fuse_f_theta", action="store_false")
    pc.set_defaults(fuse_f_theta=True)
    pc.add_argument("--lambda-rec", type=float, default=1.0)
    pc.add_argument("--pooling", type=str, default="mean")
    pc.add_argument("--mask-prob", type=float, default=0.15)
    pc.add_argument("--w-cons", type=float, default=1.0)
    pc.add_argument("--w-value", type=float, default=1.0)
    pc.add_argument("--dump-batch", type=int, default=512)
    pc.add_argument("--save-fp16", action="store_true")
    pc.add_argument("--no-save-fp16", dest="save_fp16", action="store_false")
    pc.set_defaults(save_fp16=True)
    pc.add_argument("--skip-export-if-exists", action="store_true")
    pc.add_argument("--no-skip-export-if-exists", dest="skip_export_if_exists", action="store_false")
    pc.set_defaults(skip_export_if_exists=True)
    pc.add_argument("--num-workers", type=int, default=0)
    pc.add_argument("--value-compress", type=str, default="signed_log1p")
    pc.add_argument("--value-clip", type=float, default=10.0)
    pc.add_argument("--value-eps", type=float, default=1e-6)
    pc.set_defaults(func=run_cmgrw)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
