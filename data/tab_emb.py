import os
import json
import math
import gc
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    root: str = r"E:/NUS/data/perdata/train_text_all_samples"
    tab_tensor_dir_name: str = "tab_tensor_v1"
    num_buckets: int = 256

    # output
    out_name: str = "tab_embed_v1_flow"  # default; can override by CLI
    save_float16: bool = True
    dump_batch: int = 4096

    # training
    epochs: int = 2
    batch_size: int = 512
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42

    # standardize
    use_global_standardize: bool = True

    # mode
    mode: str = "flow_old"  # flow_old | mlp_simclr

    cat_embed_dim: int = 16
    out_dim: int = 128
    reg_weights: Dict[str, float] = None  # {"ent":0,"lap":0,"stein":0}

    mlp_hidden_dims: Tuple[int, ...] = (256, 256)
    mlp_dropout: float = 0.1

    # SimCLR
    simclr_temp: float = 0.1
    num_noise_std: float = 0.01
    num_mask_prob: float = 0.05
    cat_replace_prob: float = 0.05

    # precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_num: torch.dtype = torch.float32


def parse_args() -> Config:
    p = argparse.ArgumentParser("Train tabular embedding encoder (flow_old or mlp_simclr)")

    p.add_argument("--root", type=str, default=r"E:/NUS/data/perdata/train_text_all_samples")
    p.add_argument("--tab_tensor_dir_name", type=str, default="tab_tensor_v1")
    p.add_argument("--num_buckets", type=int, default=256)

    p.add_argument("--mode", type=str, default="flow_old", choices=["flow_old", "mlp_simclr"])
    p.add_argument("--out_name", type=str, default=None, help="Output folder name under root")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_global_standardize", action="store_true")
    p.add_argument("--no_global_standardize", action="store_true")
    p.add_argument("--save_float16", action="store_true")
    p.add_argument("--no_save_float16", action="store_true")
    p.add_argument("--dump_batch", type=int, default=4096)

    # shared dims
    p.add_argument("--cat_embed_dim", type=int, default=16)
    p.add_argument("--out_dim", type=int, default=128)

    # flow_old regularization
    p.add_argument("--reg_ent", type=float, default=0.0)
    p.add_argument("--reg_lap", type=float, default=0.0)
    p.add_argument("--reg_stein", type=float, default=0.0)

    # mlp_simclr
    p.add_argument("--mlp_hidden_dims", type=str, default="256,256")
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    p.add_argument("--simclr_temp", type=float, default=0.1)
    p.add_argument("--num_noise_std", type=float, default=0.01)
    p.add_argument("--num_mask_prob", type=float, default=0.05)
    p.add_argument("--cat_replace_prob", type=float, default=0.05)

    args = p.parse_args()

    cfg = Config()
    cfg.root = args.root
    cfg.tab_tensor_dir_name = args.tab_tensor_dir_name
    cfg.num_buckets = args.num_buckets

    cfg.mode = args.mode
    cfg.out_name = args.out_name if args.out_name is not None else ("tab_embed_v1_flow" if cfg.mode == "flow_old" else "tab_embed_v1_mlp")

    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.seed = args.seed

    if args.use_global_standardize and args.no_global_standardize:
        raise ValueError("Cannot set both --use_global_standardize and --no_global_standardize")
    if args.use_global_standardize:
        cfg.use_global_standardize = True
    if args.no_global_standardize:
        cfg.use_global_standardize = False

    if args.save_float16 and args.no_save_float16:
        raise ValueError("Cannot set both --save_float16 and --no_save_float16")
    if args.save_float16:
        cfg.save_float16 = True
    if args.no_save_float16:
        cfg.save_float16 = False

    cfg.dump_batch = args.dump_batch

    cfg.cat_embed_dim = args.cat_embed_dim
    cfg.out_dim = args.out_dim

    cfg.reg_weights = {"ent": args.reg_ent, "lap": args.reg_lap, "stein": args.reg_stein}

    cfg.mlp_hidden_dims = tuple(int(x.strip()) for x in args.mlp_hidden_dims.split(",") if x.strip())
    cfg.mlp_dropout = args.mlp_dropout
    cfg.simclr_temp = args.simclr_temp
    cfg.num_noise_std = args.num_noise_std
    cfg.num_mask_prob = args.num_mask_prob
    cfg.cat_replace_prob = args.cat_replace_prob

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


class BucketIO:
    def __init__(self, tab_tensor_dir: str, num_buckets: int):
        self.tab_tensor_dir = tab_tensor_dir
        self.num_buckets = num_buckets

    def bucket_dir(self, b: int) -> str:
        return os.path.join(self.tab_tensor_dir, f"bucket_{b:03d}")

    def exists_bucket(self, b: int) -> bool:
        return os.path.isfile(os.path.join(self.bucket_dir(b), "schema.json"))

    def load_bucket_arrays(self, b: int):
        bd = self.bucket_dir(b)
        with open(os.path.join(bd, "schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)

        X_num = np.load(os.path.join(bd, "numeric.npy")).astype(np.float32)
        sample_ids = np.load(os.path.join(bd, "sample_id.npy"), allow_pickle=True)

        cat_cols = schema["cat_cols"]
        cat_list = []
        for i, c in enumerate(cat_cols):
            p = os.path.join(bd, f"cat_{i:02d}_{c}.npy")
            cat_list.append(np.load(p).astype(np.int64))

        return schema, sample_ids, X_num, cat_list

    def pick_first_schema(self) -> dict:
        for b in range(self.num_buckets):
            if self.exists_bucket(b):
                schema0, _, _, _ = self.load_bucket_arrays(b)
                return schema0
        raise RuntimeError(f"No bucket found under: {self.tab_tensor_dir}")

    @torch.no_grad()
    def streaming_mean_std_fast(self) -> Tuple[np.ndarray, np.ndarray]:
        n = 0
        sum_ = None
        sumsq = None

        for b in range(self.num_buckets):
            if not self.exists_bucket(b):
                continue
            _, _, X_num, _ = self.load_bucket_arrays(b)  # [N,D] float32
            x = X_num.astype(np.float64, copy=False)

            if sum_ is None:
                D = x.shape[1]
                sum_ = np.zeros((D,), dtype=np.float64)
                sumsq = np.zeros((D,), dtype=np.float64)

            sum_ += x.sum(axis=0)
            sumsq += (x * x).sum(axis=0)
            n += x.shape[0]

            del X_num
            gc.collect()

        mean = (sum_ / max(n, 1)).astype(np.float32)
        var = (sumsq / max(n, 1) - mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
        return mean, std


def iter_minibatches(X_num: torch.Tensor, cats: List[torch.Tensor], batch_size: int, shuffle: bool = True):
    N = X_num.size(0)
    idx = torch.randperm(N, device=X_num.device) if shuffle else torch.arange(N, device=X_num.device)
    for s in range(0, N, batch_size):
        j = idx[s:s + batch_size]
        yield X_num[j], [c[j] for c in cats]


def standardize(x: torch.Tensor, mean_t: Optional[torch.Tensor], std_t: Optional[torch.Tensor], use_global: bool):
    if use_global:
        assert mean_t is not None and std_t is not None
        return (x - mean_t) / std_t
    else:
        m = x.mean(0, keepdim=True)
        s = x.std(0, keepdim=True).clamp_min(1e-6)
        return (x - m) / s


class SoftCategoryEmbeddingReg(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        temperature: float = 1.0,
        reg_weights: Optional[dict] = None,
        sim_matrix: Optional[torch.Tensor] = None,
        stein_eps: float = 1e-4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = float(temperature)
        self.stein_eps = float(stein_eps)

        self.E = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.W = nn.Parameter(torch.randn(num_classes, num_classes) * 0.02)

        weights = {"ent": 0.0, "lap": 0.0, "stein": 0.0}
        if reg_weights:
            weights.update(reg_weights)
        self.reg_weights = weights

        S = sim_matrix if sim_matrix is not None else torch.eye(num_classes)
        self.register_buffer("S", S)

    def forward(self, idx: torch.LongTensor):
        Tmix = F.softmax(self.W / self.temperature, dim=-1)
        p = Tmix[idx]
        e = p @ self.E
        reg_loss = self._regularization(Tmix)
        return e, reg_loss

    def _regularization(self, Tmix: torch.Tensor) -> torch.Tensor:
        reg = Tmix.new_zeros(())
        V = self.num_classes

        w_ent = self.reg_weights.get("ent", 0.0)
        if w_ent != 0.0:
            entropy = -(Tmix * torch.log(Tmix.clamp_min(1e-8))).sum(dim=-1).mean()
            reg = reg + w_ent * entropy

        w_lap = self.reg_weights.get("lap", 0.0)
        if w_lap != 0.0:
            diff = self.E.unsqueeze(0) - self.E.unsqueeze(1)
            dist2 = (diff ** 2).sum(-1)
            lap = (self.S * dist2).sum() / (V * V)
            reg = reg + w_lap * lap

        w_stein = self.reg_weights.get("stein", 0.0)
        if w_stein != 0.0:
            E0 = self.E - self.E.mean(0, keepdim=True)
            cov = (E0.t() @ E0) / V
            cov = cov + self.stein_eps * torch.eye(
                cov.size(0), device=cov.device, dtype=cov.dtype
            )
            sol = torch.linalg.solve(cov, E0.t()).t()
            score = -sol
            stein = -torch.mean(torch.sum(score * E0, dim=-1))
            reg = reg + w_stein * stein

        return reg


class FlowMatchingContinuousEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = float(sigma)

        in_dim = input_dim + 1 + cond_dim
        layers: List[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B, _ = x0.shape
        device = x0.device

        x1 = torch.randn_like(x0) * self.sigma
        t = torch.rand(B, 1, device=device, dtype=x0.dtype)
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0

        if cond is not None:
            inp = torch.cat([x_t, t, cond], dim=-1)
        else:
            inp = torch.cat([x_t, t], dim=-1)

        v_pred = self.net(inp)
        loss = F.mse_loss(v_pred, u_t)
        return loss, x_t, v_pred

    @torch.inference_mode()
    def encode_deterministic(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B, _ = x0.shape
        t = torch.zeros(B, 1, device=x0.device, dtype=x0.dtype)
        x_t = x0
        if cond is not None:
            inp = torch.cat([x_t, t, cond], dim=-1)
        else:
            inp = torch.cat([x_t, t], dim=-1)

        v_pred = self.net(inp)
        return v_pred


class TabularEncoder_FlowOld(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: List[int],
        cat_embed_dim: int = 16,
        reg_weights: Optional[dict] = None,
        cond_mode: str = "concat",
        out_dim: int = 128,
    ):
        super().__init__()
        self.cond_mode = cond_mode

        self.cat_embeddings = nn.ModuleList(
            [
                SoftCategoryEmbeddingReg(
                    num_classes=c, embed_dim=cat_embed_dim, reg_weights=reg_weights
                )
                for c in categorical_cardinalities
            ]
        )

        cond_dim = len(categorical_cardinalities) * cat_embed_dim if cond_mode == "concat" else 0
        self.cont_encoder = FlowMatchingContinuousEncoder(
            input_dim=numeric_dim, cond_dim=cond_dim
        )

        in_dim = numeric_dim + cond_dim
        self.out_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, numeric_tensor: torch.Tensor, categorical_idx: List[torch.Tensor]):
        cat_vecs: List[torch.Tensor] = []
        reg_total = numeric_tensor.new_zeros(())

        for emb, idx in zip(self.cat_embeddings, categorical_idx):
            e, reg = emb(idx)
            cat_vecs.append(e)
            reg_total = reg_total + reg

        cat_cond = torch.cat(cat_vecs, dim=-1) if len(cat_vecs) > 0 else None

        fm_loss, x_t, _ = self.cont_encoder(
            numeric_tensor, cond=cat_cond if self.cond_mode == "concat" else None
        )

        feat = torch.cat([x_t, cat_cond], dim=-1) if cat_cond is not None else x_t
        tab_embed = self.out_proj(feat)
        total_loss = fm_loss + reg_total
        return tab_embed, total_loss

    @torch.inference_mode()
    def encode(self, numeric_tensor: torch.Tensor, categorical_idx: List[torch.Tensor]):
        cat_vecs: List[torch.Tensor] = []
        for emb, idx in zip(self.cat_embeddings, categorical_idx):
            e, _ = emb(idx)
            cat_vecs.append(e)
        cat_cond = torch.cat(cat_vecs, dim=-1) if len(cat_vecs) > 0 else None

        x_det = self.cont_encoder.encode_deterministic(
            numeric_tensor, cond=cat_cond if self.cond_mode == "concat" else None
        )

        feat = torch.cat([x_det, cat_cond], dim=-1) if cat_cond is not None else x_det
        tab_embed = self.out_proj(feat)
        return tab_embed


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0).float()  # fp32
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(torch.finfo(sim.dtype).min)

    pos = torch.arange(B, device=z.device)
    labels = torch.cat([pos + B, pos], dim=0)
    return F.cross_entropy(sim, labels)


class TabularEncoder_EmbedMLP(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: List[int],
        cat_embed_dim: int = 16,
        hidden_dims: List[int] = [256, 256],
        out_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.categorical_cardinalities = categorical_cardinalities

        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_embeddings=c, embedding_dim=cat_embed_dim)
            for c in categorical_cardinalities
        ])

        tab_in_dim = numeric_dim + len(categorical_cardinalities) * cat_embed_dim

        layers = []
        prev = tab_in_dim
        for h in hidden_dims:
            layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, h),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers += [
            nn.LayerNorm(prev),
            nn.Linear(prev, out_dim),
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, numeric_tensor: torch.Tensor, categorical_idx: List[torch.Tensor]):
        cat_vecs = [emb(idx) for emb, idx in zip(self.cat_embs, categorical_idx)]
        cat = torch.cat(cat_vecs, dim=-1) if cat_vecs else None
        x = torch.cat([numeric_tensor, cat], dim=-1) if cat is not None else numeric_tensor
        tab_embed = self.mlp(x)
        return tab_embed


def make_tab_view(
    x_num: torch.Tensor,
    cats: List[torch.Tensor],
    cat_cardinalities: List[int],
    num_noise_std: float = 0.01,
    num_mask_prob: float = 0.05,
    cat_replace_prob: float = 0.05,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    device = x_num.device

    x2 = x_num
    if num_noise_std > 0:
        x2 = x2 + torch.randn_like(x2) * num_noise_std
    if num_mask_prob > 0:
        m = (torch.rand_like(x2) < num_mask_prob).to(x2.dtype)
        x2 = x2 * (1.0 - m)

    cats2 = []
    for i, c in enumerate(cats):
        c2 = c.clone()
        if cat_replace_prob > 0 and cat_cardinalities[i] > 1:
            mask = (torch.rand_like(c2.float()) < cat_replace_prob)
            repl = torch.randint(low=0, high=cat_cardinalities[i], size=c2.shape, device=device, dtype=c2.dtype)
            c2[mask] = repl[mask]
        cats2.append(c2)

    return x2, cats2


def prepare_out_dir(cfg: Config) -> str:
    out_dir = os.path.join(cfg.root, cfg.out_name)
    safe_mkdir(out_dir)
    return out_dir


def load_or_compute_global_stats(cfg: Config, out_dir: str, bio: BucketIO) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not cfg.use_global_standardize:
        return None, None

    stats_fp = os.path.join(out_dir, "global_mean_std.json")
    if os.path.exists(stats_fp):
        with open(stats_fp, "r", encoding="utf-8") as f:
            st = json.load(f)
        mean = np.array(st["mean"], dtype=np.float32)
        std = np.array(st["std"], dtype=np.float32)
        print("[STATS] loaded:", stats_fp)
    else:
        print("[STATS] computing global mean/std (streaming fast)...")
        mean, std = bio.streaming_mean_std_fast()
        with open(stats_fp, "w", encoding="utf-8") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        print("[STATS] saved:", stats_fp)

    mean_t = torch.from_numpy(mean).to(cfg.device).view(1, -1)
    std_t = torch.from_numpy(std).to(cfg.device).view(1, -1).clamp_min(1e-6)
    return mean_t, std_t


def build_model(cfg: Config, schema0: dict) -> nn.Module:
    num_dim = len(schema0["num_cols"])
    cat_cards = schema0["cat_cardinalities"]

    if cfg.mode == "flow_old":
        model = TabularEncoder_FlowOld(
            numeric_dim=num_dim,
            categorical_cardinalities=cat_cards,
            cat_embed_dim=cfg.cat_embed_dim,
            reg_weights=cfg.reg_weights if cfg.reg_weights is not None else {"ent": 0.0, "lap": 0.0, "stein": 0.0},
            out_dim=cfg.out_dim,
        )
        return model

    if cfg.mode == "mlp_simclr":
        model = TabularEncoder_EmbedMLP(
            numeric_dim=num_dim,
            categorical_cardinalities=cat_cards,
            cat_embed_dim=cfg.cat_embed_dim,
            hidden_dims=list(cfg.mlp_hidden_dims),
            out_dim=cfg.out_dim,
            dropout=cfg.mlp_dropout,
        )
        return model

    raise ValueError(f"Unknown mode: {cfg.mode}")


def train_one_epoch_flow_old(
    cfg: Config,
    model: TabularEncoder_FlowOld,
    opt: torch.optim.Optimizer,
    bio: BucketIO,
    mean_t: Optional[torch.Tensor],
    std_t: Optional[torch.Tensor],
):
    model.train()
    total = 0.0
    n_batches = 0

    for b in range(cfg.num_buckets):
        if not bio.exists_bucket(b):
            continue

        schema, sample_ids, X_num_np, cat_list_np = bio.load_bucket_arrays(b)
        x = torch.from_numpy(X_num_np).to(cfg.device, dtype=cfg.dtype_num)
        cats = [torch.from_numpy(a).to(cfg.device) for a in cat_list_np]

        x = standardize(x, mean_t, std_t, cfg.use_global_standardize)

        for xb, cb in iter_minibatches(x, cats, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            _, loss = model(xb, cb)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            n_batches += 1

        del schema, sample_ids, X_num_np, cat_list_np, x, cats
        gc.collect()

    return total / max(n_batches, 1)


def train_one_epoch_mlp_simclr(
    cfg: Config,
    model: TabularEncoder_EmbedMLP,
    opt: torch.optim.Optimizer,
    bio: BucketIO,
    mean_t: Optional[torch.Tensor],
    std_t: Optional[torch.Tensor],
    schema0: dict,
):
    model.train()
    total = 0.0
    n_batches = 0

    cat_cards = schema0["cat_cardinalities"]

    for b in range(cfg.num_buckets):
        if not bio.exists_bucket(b):
            continue

        schema, sample_ids, X_num_np, cat_list_np = bio.load_bucket_arrays(b)
        x = torch.from_numpy(X_num_np).to(cfg.device, dtype=cfg.dtype_num)
        cats = [torch.from_numpy(a).to(cfg.device) for a in cat_list_np]

        x = standardize(x, mean_t, std_t, cfg.use_global_standardize)

        for xb, cb in iter_minibatches(x, cats, cfg.batch_size, shuffle=True):
            xb1, cb1 = make_tab_view(
                xb, cb, cat_cards,
                num_noise_std=cfg.num_noise_std,
                num_mask_prob=cfg.num_mask_prob,
                cat_replace_prob=cfg.cat_replace_prob,
            )
            xb2, cb2 = make_tab_view(
                xb, cb, cat_cards,
                num_noise_std=cfg.num_noise_std,
                num_mask_prob=cfg.num_mask_prob,
                cat_replace_prob=cfg.cat_replace_prob,
            )

            opt.zero_grad(set_to_none=True)
            z1 = model(xb1, cb1)
            z2 = model(xb2, cb2)
            loss = nt_xent_loss(z1, z2, temp=cfg.simclr_temp)
            loss.backward()
            opt.step()

            total += float(loss.detach().cpu())
            n_batches += 1

        del schema, sample_ids, X_num_np, cat_list_np, x, cats
        gc.collect()

    return total / max(n_batches, 1)


@torch.inference_mode()
def dump_embeddings(
    cfg: Config,
    model: nn.Module,
    bio: BucketIO,
    out_dir: str,
    mean_t: Optional[torch.Tensor],
    std_t: Optional[torch.Tensor],
    schema0: dict,
):
    model.eval()

    for b in range(cfg.num_buckets):
        if not bio.exists_bucket(b):
            continue

        schema, sample_ids, X_num_np, cat_list_np = bio.load_bucket_arrays(b)
        x = torch.from_numpy(X_num_np).to(cfg.device, dtype=cfg.dtype_num)
        cats = [torch.from_numpy(a).to(cfg.device) for a in cat_list_np]

        x = standardize(x, mean_t, std_t, cfg.use_global_standardize)

        embeds = []
        for xb, cb in iter_minibatches(x, cats, cfg.dump_batch, shuffle=False):
            if cfg.mode == "flow_old":
                eb = model.encode(xb, cb)  # type: ignore[attr-defined]
            else:
                eb = model(xb, cb)
            embeds.append(eb.detach().cpu())

        E = torch.cat(embeds, dim=0).numpy()
        if cfg.save_float16:
            E = E.astype(np.float16)

        out_b = os.path.join(out_dir, f"bucket_{b:03d}")
        safe_mkdir(out_b)

        np.save(os.path.join(out_b, "sample_id.npy"), sample_ids)
        np.save(os.path.join(out_b, "tab_embed.npy"), E)

        with open(os.path.join(out_b, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"bucket": b, "out_dim": int(E.shape[1]), "num_rows": int(E.shape[0])},
                f, ensure_ascii=False, indent=2
            )

        print(f"[DUMP] bucket {b:03d}: {E.shape} -> {out_b}")

        del schema, sample_ids, X_num_np, cat_list_np, x, cats, embeds, E
        gc.collect()


def save_ckpt(cfg: Config, out_dir: str, model: nn.Module, schema0: dict):
    ckpt_name = "tab_encoder_flow_old.pt" if cfg.mode == "flow_old" else "tab_encoder_mlp_simclr.pt"
    ckpt = os.path.join(out_dir, ckpt_name)
    torch.save({"state_dict": model.state_dict(), "schema": schema0, "cfg": cfg.__dict__}, ckpt)
    print("[SAVE] model:", ckpt)


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    tab_tensor_dir = os.path.join(cfg.root, cfg.tab_tensor_dir_name)
    out_dir = prepare_out_dir(cfg)

    bio = BucketIO(tab_tensor_dir=tab_tensor_dir, num_buckets=cfg.num_buckets)
    schema0 = bio.pick_first_schema()

    # stats
    mean_t, std_t = load_or_compute_global_stats(cfg, out_dir, bio)

    # model
    model = build_model(cfg, schema0).to(cfg.device)

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"[CFG] mode={cfg.mode} device={cfg.device} out_dir={out_dir}")
    print(f"[DATA] tab_tensor_dir={tab_tensor_dir} buckets={cfg.num_buckets}")
    print(f"[SCHEMA] num_dim={len(schema0['num_cols'])} cat_cols={len(schema0['cat_cols'])} out_dim={cfg.out_dim}")

    # train
    for ep in range(cfg.epochs):
        if cfg.mode == "flow_old":
            avg = train_one_epoch_flow_old(cfg, model, opt, bio, mean_t, std_t)  # type: ignore[arg-type]
        else:
            avg = train_one_epoch_mlp_simclr(cfg, model, opt, bio, mean_t, std_t, schema0)  # type: ignore[arg-type]
        print(f"[TRAIN] epoch {ep+1}/{cfg.epochs} | avg_loss={avg:.6f}")

    # save ckpt
    save_ckpt(cfg, out_dir, model, schema0)

    # dump embeddings
    dump_embeddings(cfg, model, bio, out_dir, mean_t, std_t, schema0)

    print("[DONE] embeddings saved to:", out_dir)


if __name__ == "__main__":
    main()

