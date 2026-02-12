import os
import json
import argparse
from typing import Dict, Optional, Tuple
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# metrics
try:
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
except Exception:
    roc_auc_score = None
    precision_recall_fscore_support = None


def load_sample_index(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    sid2y: Dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            arr = json.load(f)
            for r in arr:
                sid = str(r.get("sample_id", "")).strip()
                y = r.get("label_multi", None)
                if sid and isinstance(y, list) and len(y) == 8:
                    sid2y[sid] = np.asarray(y, dtype=np.float32)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                sid = str(r.get("sample_id", "")).strip()
                y = r.get("label_multi", None)
                if sid and isinstance(y, list) and len(y) == 8:
                    sid2y[sid] = np.asarray(y, dtype=np.float32)
    return sid2y


def decode_sample_id(row) -> str:
    if row is None:
        return ""
    if isinstance(row, (bytes, np.bytes_)):
        return row.decode("utf-8", errors="ignore").strip()
    if isinstance(row, str):
        return row.strip()
    if isinstance(row, np.ndarray):
        flat = row.reshape(-1)
        for v in flat:
            if isinstance(v, (bytes, np.bytes_)):
                s = v.decode("utf-8", errors="ignore").strip()
            else:
                s = str(v).strip()
            if s and s.lower() not in ("b''", "none"):
                return s
        return ""
    try:
        return str(row).strip()
    except Exception:
        return ""


def normalize_to_float01(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x.astype(np.float32) / 255.0).clip(0.0, 1.0)
    x = x.astype(np.float32, copy=False)
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if xmax <= 1.5 and xmin >= -0.5:
        return np.clip(x, 0.0, 1.0)
    if xmax - xmin < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - xmin) / (xmax - xmin + 1e-6)
    return np.clip(x, 0.0, 1.0)

class H5IndexMultiLabelDataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        idx_path: str,
        sid2y: Dict[str, np.ndarray],
        image_key: str = "images",
        id_key: str = "sample_id",
        resize_to_224: bool = True,
        drop_missing_label: bool = True,
        filter_labels_once: bool = True,
    ):
        self.h5_path = h5_path
        self.image_key = image_key
        self.id_key = id_key
        self.sid2y = sid2y
        self.resize_to_224 = resize_to_224
        self.drop_missing_label = drop_missing_label

        if not os.path.exists(idx_path):
            raise FileNotFoundError(idx_path)
        self.idx = np.load(idx_path).astype(np.int64, copy=False).reshape(-1)

        self._f = None
        self._imgs = None
        self._ids = None

        with h5py.File(self.h5_path, "r") as f:
            assert self.image_key in f, f"Missing {self.image_key} in {self.h5_path}"
            self._H = int(f[self.image_key].shape[1])
            self._W = int(f[self.image_key].shape[2])
            self._has_id = self.id_key in f

        # filter missing labels once
        if filter_labels_once and self.drop_missing_label and self._has_id:
            keep = []
            with h5py.File(self.h5_path, "r") as f:
                ids = f[self.id_key]
                for i in tqdm(self.idx, desc=f"Filter labels ({os.path.basename(idx_path)})", leave=True):
                    sid = decode_sample_id(ids[int(i)])
                    if sid in self.sid2y:
                        keep.append(int(i))
            self.idx = np.asarray(keep, dtype=np.int64)
            if self.idx.size == 0:
                raise RuntimeError(f"After filtering, no samples left for {idx_path}")

    def __len__(self):
        return int(self.idx.shape[0])

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
            self._imgs = self._f[self.image_key]
            self._ids = self._f[self.id_key] if self._has_id else None

    def __getitem__(self, j: int):
        self._ensure_open()
        i = int(self.idx[j])

        x = np.asarray(self._imgs[i])
        x = normalize_to_float01(x)
        xt = torch.from_numpy(x).unsqueeze(0)

        if self.resize_to_224 and (self._H != 224 or self._W != 224):
            xt = TF.resize(xt, [224, 224], antialias=True)

        xt = xt.repeat(3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=xt.dtype)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=xt.dtype)[:, None, None]
        xt = (xt - mean) / std

        sid = decode_sample_id(self._ids[i]) if (self._ids is not None) else str(i)
        y = self.sid2y.get(sid)
        if y is None:
            y = np.zeros((8,), dtype=np.float32) if self.drop_missing_label else np.full((8,), -1.0, dtype=np.float32)

        return xt, torch.from_numpy(y).float(), sid

class MultiLabelBackbone(nn.Module):
    def __init__(self, backbone: str, pretrained: bool, num_labels: int = 8):
        super().__init__()
        bb = backbone.lower().strip()

        if bb == "densenet121":
            weights = torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None
            m = torchvision.models.densenet121(weights=weights)
            in_dim = m.classifier.in_features
            m.classifier = nn.Identity()
            self.encoder = m
            self.head = nn.Linear(in_dim, num_labels)

        elif bb == "resnet50":
            weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
            m = torchvision.models.resnet50(weights=weights)
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.encoder = m
            self.head = nn.Linear(in_dim, num_labels)

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

def _safe_auc(y_true_1d: np.ndarray, y_prob_1d: np.ndarray) -> Optional[float]:
    if y_true_1d.size == 0:
        return None
    if y_true_1d.max() == y_true_1d.min():
        return None
    try:
        return float(roc_auc_score(y_true_1d, y_prob_1d))
    except Exception:
        return None


def compute_auc_stats(y_true_bin: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if roc_auc_score is None:
        out["macro_auc"] = float("nan")
        out["micro_auc"] = float("nan")
        out["auc_valid_classes"] = 0.0
        for c in range(y_true_bin.shape[1]):
            out[f"auc_c{c}"] = float("nan")
        return out

    aucs = []
    valid = 0
    for c in range(y_true_bin.shape[1]):
        a = _safe_auc(y_true_bin[:, c], y_prob[:, c])
        out[f"auc_c{c}"] = float("nan") if a is None else a
        if a is not None:
            aucs.append(a)
            valid += 1
    out["auc_valid_classes"] = float(valid)
    out["macro_auc"] = float(np.mean(aucs)) if len(aucs) else float("nan")

    try:
        yt = y_true_bin.reshape(-1)
        yp = y_prob.reshape(-1)
        if yt.max() == yt.min():
            out["micro_auc"] = float("nan")
        else:
            out["micro_auc"] = float(roc_auc_score(yt, yp))
    except Exception:
        out["micro_auc"] = float("nan")
    return out


def compute_prf_stats(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if precision_recall_fscore_support is None:
        for k in ["macro_p", "macro_r", "macro_f1", "micro_p", "micro_r", "micro_f1"]:
            out[k] = float("nan")
        return out

    y_true_bin = (y_true_bin > 0).astype(np.int32)
    y_pred_bin = (y_pred_bin > 0).astype(np.int32)

    p, r, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    out["macro_p"] = float(p)
    out["macro_r"] = float(r)
    out["macro_f1"] = float(f1)

    p, r, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="micro", zero_division=0)
    out["micro_p"] = float(p)
    out["micro_r"] = float(r)
    out["micro_f1"] = float(f1)
    return out


@torch.no_grad()
def collect_probs(model, dl, device, name: str) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_true = []
    all_prob = []

    for x, y, _sid in tqdm(dl, desc=name, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        all_true.append(y.detach().cpu().numpy())
        all_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_true = np.concatenate(all_true, axis=0).astype(np.float32) if all_true else np.zeros((0, 8), dtype=np.float32)
    y_prob = np.concatenate(all_prob, axis=0).astype(np.float32) if all_prob else np.zeros((0, 8), dtype=np.float32)
    y_true_bin = (y_true > 0).astype(np.int32)
    avg_loss = total_loss / max(1, n)
    return avg_loss, y_true_bin, y_prob


def search_best_threshold_macro_f1(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    thr_min: float = 0.05,
    thr_max: float = 0.95,
    thr_step: float = 0.05,
) -> Tuple[float, float]:
    if precision_recall_fscore_support is None:
        return 0.5, float("nan")

    best_thr = 0.5
    best_f1 = -1.0
    thr = thr_min
    while thr <= thr_max + 1e-9:
        y_pred = (y_prob >= float(thr)).astype(np.int32)
        stats = compute_prf_stats(y_true_bin, y_pred)
        f1 = float(stats.get("macro_f1", float("nan")))
        if np.isfinite(f1) and f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
        thr += thr_step
    return best_thr, best_f1


def eval_split(model, dl, device, name: str, thr: float) -> Dict[str, float]:
    loss, y_true_bin, y_prob = collect_probs(model, dl, device, name=name)
    y_pred = (y_prob >= float(thr)).astype(np.int32)

    out: Dict[str, float] = {"loss": float(loss), "thr_used": float(thr)}
    out.update(compute_auc_stats(y_true_bin, y_prob))
    out.update(compute_prf_stats(y_true_bin, y_pred))
    return out


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, type=str)
    ap.add_argument("--split_index_dir", required=True, type=str)
    ap.add_argument("--index_json", required=True, type=str)

    ap.add_argument("--image_key", default="images", type=str)
    ap.add_argument("--id_key", default="sample_id", type=str)

    ap.add_argument("--backbone", required=True, choices=["densenet121", "resnet50"])
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--epochs", default=5, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--wd", default=1e-4, type=float)
    ap.add_argument("--num_workers", default=0, type=int)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--grad_clip", default=1.0, type=float)

    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--out_ckpt", required=True, type=str)

    # selection on TEST (NEW)
    ap.add_argument(
        "--select_by",
        default="macro_auc",
        choices=["macro_auc", "micro_auc", "macro_f1", "loss"],
        help="criterion to select best model on TEST split",
    )

    # threshold policy (on TEST)
    ap.add_argument("--thr", default=0.5, type=float, help="fixed threshold (used when --auto_thr not set)")
    ap.add_argument("--auto_thr", action="store_true", help="auto search threshold on TEST to maximize macro F1")
    ap.add_argument("--thr_min", default=0.05, type=float)
    ap.add_argument("--thr_max", default=0.95, type=float)
    ap.add_argument("--thr_step", default=0.05, type=float)

    # tensorboard
    ap.add_argument("--log_dir", default="", type=str)
    ap.add_argument("--exp_name", default="", type=str)
    ap.add_argument("--run_id", default="", type=str)
    ap.add_argument("--log_every", default=1, type=int, help="log train loss every N steps")
    args = ap.parse_args()

    if (roc_auc_score is None) or (precision_recall_fscore_support is None):
        print("[Warn] sklearn not found. Install: pip install scikit-learn (metrics -> NaN)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.fp16 and (device == "cuda")
    print(f"[Info] device={device}, amp={use_amp}")

    sid2y = load_sample_index(args.index_json)
    print(f"[Info] loaded labels: {len(sid2y)} from {args.index_json}")

    tr_idx = os.path.join(args.split_index_dir, "train_idx.npy")
    te_idx = os.path.join(args.split_index_dir, "test_idx.npy")
    ev_idx = os.path.join(args.split_index_dir, "eval_idx.npy")

    ds_tr = H5IndexMultiLabelDataset(args.h5, tr_idx, sid2y, args.image_key, args.id_key, True, True, True)
    ds_te = H5IndexMultiLabelDataset(args.h5, te_idx, sid2y, args.image_key, args.id_key, True, True, True)
    ds_ev = H5IndexMultiLabelDataset(args.h5, ev_idx, sid2y, args.image_key, args.id_key, True, True, True)
    print(f"[Info] sizes after label-filter: train={len(ds_tr)} test={len(ds_te)} eval={len(ds_ev)}")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)
    dl_ev = DataLoader(ds_ev, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, drop_last=False)

    model = MultiLabelBackbone(args.backbone, pretrained=args.pretrained, num_labels=8).to(device)

    if args.freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[Info] freeze_backbone=True (train head only)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # tb dirs
    ckpt_dir = os.path.dirname(args.out_ckpt) if os.path.dirname(args.out_ckpt) else "."
    os.makedirs(ckpt_dir, exist_ok=True)

    exp = args.exp_name.strip()
    if not exp:
        exp = f"{args.backbone}_pre{int(args.pretrained)}_freeze{int(args.freeze_backbone)}_bs{args.batch_size}_lr{args.lr}_wd{args.wd}_fp16{int(args.fp16)}_sel{args.select_by}"
    run_id = args.run_id.strip() if args.run_id.strip() else datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.log_dir.strip():
        log_dir = args.log_dir.strip()
    else:
        log_dir = os.path.join(ckpt_dir, "tb", exp, run_id)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[TB ] log_dir: {log_dir}")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("hparams", f"exp={exp}, run_id={run_id}\nargs={vars(args)}")

    # selection init
    if args.select_by == "loss":
        best_key = float("inf")
    else:
        best_key = -1e9
    best_ep = 0
    best_thr = float(args.thr)
    best_test_snapshot = {}
    global_step = 0

    def _get_key(te: Dict[str, float]) -> float:
        if args.select_by == "loss":
            return float(te["loss"])
        if args.select_by == "macro_auc":
            return float(te["macro_auc"])
        if args.select_by == "micro_auc":
            return float(te["micro_auc"])
        if args.select_by == "macro_f1":
            return float(te["macro_f1"])
        return float(te["macro_auc"])

    def _is_improved(key_now: float) -> bool:
        nonlocal best_key
        if args.select_by == "loss":
            return key_now < best_key
        return key_now > best_key

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    for ep in tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True):
        model.train()
        total = 0.0
        n = 0

        pbar = tqdm(dl_tr, desc=f"train ep{ep:02d}", leave=False)
        for x, y, _sid in pbar:
            global_step += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
            else:
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt.step()

            loss_val = float(loss.item())
            total += loss_val * x.size(0)
            n += x.size(0)
            pbar.set_postfix(loss=loss_val)

            if (global_step % max(1, args.log_every)) == 0:
                writer.add_scalar("train/loss_step", loss_val, global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

        tr_loss = total / max(1, n)
        writer.add_scalar("loss/train_epoch", tr_loss, global_step)

        test_loss, y_true_bin_te, y_prob_te = collect_probs(model, dl_te, device, name="test")

        # threshold on TEST (optional)
        if args.auto_thr:
            thr_now, thr_bestf1 = search_best_threshold_macro_f1(
                y_true_bin_te, y_prob_te,
                thr_min=args.thr_min, thr_max=args.thr_max, thr_step=args.thr_step
            )
        else:
            thr_now = float(args.thr)
            thr_bestf1 = float("nan")

        te = {"loss": float(test_loss)}
        te.update(compute_auc_stats(y_true_bin_te, y_prob_te))
        y_pred_te = (y_prob_te >= float(thr_now)).astype(np.int32)
        te.update(compute_prf_stats(y_true_bin_te, y_pred_te))
        te["thr_used"] = float(thr_now)
        te["thr_best_macro_f1"] = float(thr_bestf1)

        # tb (epoch-level)
        writer.add_scalar("loss/test", te["loss"], global_step)
        writer.add_scalar("auc/macro_test", te["macro_auc"], global_step)
        writer.add_scalar("auc/micro_test", te["micro_auc"], global_step)
        writer.add_scalar("f1/macro_test", te["macro_f1"], global_step)
        writer.add_scalar("f1/micro_test", te["micro_f1"], global_step)
        writer.add_scalar("thr/test_used", te["thr_used"], global_step)
        if np.isfinite(te["thr_best_macro_f1"]):
            writer.add_scalar("thr/test_best_macro_f1", te["thr_best_macro_f1"], global_step)
        if "auc_c0" in te:
            for c in range(8):
                writer.add_scalar(f"auc_class/test_c{c}", te.get(f"auc_c{c}", float("nan")), global_step)

        key_now = _get_key(te)
        improved = _is_improved(key_now)

        print(
            f"[Epoch {ep:02d}] "
            f"train_loss={tr_loss:.4f} | "
            f"test_loss={te['loss']:.4f} test_macroAUC={te['macro_auc']:.4f} test_microAUC={te['micro_auc']:.4f} "
            f"test_macroF1={te['macro_f1']:.4f} test_microF1={te['micro_f1']:.4f} thr={te['thr_used']:.3f} "
            f"| select_by={args.select_by} key_now={key_now:.4f} best_key={best_key:.4f}"
        )

        # save best by TEST criterion
        if improved:
            best_key = float(key_now)
            best_ep = int(ep)
            best_thr = float(thr_now)
            best_test_snapshot = dict(te)

            torch.save(
                {
                    "backbone": args.backbone,
                    "pretrained": bool(args.pretrained),
                    "freeze_backbone": bool(args.freeze_backbone),
                    "state_dict": model.state_dict(),
                    "best_ep": int(best_ep),
                    "best_thr_on_test": float(best_thr),
                    "select_by": args.select_by,
                    "best_key": float(best_key),
                    "best_test": best_test_snapshot,
                    "log_dir": log_dir,
                    "exp": exp,
                    "run_id": run_id,
                    "args": vars(args),
                },
                args.out_ckpt,
            )
            print(
                f"[Save] best ckpt -> {args.out_ckpt} "
                f"(ep={best_ep:02d}, select_by={args.select_by}, best_key={best_key:.4f}, thr={best_thr:.3f})"
            )

        writer.flush()

    writer.close()

    if not os.path.exists(args.out_ckpt):
        print("[Final] no ckpt saved (unexpected). Done.")
        return

    print("\n[Final] load BEST ckpt and evaluate on EVAL split ONCE ...")
    ckpt = torch.load(args.out_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    thr_final = float(ckpt.get("best_thr_on_test", args.thr))

    ev = eval_split(model, dl_ev, device, name="eval", thr=thr_final)

    # print final summary (what you care about)
    bt = ckpt.get("best_test", {})
    print(
        f"[BEST by TEST] ep={int(ckpt.get('best_ep', -1))} select_by={ckpt.get('select_by','?')} best_key={float(ckpt.get('best_key', float('nan'))):.4f} "
        f"thr={thr_final:.3f}\n"
        f"  TEST: loss={float(bt.get('loss', float('nan'))):.4f} macroAUC={float(bt.get('macro_auc', float('nan'))):.4f} microAUC={float(bt.get('micro_auc', float('nan'))):.4f} "
        f"macroF1={float(bt.get('macro_f1', float('nan'))):.4f} microF1={float(bt.get('micro_f1', float('nan'))):.4f}\n"
        f"  EVAL: loss={ev['loss']:.4f} macroAUC={ev['macro_auc']:.4f} microAUC={ev['micro_auc']:.4f} "
        f"macroF1={ev['macro_f1']:.4f} microF1={ev['micro_f1']:.4f}"
    )

    print("[Done]")
    print(f"[TB ] for THIS run: tensorboard --logdir \"{log_dir}\"")
    print(f"[TB ] compare MANY runs under ckpt_dir: tensorboard --logdir \"{os.path.join(ckpt_dir, 'tb')}\"")


if __name__ == "__main__":
    train()