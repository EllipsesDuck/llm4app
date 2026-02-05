import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import h5py

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent)) 

from zero_shot import make   

cxr_filepath = r"E:/NUS/data/perdata/train_text_all_samples/chexpert_test.h5"
model_dir    = r"E:/github_projects/checkpoints/chexzero_weights"

out_dir = Path(r"E:/github_projects/results/predictions")
cache_dir = out_dir / "cached"
out_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

SAVE_NAME = "chexpert_test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_AMP = True
L2_NORMALIZE = True

MAX_MODELS_TO_RUN = 10

model_paths = []
for subdir, _, files in os.walk(model_dir):
    for f in files:
        if f.endswith((".pt", ".pth")):
            model_paths.append(os.path.join(subdir, f))

model_paths = sorted(model_paths)[:MAX_MODELS_TO_RUN]

print(f"Found {len(model_paths)} model(s):")
for p in model_paths:
    print("  ", p)

def run_image_embed_eval(
    model,
    loader,
    device="cuda",
    use_amp=True,
    l2_normalize=True,
):
    model.eval()
    model = model.to(device)

    if device.startswith("cuda"):
        model = model.half()

    embs = []

    def encode_image(m, x):
        if hasattr(m, "encode_image"):
            return m.encode_image(x)
        if hasattr(m, "model") and hasattr(m.model, "encode_image"):
            return m.model.encode_image(x)
        if hasattr(m, "visual"):
            return m.visual(x)
        if hasattr(m, "model") and hasattr(m.model, "visual"):
            return m.model.visual(x)
        raise AttributeError("Cannot find image encoder in model")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            # elif isinstance(batch, dict):
            #     x = batch.get("images", batch.get("image"))
            
            elif isinstance(batch, dict):
                for k in ("images", "image", "cxr", "img", "pixel_values", "x"):
                    if k in batch:
                        x = batch[k]
                        break
                else:
                    raise KeyError(f"Batch dict keys={list(batch.keys())}, cannot find image tensor.")

            else:
                x = batch

            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim == 4 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)

            x = x.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                z = encode_image(model, x)

            if z.ndim > 2:
                z = z.mean(dim=1)

            z = z.float()
            if l2_normalize:
                z = F.normalize(z, dim=-1)

            embs.append(z.cpu().numpy())

    return np.concatenate(embs, axis=0)

def main():
    embeddings_all = []

    for path in model_paths:
        model_name = Path(path).stem
        cache_path = cache_dir / f"{SAVE_NAME}_{model_name}_emb.npy"

        if cache_path.exists():
            print(f"[Cache] Loading {cache_path}")
            emb = np.load(cache_path)
        else:
            print(f"[Run] Extracting embeddings from {model_name}")
            model, loader = make(
                model_path=path,
                cxr_filepath=cxr_filepath,
            )
            emb = run_image_embed_eval(
                model=model,
                loader=loader,
                device=DEVICE,
                use_amp=USE_AMP,
                l2_normalize=L2_NORMALIZE,
            )
            np.save(cache_path, emb)
            print(f"[Saved] {cache_path}")

            del model
            torch.cuda.empty_cache()
            gc.collect()

        embeddings_all.append(emb)

    emb_avg = np.mean(embeddings_all, axis=0)

    out_emb = out_dir / f"{SAVE_NAME}_emb_avg.npy"
    np.save(out_emb, emb_avg)
    print(f"[Saved] {out_emb}, shape={emb_avg.shape}")

    with h5py.File(cxr_filepath, "r") as f:
        sample_id = f["sample_id"][:]

    out_sid = out_dir / f"{SAVE_NAME}_sample_id.npy"
    np.save(out_sid, sample_id)
    print(f"[Saved] {out_sid}, shape={sample_id.shape}")

    print("DONE.")

if __name__ == "__main__":
    main()
