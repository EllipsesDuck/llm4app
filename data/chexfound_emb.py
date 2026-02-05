import os
import json
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from chexfound.eval.setup import setup_and_build_model
from chexfound.data.transforms import make_classification_eval_transform


def load_backbone_ckpt(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["teacher"] if isinstance(ckpt, dict) and "teacher" in ckpt else ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone"):
            ls = k.split(".")
            if "blocks" in k:
                new_k = ".".join([ls[1], *ls[3:]])
            else:
                new_k = ".".join(ls[1:])
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[WARN] Unexpected keys (showing up to 20): {unexpected[:20]}")


@torch.no_grad()
def extract_embedding_from_features(features, use_cls: bool = True, layer_reduce: str = "last"):
    """
    features: output of model.get_intermediate_layers(...)
    Return: (B, D) L2-normalized embedding
    """
    if isinstance(features, (torch.Tensor, tuple)):
        layer_list = [features]
    else:
        layer_list = list(features)

    vecs = []
    for layer_out in layer_list:
        if isinstance(layer_out, (tuple, list)) and len(layer_out) == 2:
            patch_tok, cls_tok = layer_out
            if cls_tok.ndim == 3:
                cls_tok = cls_tok[:, 0, :]
            v = cls_tok if use_cls else patch_tok.mean(dim=1)
        else:
            x = layer_out  # (B, N, C)
            if use_cls:
                v = x[:, 0, :]
            else:
                v = x[:, 1:, :].mean(dim=1)
        vecs.append(v)

    if layer_reduce == "last":
        emb = vecs[-1]
    elif layer_reduce == "mean":
        emb = torch.stack(vecs, dim=0).mean(dim=0)
    elif layer_reduce == "concat":
        emb = torch.cat(vecs, dim=-1)
    else:
        raise ValueError("layer_reduce must be one of {'last','mean','concat'}")

    emb = F.normalize(emb, p=2, dim=-1)
    return emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/fast/yangz16/CheXFound")

    parser.add_argument("--h5_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--pretrained_weights", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)  
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--n_last_blocks", type=int, default=4)
    parser.add_argument("--return_class_token", action="store_true", default=True)

    parser.add_argument("--use_cls", action="store_true", default=True)
    parser.add_argument("--layer_reduce", type=str, default="last", choices=["last", "mean", "concat"])

    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--fp16_save", action="store_true", default=False)

    parser.add_argument("--limit", type=int, default=0, help="0 means no limit")
    args, _ = parser.parse_known_args()

    os.chdir(args.repo_root)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print(f"[INFO] device = {device}")

    # CheXFound setup expects some fields
    args.opts = []
    args.n_register_tokens = 4
    args.num_classes = 40
    args.num_heads = 8

    # Build model
    model, autocast_dtype = setup_and_build_model(args)
    model.eval().to(device)
    load_backbone_ckpt(model, args.pretrained_weights)

    # Transform (expects PIL-like input; we will provide numpy but apply similarly)
    # make_classification_eval_transform returns a callable that typically accepts PIL.Image
    transform = make_classification_eval_transform(
        resize_size=args.image_size,
        crop_size=args.image_size
    )

    # Open H5
    h5_path = args.h5_path
    with h5py.File(h5_path, "r") as f:
        imgs = f["images"]       # (N,224,224) float32
        sids = f["sample_id"]    # (N,) bytes
        N = imgs.shape[0]
        if args.limit and args.limit > 0:
            N = min(N, args.limit)

        print(f"[INFO] H5 images shape: {imgs.shape}, dtype={imgs.dtype}")
        print(f"[INFO] Using first N={N}")

        # Determine embedding dim by a dummy forward
        # build one sample: (1,3,H,W)
        x0 = imgs[0]  # (224,224) float32
        # ensure 0..1 if your stored range is 0..1; if it's 0..255, uncomment normalization below
        # x0 = x0 / 255.0

        # make 3-channel uint8-like for transform pipeline (but keep float)
        x0_3 = np.stack([x0, x0, x0], axis=-1)  # (224,224,3)

        # transform expects PIL Image; create PIL safely
        # If x0 is 0..1 float, convert to uint8 first to avoid PIL mode issues
        x0_u8 = np.clip(x0_3 * 255.0, 0, 255).astype(np.uint8)
        from PIL import Image
        pil0 = Image.fromarray(x0_u8, mode="RGB")
        tx0 = transform(pil0)  # torch tensor (3, S, S)
        tx0 = tx0.unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.startswith("cuda") and args.amp)):
                feats0 = model.get_intermediate_layers(tx0, n=args.n_last_blocks, return_class_token=args.return_class_token)
                emb0 = extract_embedding_from_features(feats0, use_cls=args.use_cls, layer_reduce=args.layer_reduce)

        D = emb0.shape[-1]
        print(f"[INFO] embedding dim = {D}")

        # Pre-allocate outputs 
        out_dtype = np.float16 if args.fp16_save else np.float32
        emb_path = os.path.join(args.output_dir, "embeddings.npy")
        sid_path = os.path.join(args.output_dir, "sample_id.npy")

        embs_mm = np.memmap(emb_path, mode="w+", dtype=out_dtype, shape=(N, D))
        sid_out = np.empty((N,), dtype="S16")

        # Iterate in batches
        bs = args.batch_size
        for start in range(0, N, bs):
            end = min(N, start + bs)

            batch = imgs[start:end]  # (B,224,224) float32
            # batch = batch / 255.0

            # Convert to RGB for transform; transform operates per image
            tx_list = []
            sid_batch = sids[start:end]

            for i in range(end - start):
                x = batch[i]
                x3 = np.stack([x, x, x], axis=-1)
                x_u8 = np.clip(x3 * 255.0, 0, 255).astype(np.uint8)
                pil = Image.fromarray(x_u8, mode="RGB")
                tx = transform(pil)   # (3,S,S)
                tx_list.append(tx)

            x_tensor = torch.stack(tx_list, dim=0).to(device)  # (B,3,S,S)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.startswith("cuda") and args.amp)):
                    feats = model.get_intermediate_layers(
                        x_tensor,
                        n=args.n_last_blocks,
                        return_class_token=args.return_class_token,
                    )
                    emb = extract_embedding_from_features(
                        feats,
                        use_cls=args.use_cls,
                        layer_reduce=args.layer_reduce,
                    )  # (B,D)

            emb_np = emb.detach().cpu().numpy().astype(out_dtype, copy=False)
            embs_mm[start:end] = emb_np
            sid_out[start:end] = sid_batch

            if (start // bs) % 20 == 0:
                print(f"[INFO] {end}/{N} done")

        # Flush memmap
        embs_mm.flush()
        np.save(sid_path, sid_out)

    meta = {
        "h5_path": h5_path,
        "N": int(N),
        "D": int(D),
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "n_last_blocks": args.n_last_blocks,
        "use_cls": bool(args.use_cls),
        "layer_reduce": args.layer_reduce,
        "amp": bool(args.amp),
        "fp16_save": bool(args.fp16_save),
        "l2_normalized": True,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] embeddings saved to: {emb_path}")
    print(f"[DONE] sample_id saved to: {sid_path}")
    print(f"[DONE] meta saved to: {os.path.join(args.output_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
