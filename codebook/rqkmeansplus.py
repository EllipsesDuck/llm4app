import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from rqvae import RQVAE


class ResidualEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # 判断倒数第二层和最后一层是否能 residual
        layers = list(self.encoder.mlp)
        
        # 找到最后两个 Linear
        self.last_linear = None
        self.pre_last_linear = None
        for m in reversed(layers):
            if isinstance(m, nn.Linear):
                if self.last_linear is None:
                    self.last_linear = m
                elif self.pre_last_linear is None:
                    self.pre_last_linear = m
                    break

        # 只有当这两层输出维度一致才允许 residual
        if self.last_linear.in_features == self.last_linear.out_features:
            self.can_residual = True
        else:
            self.can_residual = False

    @property
    def mlp(self):
        """Forwarding mlp attribute"""
        return self.encoder.mlp

    def forward(self, x):
        out = self.encoder(x)
        if self.can_residual:
            return out + x[:, :out.size(1)]  # 对齐前 out 的维度
        else:
            return out



def _find_last_linear(module: nn.Module):
    """Find the last nn.Linear inside an MLP (robust version)"""
    last_linear = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    return last_linear


def apply_rqkmeans_plus_strategy(model, codebook_path, device):
    logging.info("\n>>> [RQ-KMeans+] Applying Strategy: Residual Wrapper + Warm Start + Zero Init")

    # ----------------------------------------------------------------------
    # (1) Wrap encoder with residual connection
    # ----------------------------------------------------------------------
    if hasattr(model, "encoder"):
        original_encoder = model.encoder
        model.encoder = ResidualEncoderWrapper(original_encoder).to(device)
        logging.info("    ✓ Encoder wrapped → Z = X + MLP(X)")
    else:
        logging.error("    ✗ Model missing attribute `encoder`")
        return model

    # ----------------------------------------------------------------------
    # (2) Zero-init last Linear layer inside the encoder
    # ----------------------------------------------------------------------
    mlp = model.encoder.mlp
    last_linear = _find_last_linear(mlp)

    if last_linear is None:
        logging.warning("    ⚠ No Linear layer found inside encoder → Skip zero-init")
    else:
        with torch.no_grad():
            last_linear.weight.zero_()
            if last_linear.bias is not None:
                last_linear.bias.zero_()
        logging.info(f"    ✓ Zero-initialized last Linear layer: {last_linear}")

    # ----------------------------------------------------------------------
    # (3) Load codebook npz file
    # ----------------------------------------------------------------------
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"[ERROR] Codebook file not found: {codebook_path}")

    logging.info(f"    ✓ Loading codebooks from: {codebook_path}")
    npz_data = np.load(codebook_path)

    # ----------------------------------------------------------------------
    # (4) Locate RVQ / VQ layers
    # ----------------------------------------------------------------------
    if not hasattr(model, "rq") or not hasattr(model.rq, "vq_layers"):
        logging.error("    ✗ Cannot find model.rq.vq_layers → Skip codebook init")
        return model

    vq_layers = model.rq.vq_layers
    num_levels = len(vq_layers)

    success_count = 0

    for level_idx in range(num_levels):

        # Get the layer
        layer = vq_layers[level_idx]
        emb_layer = layer.embedding if hasattr(layer, "embedding") else layer

        cb_key = f"codebook_{level_idx}"

        if cb_key not in npz_data:
            logging.warning(f"    ⚠ Missing key `{cb_key}` in npz file → skip this level")
            continue

        centroids = npz_data[cb_key]  # (K, D)
        centroids_torch = torch.tensor(centroids, dtype=emb_layer.weight.dtype, device=device)

        # shape check
        if emb_layer.weight.shape != centroids_torch.shape:
            logging.error(
                f"    ✗ Shape mismatch at level {level_idx}: "
                f"embedding={tuple(emb_layer.weight.shape)} vs codebook={tuple(centroids_torch.shape)}"
            )
            continue

        # load
        with torch.no_grad():
            emb_layer.weight.copy_(centroids_torch)

        logging.info(f"    ✓ Loaded codebook for level {level_idx}: shape={centroids_torch.shape}")
        success_count += 1

    # ----------------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------------
    if success_count == 0:
        logging.warning("    ⚠ No codebooks were loaded. Check your .npz keys.")

    logging.info(f"    → Finished RQ-KMeans+ Warm-Start: loaded {success_count}/{num_levels} levels\n")

    return model


import torch
import numpy as np
import os


# --------------------------------------------------
# 1. 随机生成一些 Fake 数据
# --------------------------------------------------
BATCH = 8
DIM = 64   # 输入维度
x = torch.randn(BATCH, DIM)

# --------------------------------------------------
# 2. 构造一个假的 RQ-KMeans codebook 作为 warm-start
# --------------------------------------------------
num_levels = 3      # 量化层数
emb_dim = 32        # 每层 embedding 维度
emb_per_level = 16  # 每层 codebook 中有多少个 centroid

codebooks = {}
for i in range(num_levels):
    codebooks[f"codebook_{i}"] = np.random.randn(emb_per_level, emb_dim).astype(np.float32)

np.savez("fake_codebook.npz", **codebooks)
print("✔ Fake codebook saved as fake_codebook.npz")

# --------------------------------------------------
# 3. 构造一个 RQVAE 模型
# --------------------------------------------------
model = RQVAE(
    in_dim=DIM,
    num_emb_list=[emb_per_level] * num_levels,  # e.g. [16,16,16]
    e_dim=emb_dim,           # embedding dim
    layers=[128, 128],       # encoder/decoder MLP 层
    dropout_prob=0.0,
    bn=False,
    loss_type="l2",
    quant_loss_weight=1.0,
    beta=0.25,
    kmeans_init=False,
    kmeans_iters=10,
    sk_epsilons=[0.1, 0.1, 0.1],
    sk_iters=5,
)

device = torch.device("cpu")
model = model.to(device)
x = x.to(device)

print("✔ RQVAE initialized")

# --------------------------------------------------
# 4. 应用你写的 RQ-KMeans+ Warm-Start 策略
# --------------------------------------------------
model = apply_rqkmeans_plus_strategy(
    model,
    "fake_codebook.npz",
    device
)

print("✔ Warm-start applied")

# --------------------------------------------------
# 5. 前向推理一次
# --------------------------------------------------
model.eval()
with torch.no_grad():
    out, quant_loss, indices = model(x)

print("\n================= TEST RESULT =================")
print("Input shape:", x.shape)
print("Output shape:", out.shape)
print("RQ Loss:", quant_loss.item())
print("Indices shape:", indices.shape)
print("Indices example:", indices[0])
print("================================================")

