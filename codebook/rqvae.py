import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLPLayers
from rq import ResidualVectorQuantizer


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
        assert len(num_emb_list) == len(sk_epsilons), \
            "num_emb_list and sk_epsilons length must same"

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
        self.encoder = MLPLayers(
            layers=self.encode_layer_dims,
            dropout=self.dropout_prob,
            use_bn=self.bn
        )

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
        self.decoder = MLPLayers(
            layers=self.decode_layer_dims,
            dropout=self.dropout_prob,
            use_bn=self.bn
        )

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

    def reconstruct(self, xs, use_sk=False):
        with torch.no_grad():
            x_e = self.encoder(xs)
            x_q, _, _ = self.rq(x_e, use_sk=use_sk)
            return self.decoder(x_q)

    def compute_loss(self, out, quant_loss, xs):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, loss_recon


import torch
from torch import nn
from torch.nn import functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ============================================================
# 1. 伪造一批数据
# ============================================================

# 假设输入维度是 768（像 BERT embedding）
B = 16               # batch
in_dim = 768
xs = torch.randn(B, in_dim, device=device)

print("\n====== Fake dataset ======")
print("xs shape:", xs.shape)


# ============================================================
# 2. 定义一个简单的 RQVAE 结构
# ============================================================

model = RQVAE(
    in_dim=in_dim,
    num_emb_list=[256, 256, 256],   # 三层 RVQ
    e_dim=64,                       # codebook embedding dim
    layers=[512, 256, 128],         # encoder layers
    dropout_prob=0.0,
    bn=True,
    loss_type="mse",
    quant_loss_weight=1.0,
    beta=0.25,
    kmeans_init=False,              # 测试随机 init
    kmeans_iters=10,
    sk_epsilons=[0.0, 0.003, 0.01], # 每层 Sinkhorn epsilon
    sk_iters=50
).to(device)

print("\n====== Build RQVAE ======")
print(model)


# ============================================================
# 3. Forward 通过模型
# ============================================================

print("\n====== Forward ======")

out, rq_loss, indices = model(xs, use_sk=True)

print("Output shape:", out.shape)
print("RQ loss:", rq_loss.item())
print("Indices shape:", indices.shape)     # (B, n_layers)
print("Indices example:", indices[0])


# ============================================================
# 4. 计算完整 loss（重构 + 量化）
# ============================================================

print("\n====== Compute Loss ======")

loss_total, loss_recon = model.compute_loss(out, rq_loss, xs)

print("Reconstruction loss:", loss_recon.item())
print("Total loss:", loss_total.item())


# ============================================================
# 5. 测试 get_indices（一般用于 inference）
# ============================================================

print("\n====== Test get_indices() ======")

with torch.no_grad():
    test_indices = model.get_indices(xs, use_sk=True)

print("test_indices shape:", test_indices.shape)
print("test_indices[0]:", test_indices[0])


# ============================================================
# 6. 测试 reconstruct（重建）
# ============================================================

print("\n====== Test reconstruct() ======")

with torch.no_grad():
    recon = model.reconstruct(xs)

print("reconstructed shape:", recon.shape)
print("reconstruct mse:", F.mse_loss(recon, xs).item())

print("\n====== TEST DONE ======")
