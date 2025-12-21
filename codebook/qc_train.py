import os
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rqvae import RQVAE
from trainer import Trainer  


# =========================
# 1. 随机 Embedding 数据集
# =========================
class FakeEmbDataset(Dataset):
    """
    随机生成 embedding 数据，用来测试训练管线是否能够跑通
    """
    def __init__(self, n_samples=50000, dim=128):
        super().__init__()
        self.n_samples = n_samples
        self.dim = dim
        self.data = np.random.randn(n_samples, dim).astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


# =========================
# 2. 你的 args（保持原结构）
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_step', type=int, default=50)

    # -------- 新增：可手动选择优化器 ----------
    parser.add_argument('--learner', type=str, default="AdamW",
                        choices=["AdamW", "Adam", "SGD", "RMSprop"])

    # -------- 新增：可手动选择 scheduler ----------
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine",
                        choices=["constant", "linear", "cosine"])

    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--dropout_prob", type=float, default=0.0)
    parser.add_argument("--bn", action="store_true")

    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--kmeans_init", type=bool, default=True)
    parser.add_argument("--kmeans_iters", type=int, default=20)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument("--sk_iters", type=int, default=20)

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[64, 64, 64])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument('--layers', type=int, nargs='+', default=[256,128,64])

    parser.add_argument('--save_limit', type=int, default=3)
    parser.add_argument('--ckpt_dir', type=str, default="./ckpt")

    return parser.parse_args()


# =========================
# 3. main 训练入口
# =========================
if __name__ == '__main__':

    # 固定随机 seed
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"

    logging.basicConfig(level=logging.INFO)

    print("============== Args ==============")
    print(args)
    print("==================================")

    # ============== 构造随机数据 ==============
    dataset = FakeEmbDataset(n_samples=20000, dim=128)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ============== 构建 RQVAE 模型 ==============
    model = RQVAE(
        in_dim=128,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model = model.to(args.device)
    print(model)

    # ============== 构建 Trainer ==============
    trainer = Trainer(args, model, len(data_loader))


    # ============== 开始训练 ==============
    best_loss, best_collision = trainer.fit(data_loader)

    print("\n==================== Training Finished ====================")
    print("Best Loss:", best_loss)
    print("Best Collision Rate:", best_collision)
