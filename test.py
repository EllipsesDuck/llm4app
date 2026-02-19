import torch
import torch.nn as nn
from torch.utils.data import Dataset
from minitrainer import MiniTrainer
from generative_rec.flashdecoder.flashdecoder import LazyDecoder
import torch.nn.functional as F
from torch import optim

class FakeLazyDataset(Dataset):
    """
    为 LazyDecoder 构造虚拟数据：
    - target_ids: (B, T_gen)
    - user_static: (B, 1, d_ctx_in)
    - short_term: (B, 5, d_ctx_in)
    - long_term: (B, 10, d_ctx_in)
    """
    def __init__(self, num_samples=200, vocab_size=50, d_ctx_in=256, T_gen=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_ctx_in = d_ctx_in
        self.T_gen = T_gen

        self.data = []
        for _ in range(num_samples):
            tgt = torch.randint(0, vocab_size, (T_gen,), dtype=torch.long)

            user_static = torch.randn(1, d_ctx_in)
            short_term = torch.randn(5, d_ctx_in)
            long_term = torch.randn(10, d_ctx_in)

            self.data.append({
                "target_ids": tgt,
                "user_static": user_static,
                "short_term": short_term,
                "long_term": long_term,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LazyDecoderForTraining(nn.Module):
    """
    包装 LazyDecoder，使其返回 {"loss": ..., "logits": ...}
    """
    def __init__(self, lazy_decoder):
        super().__init__()
        self.model = lazy_decoder

    def forward(self, target_ids, user_static, short_term, long_term):
        out = self.model(
            target_ids=target_ids,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            return_hidden=False
        )
        logits = out["logits"]  # (B,T,vocab)

        # cross entropy：shifted language modeling loss
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B*T, V),
            target_ids.reshape(B*T),
            ignore_index=self.model.pad_id
        )

        return {"loss": loss, "logits": logits}


# ==== 构建 LazyDecoder 基础模型 ====
lazy = LazyDecoder(
    vocab_size=50,
    d_model=256,
    n_layers=4,
    n_heads_q=8,
    gkv=2,
    d_ff=512,
    d_ctx_in=256,
    lkv=1,
    skv=1
)

model = LazyDecoderForTraining(lazy)

# ==== 数据集 ====
train_ds = FakeLazyDataset(num_samples=400)
eval_ds = FakeLazyDataset(num_samples=80)

# ==== 优化器 ====
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = MiniTrainer(
    model=model,
    optimizer=optimizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    batch_size=16,
    num_epochs=3,
    grad_accum_steps=1,
    fp16=False,       # 支持 fp16/bf16
    log_steps=20,
    output_dir="./lazy_ckpt"
)

trainer.train()
