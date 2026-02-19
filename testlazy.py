import torch
from torch.utils.data import DataLoader
import random

from generative_rec.flashdecoder.flashdecoder import LazyDecoder, GBPOTrainer,HierarchicalSemanticIDProcessor
from logitprocessor import PrefixTrie

# ===============================
# 1. 构造 semantic id + Trie
# ===============================

semantic_ids = [
    [10, 21, 33],
    [10, 21, 34],
    [10, 22, 40],
    [11, 30, 55],
    [12, 31, 60],
]

trie = PrefixTrie()
for sid in semantic_ids:
    trie.insert(sid)

semantic_processor = HierarchicalSemanticIDProcessor(
    trie=trie,
    bos_id=1,
    eos_id=None,                 # 先不强制 EOS
    force_eos_on_leaf=False,
    allow_invalid_prefix=True,   # 虚拟数据阶段更稳
)

# ===============================
# 2. Dummy Dataset（你原样）
# ===============================

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, d_ctx_in, seq_len=4):
        self.vocab_size = vocab_size
        self.d_ctx_in = d_ctx_in
        self.seq_len = seq_len

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        T = self.seq_len

        target_ids = torch.randint(
            low=2,
            high=self.vocab_size,
            size=(T,),
            dtype=torch.long
        )
        target_ids[0] = 1  # BOS

        return {
            "target_ids": target_ids,
            "user_static": torch.randn(1, 1, self.d_ctx_in),
            "short_term":  torch.randn(1, 2, self.d_ctx_in),
            "long_term":   torch.randn(1, 2, self.d_ctx_in),
            "rewards":     torch.randn(1),
        }

def collate_fn(batch):
    return {
        "target_ids": torch.stack([b["target_ids"] for b in batch], dim=0),
        "user_static": torch.cat([b["user_static"] for b in batch], dim=0),
        "short_term":  torch.cat([b["short_term"]  for b in batch], dim=0),
        "long_term":   torch.cat([b["long_term"]   for b in batch], dim=0),
        "rewards": torch.cat([b["rewards"] for b in batch], dim=0),
    }

# ===============================
# 3. 通用配置
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 100
d_ctx_in = 256
batch_size = 16
sync_interval = 50
max_steps = 200

dataset = DummyDataset(
    vocab_size=vocab_size,
    d_ctx_in=d_ctx_in,
    seq_len=4,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

# ===============================
# 4. 运行一个实验的函数
# ===============================

def run_experiment(use_trie: bool):
    print("\n" + "=" * 80)
    print(f"🚀 Running experiment | use_trie = {use_trie}")
    print("=" * 80)

    # ---- model ----
    model = LazyDecoder(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=4,
        moe_layers=2,
        n_heads_q=4,
        gkv=2,
        d_ff=256,
        d_ctx_in=256,
        pad_id=0,
        bos_id=1,
    ).to(device)

    # ---- trainer ----
    trainer = GBPOTrainer(
        model=model,
        lambda_rl=0.1,
        gbpo_level="sequence",
        gbpo_use_clip=False,
        pad_id=0,
        bos_id=1,
        device=device,
        semantic_processor=semantic_processor if use_trie else None,
        use_auto_prefix=True,
        use_hybrid_prefix=True,
        auto_prefix_mode='cosine'
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # ---- training loop ----
    for step, batch in enumerate(loader):
        metrics = trainer.train_step(
            batch=batch,
            optimizer=optimizer,
            use_rl=True,
        )

        if step % sync_interval == 0:
            trainer.sync_old_policy()

        if step % 10 == 0:
            print(
                f"[step {step:04d}] "
                f"loss={metrics['loss']:.4f} | "
                f"ce={metrics['loss_ce']:.4f} | "
                f"gbpo={metrics['loss_gbpo']:.4f}"
            )

        if step >= max_steps:
            break


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    run_experiment(use_trie=False)   
    run_experiment(use_trie=True)    

