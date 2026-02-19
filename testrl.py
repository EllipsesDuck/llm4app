# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from lazydecoderonly import LazyDecoder, GBPOTrainer 


# def build_dummy_batch(
#     batch_size=4,
#     T=6,
#     vocab_size=50,
#     d_ctx_in=256,
#     n_static=1,
#     n_short=5,
#     n_long=10,
#     pad_id=0,
# ):
#     """
#     构造一批符合你 LazyDecoder 输入形式的假数据：
#     - target_ids: (B, T)
#     - user_static: (B, N_static, d_ctx_in)
#     - short_term: (B, N_short, d_ctx_in)
#     - long_term:  (B, N_long, d_ctx_in)
#     - rewards: (B,)   —— 用于 RL 部分
#     - group_ids: (B,) —— 用于 group-wise advantage 归一化（可选）
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 随机生成一个 token 序列，前面第一个位置当 BOS（不影响 demo）
#     target_ids = torch.randint(low=0, high=vocab_size, size=(batch_size, T), device=device)
#     target_ids[:, 0] = 1  # 假装 1 是 BOS

#     # 构造三种上下文特征
#     user_static = torch.randn(batch_size, n_static, d_ctx_in, device=device)
#     short_term  = torch.randn(batch_size, n_short,  d_ctx_in, device=device)
#     long_term   = torch.randn(batch_size, n_long,   d_ctx_in, device=device)

#     # 构造一个简单的 reward（这里用随机数，实际上你可以按业务打分）
#     rewards = torch.randn(batch_size, device=device)

#     # 构造 group_ids，假设每两个样本属于同一个 group，用于 group 内 advantage 归一化
#     group_ids = torch.tensor(
#         [i // 2 for i in range(batch_size)],
#         device=device,
#         dtype=torch.long
#     )

#     batch = {
#         "target_ids": target_ids,
#         "user_static": user_static,
#         "short_term": short_term,
#         "long_term": long_term,
#         "rewards": rewards,
#         "group_ids": group_ids,
#     }
#     return batch


# def main():
#     # ==== 1. 一些超参 ====
#     vocab_size = 50
#     d_model = 256
#     n_layers = 4
#     n_heads_q = 4
#     gkv = 1
#     d_ff = 512
#     d_ctx_in = 256

#     pad_id = 0
#     bos_id = 1

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ==== 2. 初始化 LazyDecoder ====
#     model = LazyDecoder(
#         vocab_size=vocab_size,
#         d_model=d_model,
#         n_layers=n_layers,
#         n_heads_q=n_heads_q,
#         gkv=gkv,
#         d_ff=d_ff,
#         d_ctx_in=d_ctx_in,
#         lkv=1,
#         skv=1,
#         pad_id=pad_id,
#         bos_id=bos_id,
#         attn_drop=0.0,
#         resid_drop=0.0,
#     ).to(device)

#     # ==== 3. 初始化 GBPOTrainer ====
#     gbpo_trainer = GBPOTrainer(
#         model=model,
#         lambda_rl=0.1,   # RL loss 的权重
#         clip_eps=0.2,
#         pad_id=pad_id,
#         device=device,
#     )

#     optimizer = optim.Adam(model.parameters(), lr=1e-3)

#     # ==== 4. 先跑几步纯监督（CE）训练，等价于 SFT ====
#     print("====== Supervised pre-training (CE only) ======")
#     for step in range(3):
#         batch = build_dummy_batch(
#             batch_size=4,
#             T=6,
#             vocab_size=vocab_size,
#             d_ctx_in=d_ctx_in,
#         )
#         stats = gbpo_trainer.train_step(
#             batch=batch,
#             optimizer=optimizer,
#             use_rl=False,   # 只用 CE，不加 RL
#         )
#         print(f"[SFT step {step}] loss={stats['loss']:.4f}, ce={stats['loss_ce']:.4f}, gbpo={stats['loss_gbpo']:.4f}")

#     # 同步 old_policy（后面 RL 会用）
#     gbpo_trainer.sync_old_policy()

#     # ==== 5. 再跑几步带 GBPO 的强化学习微调 ====
#     print("\n====== RL fine-tuning with GBPO ======")
#     for step in range(5):
#         batch = build_dummy_batch(
#             batch_size=4,
#             T=6,
#             vocab_size=vocab_size,
#             d_ctx_in=d_ctx_in,
#         )
#         # 这里你也可以根据 model 当前输出 + 业务规则 来构造更合理的 reward
#         stats = gbpo_trainer.train_step(
#             batch=batch,
#             optimizer=optimizer,
#             use_rl=True,    # 加上 GBPO loss
#         )
#         print(
#             f"[RL step {step}] "
#             f"total={stats['loss']:.4f}, "
#             f"ce={stats['loss_ce']:.4f}, "
#             f"gbpo={stats['loss_gbpo']:.4f}"
#         )

#     # ==== 6. 测试一下推理 generate ====
#     print("\n====== Test LazyDecoder.generate with numeric context ======")
#     with torch.no_grad():
#         test_batch = build_dummy_batch(
#             batch_size=2,
#             T=3,
#             vocab_size=vocab_size,
#             d_ctx_in=d_ctx_in,
#         )
#         gen = model.generate(
#             input_ids=test_batch["target_ids"][:, :1],   # 只给一个 BOS 起点
#             user_static=test_batch["user_static"],
#             short_term=test_batch["short_term"],
#             long_term=test_batch["long_term"],
#             max_new_tokens=5,
#             eos_id=None,
#         )
#         print("generated ids:", gen)


# if __name__ == "__main__":
#     main()

import torch
import random
from trl import GRPOConfig
from generative_rec.flashdecoder.flashdecoder import (
    LazyDecoder,
    LazyDecoderConfig,
    LazyDecoderForCausalLM,
)
from rltrainer import ReReTrainer

def lazydecoder_collator(batch):
    return batch

############################################
# 1. 构造 LazyDecoder 模型
############################################

lazy_config = LazyDecoderConfig(
    vocab_size=50,
    d_model=256,
    n_layers=4,
    n_heads_q=4,
    gkv=1,
    d_ff=512,
)

lazy_decoder = LazyDecoder(
    vocab_size=lazy_config.vocab_size,
    d_model=lazy_config.d_model,
    n_layers=lazy_config.n_layers,
    n_heads_q=lazy_config.n_heads_q,
    gkv=lazy_config.gkv,
    d_ff=lazy_config.d_ff,
    d_ctx_in=256,
)

model = LazyDecoderForCausalLM(lazy_config, lazy_decoder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

############################################
# 2. 构造 Dummy Dataset
############################################

def make_sample():
    return {
        "input_ids": torch.tensor([1, 2, 3]),
        "user_static": torch.randn(1, 256),
        "short_term": torch.randn(3, 256),
        "long_term": torch.randn(5, 256),
        "reward": random.random(),
    }

dataset = [make_sample() for _ in range(40)]


############################################
# 3. Dummy Reward Function
############################################

def dummy_reward(prompts, completions, **kw):
    return [random.random() for _ in range(len(completions))]


############################################
# 4. GRPOConfig
############################################

args = GRPOConfig(
    output_dir="./out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,

    num_generations=4,
    max_prompt_length=4,
    max_completion_length=5,

    logging_steps=1,
    save_steps=1000,

    bf16=False,   # 必须关闭，否则 MPS 直接报错
    fp16=False,   # 一并关闭，避免 transformers 自动切换

    gradient_checkpointing=False,        # ★ 关键
    gradient_checkpointing_kwargs=None,  # ★ 禁止自动启用
)


############################################
# 5. 构造 ReReTrainer
############################################

trainer = ReReTrainer(
    model=model,
    base_model=None,
    reward_funcs=dummy_reward,
    args=args,
    train_dataset=dataset,
    eval_dataset=None,
    processing_class=None,   # LazyDecoder 分支不需要 tokenizer
    data_collator=lazydecoder_collator,
)


############################################
# 6. 训练一次
############################################

trainer.train()
