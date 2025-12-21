import os
import sys
from typing import List
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
# from data import D3Dataset, SFTData, SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, PreferenceSFTDataset, UserPreference2sidSFTDataset, TitleHistory2SidSFTDataset
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset

from lazydecoderonly import LazyDecoder
from minitrainer import MiniTrainer
from llmtrainer import LLMTrainer,TokenExtender

class LazyDecoderHFWrapper(nn.Module):
    def __init__(self, lazy_decoder, pad_id=0):
        super().__init__()
        self.model = lazy_decoder
        self.pad_id = pad_id

    def forward(self, input_ids, labels=None,
                user_static=None, short_term=None, long_term=None, **kwargs):

        out = self.model(
            target_ids=input_ids,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
            return_hidden=False,
        )
        logits = out["logits"]

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)
            loss = loss_fn(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        return {"loss": loss, "logits": logits}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LazyDecoderDummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_len=32, num_samples=2000, ctx_dim=256):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.ctx_dim = ctx_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        target_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        user_static = torch.randn(1, self.ctx_dim)   # (B, Ns, d_ctx_in)
        short_term  = torch.randn(5, self.ctx_dim)
        long_term   = torch.randn(10, self.ctx_dim)

        return {
            "input_ids": target_ids,   
            "labels": target_ids,
            "user_static": user_static,
            "short_term": short_term,
            "long_term": long_term
        }



def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    use_lazy_decoder: bool = True,
    dummy_data: bool = True,  

    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,

    group_by_length: bool = False,
    freeze_LLM: bool = False,

    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    if output_dir == "" or output_dir is None:
        output_dir = "./outputs"

    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    if category in category_dict:
        category = category_dict[category]

    if use_lazy_decoder:
        print("\n==============================")
        print("   [Mode] LazyDecoder Training")
        print("==============================\n")

        lazy = LazyDecoder(
            vocab_size=50,
            d_model=256,
            n_layers=4,
            n_heads_q=8,
            gkv=2,
            d_ff=512,
            d_ctx_in=256,
            lkv=1,
            skv=1,
        )
        model = LazyDecoderHFWrapper(lazy).to(device)

        if dummy_data:
            train_data = LazyDecoderDummyDataset(num_samples=400, vocab_size=50)
            val_data = LazyDecoderDummyDataset(num_samples=80, vocab_size=50)
        else:
            raise NotImplementedError("LazyDecoder 模式暂不支持真实 dataset 输入")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        trainer = MiniTrainer(
            model=model,
            optimizer=optimizer,
            train_dataset=train_data,
            eval_dataset=val_data,
            batch_size=micro_batch_size,
            num_epochs=num_epochs,
            grad_accum_steps=batch_size // micro_batch_size,
            fp16=False,
            bf16=False,
            log_steps=20,
            output_dir=output_dir,
            ddp=False,
            local_rank=0,
            save_best=True,
            max_grad_norm=1.0,
        )

        trainer.train()

        save_path = os.path.join(output_dir, "lazy_decoder_final.pt")
        torch.save(model.state_dict(), save_path)
        print(f"[LazyDecoder] model saved to {save_path}")
        return save_path

    print("\n==============================")
    print("    [Mode] HF LLM Trainer")
    print("==============================\n")

    train_dataset = SidSFTDataset(
        train_file=train_file,
        tokenizer=None,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    val_dataset = SidSFTDataset(
        train_file=eval_file,
        tokenizer=None,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )

    trainer = LLMTrainer(
        base_model=base_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        micro_batch_size=micro_batch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        cutoff_len=cutoff_len,
        group_by_length=group_by_length,
        freeze_LLM=freeze_LLM,
        sid_index_path=sid_index_path,
        sample=sample,
        seed=seed,
        resume_from_checkpoint=resume_from_checkpoint,
        category=category,
        train_from_scratch=train_from_scratch,
        ddp=False,
        local_rank=0,
        token_extender_class=TokenExtender,
    )

    final_path = trainer.train()
    print(f"[HF LLM] final model saved to: {final_path}")

    return final_path


if __name__ == "__main__":
    fire.Fire(train)
