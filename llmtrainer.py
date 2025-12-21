# llm_trainer.py
# ==========================================================
# 全部 import 必须放在顶部（符合 PEP8）
# ==========================================================

# ---------- Python 标准库 ----------
import os
import json

# ---------- 第三方依赖 ----------
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import transformers

# ---------- 本地模块（可注入类） ----------
# 可让用户传入自定义 TokenExtender，所以这里只定义默认版本
# 若你未来有其他模块, 例如 from utils.xxx import xxx，也放这里
# ==========================================================


# ==========================================================
# Default Token Extender（可替换）
# ==========================================================
class TokenExtender:
    """从 .index.json 文件中读取新增 token"""
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load(self):
        fname = os.path.join(self.data_path, self.dataset + self.index_file)
        with open(fname, "r") as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens:
            return self.new_tokens

        if self.indices is None:
            self._load()

        unique = set()
        for lst in self.indices.values():
            for tok in lst:
                unique.add(tok)

        self.new_tokens = sorted(list(unique))
        return self.new_tokens


# ==========================================================
#                    LLM Trainer（主类）
# ==========================================================
class LLMTrainer:
    """
    完整封装的 HuggingFace LLM Trainer。
    - tokenizer / model 构造
    - 新 token 扩展
    - 冻结模型
    - 自动转换 dataset 为 HF Dataset
    - 调用 Trainer.fit()
    """

    def __init__(
        self,
        base_model,
        train_dataset,
        val_dataset,
        output_dir,
        micro_batch_size=4,
        batch_size=128,
        num_epochs=3,
        learning_rate=3e-4,
        cutoff_len=512,
        group_by_length=False,
        freeze_LLM=False,
        sid_index_path="",
        sample=-1,
        seed=42,
        resume_from_checkpoint=None,
        category="",
        tokenizer=None,
        train_from_scratch=False,
        ddp=False,
        local_rank=0,
        token_extender_class=TokenExtender,  # ⭐ 可替换 TokenExtender
    ):
        self.base_model = base_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.micro_batch_size = micro_batch_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.cutoff_len = cutoff_len
        self.group_by_length = group_by_length
        self.freeze_LLM = freeze_LLM
        self.sid_index_path = sid_index_path
        self.sample = sample
        self.seed = seed
        self.resume = resume_from_checkpoint
        self.category = category
        self.tokenizer = tokenizer
        self.train_from_scratch = train_from_scratch
        self.ddp = ddp
        self.local_rank = local_rank

        self.token_extender_class = token_extender_class

        os.makedirs(output_dir, exist_ok=True)

    # ---------------- tokenizer ----------------
    def prepare_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return tokenizer

    # ---------------- 模型构造 ----------------
    def prepare_model(self, tokenizer):
        original_vocab = len(tokenizer)

        if not self.train_from_scratch:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=torch.bfloat16
            )
        else:
            config = AutoConfig.from_pretrained(self.base_model)
            model = AutoModelForCausalLM.from_config(config)

        # ---------- add new tokens ----------
        new_tokens = None
        if self.sid_index_path and os.path.exists(self.sid_index_path):
            extender = self.token_extender_class(
                data_path=os.path.dirname(self.sid_index_path),
                dataset=os.path.basename(self.sid_index_path).split(".")[0],
            )
            new_tokens = extender.get_new_tokens()

            if new_tokens:
                tokenizer.add_tokens(new_tokens)
                model.resize_token_embeddings(len(tokenizer))
                print(f"[TokenExtender] Added {len(new_tokens)} new tokens.")

        # ---------- freeze LLM ----------
        if self.freeze_LLM:
            for p in model.parameters():
                p.requires_grad = False

            # new tokens remain trainable
            if new_tokens:
                emb = model.get_input_embeddings().weight
                emb.requires_grad = True

                def mask_grad(g):
                    g[:original_vocab].zero_()
                    return g

                emb.register_hook(mask_grad)

        return model

    # ---------------- dataset → HF ----------------
    def convert_dataset(self, dataset):
        first = dataset[0]
        col = {k: [] for k in first.keys()}

        for item in dataset:
            for k, v in item.items():
                if torch.is_tensor(v):
                    col[k].append(v.tolist())
                else:
                    col[k].append(v)

        return HFDataset.from_dict(col).shuffle(seed=self.seed)

    # ---------------- training ----------------
    def train(self):
        print("\n===============================")
        print("     LLM Training Start")
        print("===============================\n")

        tokenizer = self.prepare_tokenizer()
        model = self.prepare_model(tokenizer)

        hf_train = self.convert_dataset(self.train_dataset)
        hf_val = self.convert_dataset(self.val_dataset)

        gradient_accum = self.batch_size // self.micro_batch_size
        eval_steps = max(1, int(0.05 * len(hf_train)))

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            gradient_accumulation_steps=gradient_accum,
            per_device_train_batch_size=self.micro_batch_size,
            per_device_eval_batch_size=self.micro_batch_size,
            warmup_steps=20,
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=eval_steps,
            eval_steps=eval_steps,
            save_total_limit=1,
            group_by_length=self.group_by_length,
            load_best_model_at_end=True,
            report_to=None,
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_train,
            eval_dataset=hf_val,
            args=args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

        if hasattr(model, "config"):
            model.config.use_cache = False

        trainer.train(resume_from_checkpoint=self.resume)

        # ---------- save final ----------
        final = os.path.join(self.output_dir, "final_checkpoint")
        os.makedirs(final, exist_ok=True)
        trainer.model.save_pretrained(final)
        tokenizer.save_pretrained(final)

        print(f"\n[LLMTrainer] DONE. Saved to: {final}\n")

