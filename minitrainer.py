import os
import math
import json
import torch
import time
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm   # ★ 加进度条

class MiniTrainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataset,
        eval_dataset=None,
        batch_size=32,
        num_epochs=3,
        grad_accum_steps=1,
        fp16=False,
        bf16=False,
        log_steps=10,
        output_dir="./checkpoint",
        ddp=False,
        local_rank=0,
        save_best=True,
        max_grad_norm=1.0,
        lr_scheduler=None,
        warmup_steps=0,
        resume_path=None,
        smoothing=0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        self.fp16 = fp16
        self.bf16 = bf16
        self.log_steps = log_steps
        self.output_dir = output_dir
        self.ddp = ddp
        self.local_rank = local_rank
        self.save_best = save_best
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.resume_path = resume_path
        self.smoothing = smoothing

        self.device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.scaler = torch.cuda.amp.GradScaler(enabled=fp16)

        if ddp:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)

        os.makedirs(output_dir, exist_ok=True)
        self.best_eval_loss = float("inf")
        self.running_loss = None

        if resume_path is not None:
            self._load_checkpoint(resume_path)

    def autocast(self):
        if self.bf16:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if self.fp16:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return torch.cuda.amp.autocast(enabled=False)

    def _get_dataloader(self, dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and (not self.ddp),
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch_list):
        batch = {}
        for key in batch_list[0].keys():
            batch[key] = torch.stack([x[key] for x in batch_list], dim=0)
        return batch

    def _to_device(self, batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)
        return batch

    # ======================================================
    # ★★★ 加入训练进度条 ★★★
    # ======================================================
    def train(self):
        train_loader = self._get_dataloader(self.train_dataset, shuffle=True)
        step = 0

        for epoch in range(self.num_epochs):

            if self.local_rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=True)
            else:
                pbar = train_loader  # DDP worker 不显示进度条

            self.model.train()

            for batch in pbar:
                batch = self._to_device(batch)

                with self.autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"] / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                # smooth loss
                if self.running_loss is None:
                    self.running_loss = loss.item()
                else:
                    self.running_loss = (
                        self.smoothing * loss.item() + (1 - self.smoothing) * self.running_loss
                    )

                # ★ 更新进度条显示当前 loss
                if self.local_rank == 0:
                    pbar.set_postfix({"loss": f"{self.running_loss:.4f}"})

                step += 1

            # eval
            if self.eval_dataset is not None and self.local_rank == 0:
                eval_loss = self.evaluate()
                print(f"\nEpoch {epoch+1} Eval Loss = {eval_loss:.4f}")

                if self.save_best and eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")

        if self.local_rank == 0:
            self._save_checkpoint("final")

    # ======================================================
    # ★★★ 加入验证进度条 ★★★
    # ======================================================
    def evaluate(self):
        loader = self._get_dataloader(self.eval_dataset, shuffle=False)
        self.model.eval()
        losses = []

        pbar = tqdm(loader, desc="Evaluating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                batch = self._to_device(batch)
                with self.autocast():
                    outputs = self.model(**batch)
                    loss = outputs["loss"].item()
                    losses.append(loss)

                pbar.set_postfix({"loss": f"{loss:.4f}"})

        return sum(losses) / len(losses)

    # ---------------- Save/Load ----------------
    def _save_checkpoint(self, name):
        path = os.path.join(self.output_dir, f"{name}.pt")

        ckpt = {
            "model": self.model.module.state_dict() if self.ddp else self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }
        if self.lr_scheduler:
            ckpt["scheduler"] = self.lr_scheduler.state_dict()

        torch.save(ckpt, path)
        print(f"[Checkpoint Saved] {path}")

    def _load_checkpoint(self, path):
        print(f"[Resume] Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))

        if self.lr_scheduler and "scheduler" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["scheduler"])

        print("[Resume] Done.")


