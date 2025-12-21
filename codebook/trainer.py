import os
import math
import heapq
import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy
from time import time

from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from utils import ensure_dir, set_color, get_local_time, delete_file


class Trainer(object):
    def __init__(self, args, model, data_num: int):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs

        # Optimizer / Scheduler 
        self.optimizer_name = getattr(args, "optimizer", "adamw")
        self.scheduler_name = getattr(args, "scheduler", "cosine")

        # Warmup & steps
        self.warmup_epochs = getattr(args, "warmup_epochs", 0)
        self.warmup_steps = self.warmup_epochs * data_num
        self.max_steps = self.epochs * data_num

        # Grad Accumulation
        self.grad_accum_steps = max(1, getattr(args, "grad_accum_steps", 1))

        # AMP
        self.use_amp = bool(getattr(args, "use_amp", False))
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA
        self.use_ema = bool(getattr(args, "use_ema", False))
        self.ema_decay = float(getattr(args, "ema_decay", 0.999))
        self.ema_model = deepcopy(model).eval() if self.use_ema else None

        # device
        self.device = torch.device(args.device)
        self.model = self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

        # save
        self.save_limit = args.save_limit
        self.eval_step = min(args.eval_step, self.epochs)
        self.ckpt_dir_root = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir_root, saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"

        self.best_save_heap = []
        self.newest_save_queue = []

        # optimizer & scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.global_step = 0

    def _build_optimizer(self):
        params = self.model.parameters()
        lr = self.lr
        wd = self.weight_decay
        name = self.optimizer_name.lower()

        if name == "adamw":
            opt = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif name == "adam":
            opt = optim.Adam(params, lr=lr, weight_decay=wd)
        elif name == "sgd":
            opt = optim.SGD(params, lr=lr, weight_decay=wd)
        elif name == "adagrad":
            opt = optim.Adagrad(params, lr=lr, weight_decay=wd)
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif name == "rmsprop":
            opt = optim.RMSprop(params, lr=lr, weight_decay=wd)
        else:
            self.logger.warning(
                f"Unrecognized optimizer={name}, fallback to AdamW"
            )
            opt = optim.AdamW(params, lr=lr, weight_decay=wd)

        self.logger.info(f"[TrainerV3] optimizer = {opt.__class__.__name__}")
        return opt

    def _build_scheduler(self):
        name = self.scheduler_name.lower()

        if name == "linear":
            sch = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        elif name == "constant":
            sch = get_constant_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
            )
        else:
            self.logger.info(
                f"[TrainerV3] Use cosine + warmup scheduler "
                f"(warmup_steps={self.warmup_steps}, max_steps={self.max_steps})"
            )

            def lr_lambda(current_step: int):
                if current_step < self.warmup_steps and self.warmup_steps > 0:
                    return max(float(current_step) / float(self.warmup_steps), 1e-8)
                if self.max_steps <= self.warmup_steps:
                    return 1.0
                progress = (current_step - self.warmup_steps) / float(
                    max(1, self.max_steps - self.warmup_steps)
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            sch = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            self.scheduler_name = "cosine"

        self.logger.info(f"[TrainerV3] scheduler = {self.scheduler_name}")
        return sch

    @torch.no_grad()
    def _update_ema(self):
        if self.ema_model is None:
            return
        decay = self.ema_decay
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(self.model.named_parameters())
        for k in ema_params.keys():
            ema_params[k].data.mul_(decay).add_(
                model_params[k].data, alpha=1.0 - decay
            )

    def _check_nan(self, loss: torch.Tensor):
        if torch.isnan(loss):
            raise ValueError("Training loss is NaN")

    def _train_epoch(self, train_data, epoch_idx: int):
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0

        iter_data = tqdm(
            train_data,
            total=len(train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
        )

        self.optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(iter_data):
            batch = batch.to(self.device)

            with autocast(enabled=self.use_amp):
                out, rq_loss, _ = self.model(batch)
                loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=batch)
                loss = loss / self.grad_accum_steps

            self._check_nan(loss)

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.grad_accum_steps
            total_recon_loss += loss_recon.item()

            if (step_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                if self.scheduler is not None:
                    self.scheduler.step()
                self._update_ema()

        avg_loss = total_loss / len(train_data)
        avg_recon = total_recon_loss / len(train_data)
        return avg_loss, avg_recon

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        eval_model = self.ema_model if self.ema_model is not None else self.model
        eval_model.eval()

        from tqdm import tqdm  

        iter_data = tqdm(
            valid_data,
            total=len(valid_data),
            ncols=100,
            desc=set_color("Evaluate   ", "pink"),
        )

        indices_set = set()
        num_sample = 0

        for _, batch in enumerate(iter_data):
            num_sample += len(batch)
            batch = batch.to(self.device)

            indices = eval_model.get_indices(batch)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()

            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set)) / float(num_sample)
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1.0, ckpt_file=None):
        ckpt_path = (
            os.path.join(self.ckpt_dir, ckpt_file)
            if ckpt_file
            else os.path.join(
                self.ckpt_dir, f"epoch_{epoch}_collision_{collision_rate:.4f}_model.pth"
            )
        )

        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.use_amp:
            state["scaler"] = self.scaler.state_dict()
        if self.ema_model is not None:
            state["ema_state_dict"] = self.ema_model.state_dict()

        torch.save(state, ckpt_path, pickle_protocol=4)
        self.logger.info(set_color("Saving current", "blue") + f": {ckpt_path}")
        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        out = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        out += set_color("train loss", "blue") + ": %.4f" % loss
        out += ", "
        out += (
            set_color("reconstruction loss", "blue")
            + ": %.4f" % recon_loss
        )
        return out + "]"

    def fit(self, data: DataLoader):
        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            t0 = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            t1 = time()
            train_log = self._generate_train_loss_output(
                epoch_idx, t0, t1, train_loss, train_recon_loss
            )
            self.logger.info(train_log)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                v0 = time()
                collision_rate = self._valid_epoch(data)
                v1 = time()

                # best loss
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(
                        epoch=epoch_idx, ckpt_file=self.best_loss_ckpt
                    )

                # best collision
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(
                        epoch_idx,
                        collision_rate=collision_rate,
                        ckpt_file=self.best_collision_ckpt,
                    )
                else:
                    cur_eval_step += 1

                valid_log = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, v1 - v0, collision_rate)
                self.logger.info(valid_log)

                ckpt_path = self._save_checkpoint(
                    epoch_idx, collision_rate=collision_rate
                )
                now_save = (-collision_rate, ckpt_path)

                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)

                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)
                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])

        return self.best_loss, self.best_collision_rate

