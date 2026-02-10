import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Dict


@dataclass
class TaskSpec:
    name: str
    task_type: str  # "binary" | "multiclass" | "regression"
    out_dim: int = 1


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3, use_ln=True):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_ln:
            layers.append(nn.LayerNorm(hidden_dim))
        layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TaskTower(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def compute_multitask_losses(preds, targets, task_specs, reduction="mean"):
    assert len(preds) == len(targets) == len(task_specs)
    losses = []
    for p, y, spec in zip(preds, targets, task_specs):
        if spec.task_type == "binary":
            losses.append(F.binary_cross_entropy_with_logits(p, y, reduction=reduction))
        elif spec.task_type == "multiclass":
            losses.append(F.cross_entropy(p, y.long(), reduction=reduction))
        elif spec.task_type == "regression":
            losses.append(F.mse_loss(p, y, reduction=reduction))
        else:
            raise ValueError(f"Unknown task_type: {spec.task_type}")
    return losses


class CGC_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_tasks,
        num_specific_experts,
        num_shared_experts,
        expert_output_dim,
        expert_hidden_dim,
        expert_dropout=0.3,
        expert_use_ln=True,
        is_last_layer=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.num_specific_experts = num_specific_experts
        self.num_shared_experts = num_shared_experts
        self.expert_output_dim = expert_output_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.is_last_layer = is_last_layer

        self.shared_experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, expert_output_dim, dropout=expert_dropout, use_ln=expert_use_ln)
            for _ in range(num_shared_experts)
        ])

        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                Expert(input_dim, expert_hidden_dim, expert_output_dim, dropout=expert_dropout, use_ln=expert_use_ln)
                for _ in range(num_specific_experts)
            ])
            for _ in range(num_tasks)
        ])

        self.task_gates = nn.ModuleList([
            nn.Linear(input_dim, num_specific_experts + num_shared_experts)
            for _ in range(num_tasks)
        ])

        self.shared_gate = nn.Linear(input_dim, num_tasks * num_specific_experts + num_shared_experts)

    @staticmethod
    def _mix_experts(experts_ebd, gate_be):
        experts_bed = experts_ebd.permute(1, 0, 2).contiguous()
        gate_b1e = gate_be.unsqueeze(1)
        out_bd = torch.bmm(gate_b1e, experts_bed).squeeze(1)
        return out_bd

    def forward(self, inputs):
        shared_input = inputs[0]
        task_inputs = inputs[1:]
        assert len(task_inputs) == self.num_tasks

        shared_expert_outputs = torch.stack([e(shared_input) for e in self.shared_experts], dim=0)

        task_expert_outputs = []
        for t in range(self.num_tasks):
            outs = torch.stack([e(task_inputs[t]) for e in self.task_experts[t]], dim=0)
            task_expert_outputs.append(outs)

        task_outputs = []
        for t in range(self.num_tasks):
            experts = torch.cat([task_expert_outputs[t], shared_expert_outputs], dim=0)
            gate = F.softmax(self.task_gates[t](task_inputs[t]), dim=1)
            out = self._mix_experts(experts, gate)
            task_outputs.append(out)

        if self.is_last_layer:
            return task_outputs

        all_experts = torch.cat(task_expert_outputs + [shared_expert_outputs], dim=0)
        shared_gate = F.softmax(self.shared_gate(shared_input), dim=1)
        shared_out = self._mix_experts(all_experts, shared_gate)
        return [shared_out] + task_outputs


class MultiTask_PLE(nn.Module):
    def __init__(
        self,
        task_specs,
        input_dim,
        num_cgc_layers=2,
        num_specific_experts=3,
        num_shared_experts=4,
        expert_output_dim=32,
        expert_hidden_dim=64,
        tower_hidden_dim=32,
        use_task_embedding=True,
        task_embedding_hidden_ratio=0.5,
        expert_dropout=0.3,
        tower_dropout=0.4,
        expert_use_ln=True,
        rep_proj_dim=0,
    ):
        super().__init__()
        self.task_specs = list(task_specs)
        self.num_tasks = len(self.task_specs)
        self.input_dim = input_dim
        self.use_task_embedding = use_task_embedding
        assert num_cgc_layers >= 1

        if use_task_embedding:
            h = max(1, int(input_dim * task_embedding_hidden_ratio))
            self.task_embeddings = nn.ModuleList([
                nn.Sequential(nn.Linear(input_dim, h), nn.ReLU(), nn.Linear(h, input_dim))
                for _ in range(self.num_tasks)
            ])
        else:
            self.task_embeddings = None

        self.cgc_layers = nn.ModuleList()
        self.cgc_layers.append(
            CGC_Layer(
                input_dim=input_dim,
                num_tasks=self.num_tasks,
                num_specific_experts=num_specific_experts,
                num_shared_experts=num_shared_experts,
                expert_output_dim=expert_output_dim,
                expert_hidden_dim=expert_hidden_dim,
                expert_dropout=expert_dropout,
                expert_use_ln=expert_use_ln,
                is_last_layer=(num_cgc_layers == 1),
            )
        )
        for i in range(1, num_cgc_layers):
            is_last = (i == num_cgc_layers - 1)
            self.cgc_layers.append(
                CGC_Layer(
                    input_dim=expert_output_dim,
                    num_tasks=self.num_tasks,
                    num_specific_experts=num_specific_experts,
                    num_shared_experts=num_shared_experts,
                    expert_output_dim=expert_output_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    expert_dropout=expert_dropout,
                    expert_use_ln=expert_use_ln,
                    is_last_layer=is_last,
                )
            )

        self.towers = nn.ModuleList([
            TaskTower(expert_output_dim, spec.out_dim, tower_hidden_dim, dropout=tower_dropout)
            for spec in self.task_specs
        ])

        self.rep_proj = None
        self.rep_proj_dim = int(rep_proj_dim) if rep_proj_dim else 0
        if self.rep_proj_dim > 0:
            self.rep_proj = nn.Linear(input_dim, self.rep_proj_dim)

    def forward_features_with_shared(self, x):
        if self.use_task_embedding:
            task_feats = [emb(x) for emb in self.task_embeddings]
            shared_feat = x
        else:
            task_feats = [x for _ in range(self.num_tasks)]
            shared_feat = x

        shared_h = shared_feat
        if self.rep_proj is not None:
            shared_h = self.rep_proj(shared_h)

        inputs = [shared_feat] + task_feats
        for layer in self.cgc_layers:
            inputs = layer(inputs)
            if len(inputs) == self.num_tasks + 1:
                shared_h = inputs[0]
        if self.rep_proj is not None and shared_h.shape[-1] != self.rep_proj_dim:
            pass
        assert len(inputs) == self.num_tasks
        task_hs = inputs
        return shared_h, task_hs

    def forward(self, x):
        _, task_hs = self.forward_features_with_shared(x)
        return [self.towers[i](task_hs[i]) for i in range(self.num_tasks)]


class MMoE_Layer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_tasks,
        num_experts,
        expert_output_dim,
        expert_hidden_dim,
        expert_dropout=0.3,
        expert_use_ln=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim

        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dim, expert_output_dim, dropout=expert_dropout, use_ln=expert_use_ln)
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, num_experts)
            for _ in range(num_tasks)
        ])

    @staticmethod
    def _mix(experts_ebd, gate_be):
        experts_bed = experts_ebd.permute(1, 0, 2).contiguous()
        gate_b1e = gate_be.unsqueeze(1)
        return torch.bmm(gate_b1e, experts_bed).squeeze(1)

    def forward(self, x):
        experts_out = torch.stack([e(x) for e in self.experts], dim=0)
        outs = []
        for t in range(self.num_tasks):
            gate = F.softmax(self.gates[t](x), dim=1)
            outs.append(self._mix(experts_out, gate))
        return outs


class MMoE_Stack(nn.Module):
    def __init__(
        self,
        input_dim,
        num_tasks,
        num_layers,
        num_experts,
        expert_output_dim,
        expert_hidden_dim,
        expert_dropout=0.3,
        expert_use_ln=True,
        shared_aggregate="mean",
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.shared_aggregate = shared_aggregate

        self.layers = nn.ModuleList()
        dim_in = input_dim
        for _ in range(num_layers):
            self.layers.append(
                MMoE_Layer(
                    input_dim=dim_in,
                    num_tasks=num_tasks,
                    num_experts=num_experts,
                    expert_output_dim=expert_output_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    expert_dropout=expert_dropout,
                    expert_use_ln=expert_use_ln,
                )
            )
            dim_in = expert_output_dim

        self.concat_proj = None
        if shared_aggregate == "concat_proj":
            self.concat_proj = nn.Linear(num_tasks * expert_output_dim, expert_output_dim)

    def forward(self, x):
        shared_h = x
        task_hs = []
        for layer in self.layers:
            task_hs = layer(shared_h)
            if self.shared_aggregate == "mean":
                shared_h = torch.stack(task_hs, dim=0).mean(dim=0)
            else:
                shared_h = self.concat_proj(torch.cat(task_hs, dim=1))
        return shared_h, task_hs


class Hybrid_MMoE_PLE(nn.Module):
    def __init__(
        self,
        task_specs,
        input_dim,
        mmoe_layers=1,
        mmoe_num_experts=8,
        mmoe_expert_output_dim=64,
        mmoe_expert_hidden_dim=128,
        mmoe_dropout=0.3,
        mmoe_shared_aggregate="mean",
        ple_layers=2,
        ple_num_specific_experts=3,
        ple_num_shared_experts=4,
        ple_expert_output_dim=32,
        ple_expert_hidden_dim=64,
        ple_expert_dropout=0.3,
        tower_hidden_dim=32,
        tower_dropout=0.4,
        expert_use_ln=True,
        rep_proj_dim=0,
    ):
        super().__init__()
        self.task_specs = list(task_specs)
        self.num_tasks = len(self.task_specs)
        assert mmoe_layers >= 1
        assert ple_layers >= 1

        self.mmoe = MMoE_Stack(
            input_dim=input_dim,
            num_tasks=self.num_tasks,
            num_layers=mmoe_layers,
            num_experts=mmoe_num_experts,
            expert_output_dim=mmoe_expert_output_dim,
            expert_hidden_dim=mmoe_expert_hidden_dim,
            expert_dropout=mmoe_dropout,
            expert_use_ln=expert_use_ln,
            shared_aggregate=mmoe_shared_aggregate,
        )

        self.ple_cgc = nn.ModuleList()
        self.ple_cgc.append(
            CGC_Layer(
                input_dim=mmoe_expert_output_dim,
                num_tasks=self.num_tasks,
                num_specific_experts=ple_num_specific_experts,
                num_shared_experts=ple_num_shared_experts,
                expert_output_dim=ple_expert_output_dim,
                expert_hidden_dim=ple_expert_hidden_dim,
                expert_dropout=ple_expert_dropout,
                expert_use_ln=expert_use_ln,
                is_last_layer=(ple_layers == 1),
            )
        )
        for i in range(1, ple_layers):
            is_last = (i == ple_layers - 1)
            self.ple_cgc.append(
                CGC_Layer(
                    input_dim=ple_expert_output_dim,
                    num_tasks=self.num_tasks,
                    num_specific_experts=ple_num_specific_experts,
                    num_shared_experts=ple_num_shared_experts,
                    expert_output_dim=ple_expert_output_dim,
                    expert_hidden_dim=ple_expert_hidden_dim,
                    expert_dropout=ple_expert_dropout,
                    expert_use_ln=expert_use_ln,
                    is_last_layer=is_last,
                )
            )

        self.towers = nn.ModuleList([
            TaskTower(ple_expert_output_dim, spec.out_dim, tower_hidden_dim, dropout=tower_dropout)
            for spec in self.task_specs
        ])

        self.rep_proj = None
        self.rep_proj_dim = int(rep_proj_dim) if rep_proj_dim else 0
        if self.rep_proj_dim > 0:
            self.rep_proj = nn.Linear(mmoe_expert_output_dim, self.rep_proj_dim)

    def forward_features_with_shared(self, x):
        shared_h, task_hs = self.mmoe(x)

        shared_rep = shared_h
        if self.rep_proj is not None:
            shared_rep = self.rep_proj(shared_rep)

        inputs = [shared_h] + task_hs
        for layer in self.ple_cgc:
            inputs = layer(inputs)
            if len(inputs) == self.num_tasks + 1:
                shared_rep = inputs[0]
        assert len(inputs) == self.num_tasks
        task_hs2 = inputs
        return shared_rep, task_hs2

    def forward(self, x):
        _, task_hs = self.forward_features_with_shared(x)
        return [self.towers[i](task_hs[i]) for i in range(self.num_tasks)]


class GradNormBalancerOnRep(nn.Module):
    def __init__(
        self,
        num_tasks,
        alpha=0.5,
        eps=1e-8,
        init_equal=True,
        logw_clip=10.0,
        gn_fp32=True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.eps = eps
        self.logw_clip = float(logw_clip) if logw_clip is not None else None
        self.gn_fp32 = bool(gn_fp32)

        if init_equal:
            init = torch.zeros(num_tasks)
        else:
            init = torch.randn(num_tasks) * 0.01
        self.log_w = nn.Parameter(init)

        self.register_buffer("L0", torch.zeros(num_tasks))
        self._has_L0 = False

    def weights(self):
        if self.logw_clip is not None:
            logw = torch.clamp(self.log_w, -self.logw_clip, self.logw_clip)
        else:
            logw = self.log_w
        w = torch.exp(logw)
        w = w * (self.num_tasks / (w.sum() + self.eps))
        return w

    @torch.no_grad()
    def maybe_init_L0(self, task_losses):
        if not self._has_L0:
            self.L0.copy_(torch.tensor([l.item() for l in task_losses], device=self.L0.device))
            self._has_L0 = True

    def weighted_total_loss(self, task_losses):
        w = self.weights().to(task_losses[0].device, task_losses[0].dtype)
        total = torch.zeros((), device=task_losses[0].device, dtype=task_losses[0].dtype)
        for i, l in enumerate(task_losses):
            total = total + w[i] * l
        return total

    def gradnorm_loss_on_rep(self, task_losses, shared_rep, create_graph=True):
        assert len(task_losses) == self.num_tasks
        self.maybe_init_L0(task_losses)

        device = task_losses[0].device
        dtype = task_losses[0].dtype

        w = self.weights().to(device=device, dtype=dtype)

        if self.gn_fp32:
            shared_rep = shared_rep.float()
            task_losses = [l.float() for l in task_losses]
            w = w.float()

        G_list = []
        for i in range(self.num_tasks):
            gi = torch.autograd.grad(
                w[i] * task_losses[i],
                shared_rep,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=False,
            )[0]
            G_i = torch.sqrt((gi ** 2).mean() + self.eps)
            G_list.append(G_i)

        G = torch.stack(G_list)
        G_bar = G.mean()

        L = torch.stack(task_losses)
        L0 = self.L0.to(device=device, dtype=L.dtype)
        ratio = L / (L0 + self.eps)
        r = ratio / (ratio.mean() + self.eps)
        target = (G_bar * (r ** self.alpha)).detach()

        gn = torch.abs(G - target).sum()
        return gn, w, G.detach()


def training_step_amp(
    model,
    optimizer_model,
    x,
    y_list,
    task_specs,
    scaler=None,
    clip_grad=None,
):
    model.train()
    device = x.device
    use_amp = (scaler is not None)

    if use_amp:
        optimizer_model.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(True):
            preds = model(x)
            task_losses = compute_multitask_losses(preds, y_list, task_specs, reduction="mean")
            total_loss = sum(task_losses)
        scaler.scale(total_loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer_model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer_model)
        scaler.update()
    else:
        optimizer_model.zero_grad(set_to_none=True)
        preds = model(x)
        task_losses = compute_multitask_losses(preds, y_list, task_specs, reduction="mean")
        total_loss = sum(task_losses)
        total_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer_model.step()

    out = {"total_loss": float(total_loss.detach().item())}
    for i, l in enumerate(task_losses):
        out[f"loss_{task_specs[i].name}"] = float(l.detach().item())
    return out


def training_step_gradnorm_amp(
    model,
    balancer,
    optimizer_model,
    optimizer_w,
    x,
    y_list,
    task_specs,
    gradnorm_on=True,
    gn_update_every=10,
    gn_warmup_steps=0,
    step_idx=0,
    clip_grad=None,
    scaler=None,
):
    model.train()
    use_amp = (scaler is not None)

    do_gn = bool(gradnorm_on) and (step_idx >= gn_warmup_steps) and (gn_update_every > 0) and (step_idx % gn_update_every == 0)

    optimizer_model.zero_grad(set_to_none=True)

    if use_amp:
        with torch.cuda.amp.autocast(True):
            shared_rep, _ = model.forward_features_with_shared(x)
            preds = model(x)
            task_losses = compute_multitask_losses(preds, y_list, task_specs, reduction="mean")
            total_loss = balancer.weighted_total_loss(task_losses)

        scaler.scale(total_loss).backward(retain_graph=do_gn)

        if clip_grad is not None:
            scaler.unscale_(optimizer_model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer_model)
        scaler.update()
    else:
        shared_rep, _ = model.forward_features_with_shared(x)
        preds = model(x)
        task_losses = compute_multitask_losses(preds, y_list, task_specs, reduction="mean")
        total_loss = balancer.weighted_total_loss(task_losses)
        total_loss.backward(retain_graph=do_gn)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer_model.step()

    gn_value = 0.0
    w_value = balancer.weights().detach().cpu().tolist()

    if do_gn:
        optimizer_w.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(False):
            shared_rep_fp32 = shared_rep.float().requires_grad_(True)
            task_losses_fp32 = [l.float() for l in task_losses]
            gn_loss, w, _ = balancer.gradnorm_loss_on_rep(task_losses_fp32, shared_rep_fp32, create_graph=True)
        gn_loss.backward()
        optimizer_w.step()
        gn_value = float(gn_loss.detach().item())
        w_value = w.detach().cpu().tolist()

    out = {"total_loss": float(total_loss.detach().item()), "gradnorm_loss": gn_value}
    for i, l in enumerate(task_losses):
        out[f"loss_{task_specs[i].name}"] = float(l.detach().item())
        out[f"w_{task_specs[i].name}"] = float(w_value[i])
    out["do_gn"] = float(do_gn)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_specs = [
        TaskSpec("ctr", "binary", out_dim=1),
        TaskSpec("cate", "multiclass", out_dim=5),
        TaskSpec("gmv", "regression", out_dim=1),
    ]
    T = len(task_specs)
    B = 64
    input_dim = 128

    model = MultiTask_PLE(
        task_specs=task_specs,
        input_dim=input_dim,
        num_cgc_layers=2,
        num_specific_experts=3,
        num_shared_experts=4,
        expert_output_dim=32,
        expert_hidden_dim=64,
        tower_hidden_dim=32,
        use_task_embedding=True,
        expert_use_ln=True,
        rep_proj_dim=64,
    ).to(device)

    # model = Hybrid_MMoE_PLE(
    #     task_specs=task_specs,
    #     input_dim=input_dim,
    #     mmoe_layers=2,
    #     mmoe_num_experts=8,
    #     mmoe_expert_output_dim=64,
    #     mmoe_expert_hidden_dim=128,
    #     ple_layers=2,
    #     ple_num_specific_experts=3,
    #     ple_num_shared_experts=4,
    #     ple_expert_output_dim=32,
    #     ple_expert_hidden_dim=64,
    #     expert_use_ln=True,
    #     rep_proj_dim=64,
    # ).to(device)

    optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    x = torch.randn(B, input_dim, device=device)
    y_ctr = torch.randint(0, 2, (B, 1), device=device).float()
    y_cate = torch.randint(0, 5, (B,), device=device)
    y_gmv = torch.randn(B, 1, device=device)
    y_list = [y_ctr, y_cate, y_gmv]

    for step in range(5):
        logs = training_step_amp(
            model=model,
            optimizer_model=optimizer_model,
            x=x,
            y_list=y_list,
            task_specs=task_specs,
            scaler=scaler,
            clip_grad=5.0,
        )
        print(step, logs)

    # balancer = GradNormBalancerOnRep(
    #     num_tasks=T,
    #     alpha=0.5,
    #     init_equal=True,
    #     logw_clip=10.0,
    #     gn_fp32=True,
    # ).to(device)
    # optimizer_w = torch.optim.Adam(balancer.parameters(), lr=1e-3)
    #
    # for step in range(50):
    #     logs = training_step_gradnorm_amp(
    #         model=model,
    #         balancer=balancer,
    #         optimizer_model=optimizer_model,
    #         optimizer_w=optimizer_w,
    #         x=x,
    #         y_list=y_list,
    #         task_specs=task_specs,
    #         gradnorm_on=True,
    #         gn_update_every=10,
    #         gn_warmup_steps=20,
    #         step_idx=step,
    #         clip_grad=5.0,
    #         scaler=scaler,
    #     )
    #     if step % 10 == 0:
    #         print(step, logs) 