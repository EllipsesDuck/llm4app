import math
from typing import List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCategoryEmbeddingRegFast(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        temperature: float = 1.0,
        reg_weights: Optional[dict] = None,
        sim_matrix: Optional[torch.Tensor] = None,
        stein_eps: float = 1e-4,
        reg_every: int = 1,
        use_laplacian_trace: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = float(temperature)
        self.stein_eps = float(stein_eps)
        self.reg_every = int(reg_every)
        self.use_laplacian_trace = bool(use_laplacian_trace)

        self.E = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.W = nn.Parameter(torch.randn(num_classes, num_classes) * 0.02)

        weights = {"ent": 0.0, "lap": 0.0, "stein": 0.0}
        if reg_weights:
            weights.update(reg_weights)
        self.reg_weights = weights

        S = sim_matrix if sim_matrix is not None else torch.eye(num_classes)
        self.register_buffer("S", S)

        if self.use_laplacian_trace:
            with torch.no_grad():
                d = S.sum(dim=-1)
                L = torch.diag(d) - S
            self.register_buffer("L", L)
        else:
            self.L = None

        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def forward(self, idx: torch.LongTensor):
        Tmix = F.softmax(self.W / self.temperature, dim=-1)
        p = Tmix[idx]
        e = p @ self.E

        self._step += 1
        if self.reg_every > 1 and int(self._step.item()) % self.reg_every != 0:
            reg_loss = e.new_zeros(())
        else:
            reg_loss = self._regularization(Tmix)

        return e, reg_loss

    def _regularization(self, Tmix: torch.Tensor) -> torch.Tensor:
        reg = Tmix.new_zeros(())
        V = self.num_classes

        w_ent = self.reg_weights.get("ent", 0.0)
        if w_ent != 0.0:
            entropy = -(Tmix * torch.log(Tmix.clamp_min(1e-8))).sum(dim=-1).mean()
            reg = reg + w_ent * entropy

        w_lap = self.reg_weights.get("lap", 0.0)
        if w_lap != 0.0:
            if self.use_laplacian_trace:
                LE = self.L @ self.E
                tr = torch.sum(self.E * LE)
                lap = (2.0 * tr) / (V * V)
            else:
                diff = self.E.unsqueeze(0) - self.E.unsqueeze(1)
                dist2 = (diff ** 2).sum(-1)
                lap = (self.S * dist2).sum() / (V * V)
            reg = reg + w_lap * lap

        w_stein = self.reg_weights.get("stein", 0.0)
        if w_stein != 0.0:
            E0 = self.E - self.E.mean(0, keepdim=True)
            cov = (E0.t() @ E0) / V
            cov = cov + self.stein_eps * torch.eye(
                cov.size(0), device=cov.device, dtype=cov.dtype
            )
            Lc = torch.linalg.cholesky(cov)
            sol = torch.cholesky_solve(E0.t(), Lc).t()
            score = -sol
            stein = -torch.mean(torch.sum(score * E0, dim=-1))
            reg = reg + w_stein * stein

        return reg


class SoftCategoryEmbeddingReg(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        temperature: float = 1.0,
        reg_weights: Optional[dict] = None,
        sim_matrix: Optional[torch.Tensor] = None,
        stein_eps: float = 1e-4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = float(temperature)
        self.stein_eps = float(stein_eps)

        self.E = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)
        self.W = nn.Parameter(torch.randn(num_classes, num_classes) * 0.02)

        weights = {"ent": 0.0, "lap": 0.0, "stein": 0.0}
        if reg_weights:
            weights.update(reg_weights)
        self.reg_weights = weights

        S = sim_matrix if sim_matrix is not None else torch.eye(num_classes)
        self.register_buffer("S", S)

    def forward(self, idx: torch.LongTensor):
        Tmix = F.softmax(self.W / self.temperature, dim=-1)
        p = Tmix[idx]
        e = p @ self.E
        reg_loss = self._regularization(Tmix)
        return e, reg_loss

    def _regularization(self, Tmix: torch.Tensor) -> torch.Tensor:
        reg = Tmix.new_zeros(())
        V = self.num_classes

        w_ent = self.reg_weights.get("ent", 0.0)
        if w_ent != 0.0:
            entropy = -(Tmix * torch.log(Tmix.clamp_min(1e-8))).sum(dim=-1).mean()
            reg = reg + w_ent * entropy

        w_lap = self.reg_weights.get("lap", 0.0)
        if w_lap != 0.0:
            diff = self.E.unsqueeze(0) - self.E.unsqueeze(1)
            dist2 = (diff ** 2).sum(-1)
            lap = (self.S * dist2).sum() / (V * V)
            reg = reg + w_lap * lap

        w_stein = self.reg_weights.get("stein", 0.0)
        if w_stein != 0.0:
            E0 = self.E - self.E.mean(0, keepdim=True)
            cov = (E0.t() @ E0) / V
            cov = cov + self.stein_eps * torch.eye(
                cov.size(0), device=cov.device, dtype=cov.dtype
            )
            sol = torch.linalg.solve(cov, E0.t()).t()
            score = -sol
            stein = -torch.mean(torch.sum(score * E0, dim=-1))
            reg = reg + w_stein * stein

        return reg


class FlowMatchingContinuousEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 3,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.sigma = float(sigma)

        in_dim = input_dim + 1 + cond_dim
        layers: List[nn.Module] = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x0: torch.Tensor, cond: Optional[torch.Tensor] = None):
        B, _ = x0.shape
        device = x0.device

        x1 = torch.randn_like(x0) * self.sigma
        t = torch.rand(B, 1, device=device, dtype=x0.dtype)
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0

        if cond is not None:
            inp = torch.cat([x_t, t, cond], dim=-1)
        else:
            inp = torch.cat([x_t, t], dim=-1)

        v_pred = self.net(inp)
        loss = F.mse_loss(v_pred, u_t)
        return loss, x_t, v_pred

    @torch.inference_mode()
    def reverse_flow(
        self,
        x1: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: int = 20,
    ):
        x = x1
        t_vals = torch.linspace(1.0, 0.0, steps, device=x1.device, dtype=x1.dtype)
        dt = -1.0 / (steps - 1)

        for t in t_vals:
            t_batch = t.expand(x.size(0), 1)
            if cond is not None:
                inp = torch.cat([x, t_batch, cond], dim=-1)
            else:
                inp = torch.cat([x, t_batch], dim=-1)
            v = self.net(inp)
            x = x + dt * v

        return x


class TabularEncoder(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: List[int],
        cat_embed_dim: int = 16,
        reg_weights: Optional[dict] = None,
        cond_mode: str = "concat",
    ):
        super().__init__()
        self.cond_mode = cond_mode

        self.cat_embeddings = nn.ModuleList(
            [
                SoftCategoryEmbeddingReg(
                    num_classes=c, embed_dim=cat_embed_dim, reg_weights=reg_weights
                )
                for c in categorical_cardinalities
            ]
        )

        cond_dim = len(categorical_cardinalities) * cat_embed_dim if cond_mode == "concat" else 0
        self.cont_encoder = FlowMatchingContinuousEncoder(
            input_dim=numeric_dim, cond_dim=cond_dim
        )

    def forward(self, numeric_tensor: torch.Tensor, categorical_idx: List[torch.Tensor]):
        cat_vecs: List[torch.Tensor] = []
        reg_total = numeric_tensor.new_zeros(())

        for emb, idx in zip(self.cat_embeddings, categorical_idx):
            e, reg = emb(idx)
            cat_vecs.append(e)
            reg_total = reg_total + reg

        cat_cond = torch.cat(cat_vecs, dim=-1) if len(cat_vecs) > 0 else None

        fm_loss, x_t, _ = self.cont_encoder(
            numeric_tensor, cond=cat_cond if self.cond_mode == "concat" else None
        )

        tab_embed = torch.cat([x_t, cat_cond], dim=-1) if cat_cond is not None else x_t
        total_loss = fm_loss + reg_total
        return tab_embed, total_loss


class TemporalEncoderCMGRW_Flex(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        sigma_min: float = 0.05,
        sigma_max: float = 0.5,
        lambda_rec: float = 1.0,
        model_type: str = "gru",
        pooling: Literal["last", "mean"] = "last",
        sigma_schedule: Literal["uniform", "log_uniform"] = "log_uniform",
        consistency_mode: Literal["two_sample", "teacher_student"] = "two_sample",
        teacher_on: Literal["f2", "f1"] = "f2",
        detach_teacher: bool = True,
        mlp_mult: int = 2,
        fuse_f_theta: bool = True,
        nhead: int = 4,
        transformer_ff_mult: int = 2,
        transformer_dropout: float = 0.1,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        tcn_activation: Literal["relu", "silu", "gelu"] = "relu",
        tcn_causal: bool = True,
        downsample_stride: int = 1,
        downsample_mode: Literal["avg", "conv"] = "avg",
    ):
        super().__init__()
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lambda_rec = float(lambda_rec)
        self.model_type = model_type.lower()
        self.pooling = pooling
        self.sigma_schedule = sigma_schedule
        self.consistency_mode = consistency_mode
        self.teacher_on = teacher_on
        self.detach_teacher = detach_teacher
        self.fuse_f_theta = fuse_f_theta
        self.tcn_causal = tcn_causal

        self.downsample_stride = int(downsample_stride)
        self.downsample_mode = downsample_mode

        if self.downsample_stride > 1:
            if downsample_mode == "avg":
                self.downsampler = nn.AvgPool1d(
                    kernel_size=self.downsample_stride,
                    stride=self.downsample_stride,
                )
            elif downsample_mode == "conv":
                self.downsampler = nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=self.downsample_stride,
                    stride=self.downsample_stride,
                    padding=0,
                    groups=1,
                    bias=True,
                )
            else:
                raise ValueError(f"Unsupported downsample_mode={downsample_mode}")
        else:
            self.downsampler = None

        if self.model_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=False,
            )
            self.input_proj = None

        elif self.model_type == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * transformer_ff_mult,
                batch_first=True,
                dropout=transformer_dropout,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        elif self.model_type == "tcn":
            act_cls = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[tcn_activation]
            layers: List[nn.Module] = []
            in_ch = input_dim
            self._tcn_total_pad = 0

            for i in range(n_layers):
                out_ch = hidden_dim
                dilation = 2**i
                pad = (tcn_kernel_size - 1) * dilation
                self._tcn_total_pad = pad
                layers.append(
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        tcn_kernel_size,
                        dilation=dilation,
                        padding=pad,
                    )
                )
                layers.append(act_cls())
                layers.append(nn.Dropout(tcn_dropout))
                in_ch = out_ch

            self.encoder = nn.Sequential(*layers)
            self.input_proj = None

        else:
            raise ValueError(f"Unsupported model_type={model_type}")

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.f_theta = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * mlp_mult),
            nn.SiLU(),
            nn.Linear(hidden_dim * mlp_mult, hidden_dim),
        )

    def _maybe_downsample(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.downsampler is None:
            return x_seq
        x = x_seq.transpose(1, 2)
        x = self.downsampler(x)
        x = x.transpose(1, 2)
        return x

    def _sample_sigma(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sigma_schedule == "uniform":
            u = torch.rand(B, 1, device=device, dtype=dtype)
            return u * (self.sigma_max - self.sigma_min) + self.sigma_min

        if self.sigma_schedule == "log_uniform":
            u = torch.rand(B, 1, device=device, dtype=dtype)
            log_min = math.log(self.sigma_min)
            log_max = math.log(self.sigma_max)
            return torch.exp(log_min + u * (log_max - log_min))

        raise ValueError(f"Unsupported sigma_schedule={self.sigma_schedule}")

    def _encode_to_z0(self, x_seq: torch.Tensor) -> torch.Tensor:
        if self.model_type == "gru":
            z_enc, h_n = self.encoder(x_seq)
            if self.pooling == "last":
                z0 = h_n[-1]
            elif self.pooling == "mean":
                z0 = z_enc.mean(dim=1)
            else:
                raise ValueError(f"Unsupported pooling={self.pooling}")
            return self.proj(z0)

        if self.model_type == "transformer":
            h = self.input_proj(x_seq)
            z_enc = self.encoder(h)
            z0 = z_enc.mean(dim=1)
            return self.proj(z0)

        h = x_seq.transpose(1, 2)
        z = self.encoder(h)

        if self.tcn_causal:
            L = x_seq.size(1)
            z = z[..., -L:]

        z0 = z.transpose(1, 2).mean(dim=1)
        return self.proj(z0)

    def _f_theta_batched(
        self,
        zt1: torch.Tensor,
        s1: torch.Tensor,
        zt2: torch.Tensor,
        s2: torch.Tensor,
    ):
        if self.fuse_f_theta:
            zt = torch.cat([zt1, zt2], dim=0)
            s = torch.cat([s1, s2], dim=0)
            f = self.f_theta(torch.cat([zt, s], dim=-1))
            f1, f2 = f.chunk(2, dim=0)
            return f1, f2

        f1 = self.f_theta(torch.cat([zt1, s1], dim=-1))
        f2 = self.f_theta(torch.cat([zt2, s2], dim=-1))
        return f1, f2

    def forward(self, x_seq: torch.Tensor):
        x_seq = self._maybe_downsample(x_seq)

        B = x_seq.size(0)
        device = x_seq.device
        dtype = x_seq.dtype

        z0 = self._encode_to_z0(x_seq)

        eps1 = torch.randn_like(z0)
        eps2 = torch.randn_like(z0)
        sigma_t1 = self._sample_sigma(B, device, dtype)
        sigma_t2 = self._sample_sigma(B, device, dtype)

        zt1 = z0 + sigma_t1 * eps1
        zt2 = z0 + sigma_t2 * eps2

        f1, f2 = self._f_theta_batched(zt1, sigma_t1, zt2, sigma_t2)

        if self.consistency_mode == "two_sample":
            loss_cons = F.mse_loss(f1, f2) + self.lambda_rec * F.mse_loss(f1, z0)
        elif self.consistency_mode == "teacher_student":
            if self.teacher_on == "f2":
                teacher = f2.detach() if self.detach_teacher else f2
                student = f1
            elif self.teacher_on == "f1":
                teacher = f1.detach() if self.detach_teacher else f1
                student = f2
            else:
                raise ValueError(f"Unsupported teacher_on={self.teacher_on}")

            loss_cons = F.mse_loss(student, teacher) + self.lambda_rec * F.mse_loss(student, z0)
        else:
            raise ValueError(f"Unsupported consistency_mode={self.consistency_mode}")

        return z0, loss_cons

    @torch.inference_mode()
    def geodesic_random_walk(self, z0: torch.Tensor, steps: int = 5, step_size: float = 0.1):
        z = z0
        traj = [z]
        for _ in range(steps):
            noise = torch.randn_like(z)
            noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            z = z + step_size * noise
            traj.append(z)
        return torch.stack(traj, dim=1)


# Text Encoder
class TextEncoderBioClinicalBERT(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", output_pooling="cls", max_length=256, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.output_pooling = output_pooling
        self.max_length = max_length

    @torch.inference_mode()
    def forward(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.bert(**enc)
        hidden = out.last_hidden_state  # [B,L,H]
        if self.output_pooling == "mean":
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            emb = hidden[:, 0, :]
        return emb.detach()


# Image Encoder
class ImageEncoderDenseNetCheXpert(nn.Module):
    def __init__(self, use_xrv_first=True, custom_weight_path: Optional[str] = None, device=None, image_size=224):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_xrv = False
        self.backbone_out = 1024

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        try:
            import torchxrayvision as xrv
            if use_xrv_first:
                self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
                self.use_xrv = True
        except Exception:
            self.use_xrv = False

        if not self.use_xrv:
            tv = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = tv.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            if custom_weight_path and os.path.isfile(custom_weight_path):
                state = torch.load(custom_weight_path, map_location="cpu")
                tv.load_state_dict(state, strict=False)
        self.eval().to(self.device)

    def _load_image(self, item: Union[str, Image.Image]) -> Image.Image:
        if isinstance(item, str):
            img = Image.open(item).convert("RGB")
        elif isinstance(item, Image.Image):
            img = item.convert("RGB")
        else:
            raise ValueError("image item must be a path or PIL.Image")
        return img

    @torch.inference_mode()
    def forward(self, images: List[Union[str, Image.Image]]) -> torch.Tensor:
        batch = [self.tf(self._load_image(it)) for it in images]
        x = torch.stack(batch, dim=0).to(self.device)
        if self.use_xrv:
            feats = self.model.features(x)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1)).reshape(x.size(0), -1)
        else:
            feats = self.features(x)
            feats = F.relu(feats, inplace=True)
            feats = self.pool(feats).reshape(x.size(0), -1)
        return feats


class MultiModalPreprocessor:
    def __init__(
        self,
        numeric_dim: int,
        categorical_cardinalities: List[int],
        seq_input_dim: int,
        cat_embed_dim: int = 16,
        reg_weights: dict = None,
        prefer_gpu: bool = True,
        temporal_backbone: str = "gru", 
    ):
        device = torch.device("cuda" if torch.cuda.is_available() and prefer_gpu else "cpu")

        # Tabular encoder
        self.tab_encoder = TabularEncoder(
            numeric_dim=numeric_dim,
            categorical_cardinalities=categorical_cardinalities,
            cat_embed_dim=cat_embed_dim,
            reg_weights=reg_weights,
        ).to(device)

        # Temporal encoder
        self.temporal_encoder = TemporalEncoderCMGRW_Flex(
            input_dim=seq_input_dim, hidden_dim=128, model_type=temporal_backbone
        ).to(device)

        # Text encoder
        self.text_encoder = TextEncoderBioClinicalBERT(device=device)

        # Image encoder
        self.img_encoder = ImageEncoderDenseNetCheXpert(device=device)

        self.device = device

    # Encoders
    def encode_tabular(self, numeric_tensor: torch.Tensor, categorical_idx: List[torch.Tensor]):
        return self.tab_encoder(numeric_tensor, categorical_idx)

    def encode_temporal(self, seq_tensor: torch.Tensor):
        return self.temporal_encoder(seq_tensor)

    def encode_text(self, texts: List[str]):
        return self.text_encoder(texts)

    def encode_image(self, images: List[Union[str, Image.Image]]):
        return self.img_encoder(images)


class TextVisionAligner(nn.Module):
    def __init__(self, embed_dim=768, temperature=0.07):
        super().__init__()
        self.tau = nn.Parameter(torch.ones([]) * temperature)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.vis_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_emb, vis_emb):
        # L2 normalization
        t = F.normalize(self.text_proj(text_emb), dim=-1)
        v = F.normalize(self.vis_proj(vis_emb), dim=-1)

        logits = (v @ t.T) / self.tau.clamp(min=1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_t2v + loss_v2t)
        return loss, v, t


class QFormerLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, Q, X):
        B, M, D = Q.shape
        N = X.size(1)

        q = self.W_Q(Q).reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_K(X).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(X).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, M, D)
        Q = self.norm1(Q + out)
        Q = self.norm2(Q + self.ffn(Q))
        return Q


class QFormer3L(nn.Module):
    def __init__(self, d_model=768, n_heads=8, n_queries=8, dropout=0.1):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList([
            QFormerLayer(d_model, n_heads, dropout)
            for _ in range(3)
        ])

    def forward(self, X):
        B = X.size(0)
        Q = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            Q = layer(Q, X)
        return Q  # [B, M, D]


class MultiModalFusion_QFormer3L(nn.Module):
    def __init__(self, embed_dim=768, n_queries=8, n_heads=8, dropout=0.1):
        super().__init__()
        self.aligner = TextVisionAligner(embed_dim)
        self.qformer = QFormer3L(d_model=embed_dim, n_heads=n_heads, n_queries=n_queries, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, vis_emb, text_emb, struct_emb, return_loss=False):
        align_loss, vis_aligned, text_aligned = self.aligner(text_emb, vis_emb)

        X = torch.cat([vis_aligned, text_aligned, struct_emb], dim=1)  # [B, N, D]

        Z = self.qformer(X)  # [B, M, D]

        z_patient = self.pool(Z.transpose(1, 2)).squeeze(-1)  # [B, D]
        z_patient = self.proj(z_patient)

        if return_loss:
            return z_patient, align_loss
        return z_patient

class SemIdEmbeddingBatch(NamedTuple):
    seq: Tensor
    fut: Tensor


class SemIdEmbedder(nn.Module):
    def __init__(self, num_embeddings, sem_ids_dim, embeddings_dim) -> None:
        super().__init__()
        
        self.sem_ids_dim = sem_ids_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = sem_ids_dim*num_embeddings
        
        self.emb = nn.Embedding(
            num_embeddings=num_embeddings*self.sem_ids_dim+1,
            embedding_dim=embeddings_dim,
            padding_idx=self.padding_idx
        )
    
    def forward(self, batch: TokenizedSeqBatch) -> Tensor:
        sem_ids = batch.token_type_ids*self.num_embeddings + batch.sem_ids
        sem_ids[~batch.seq_mask] = self.padding_idx

        if batch.sem_ids_fut is not None:
            sem_ids_fut = batch.token_type_ids_fut*self.num_embeddings + batch.sem_ids_fut
            sem_ids_fut = self.emb(sem_ids_fut)
        else:
            sem_ids_fut = None
        return SemIdEmbeddingBatch(
            seq=self.emb(sem_ids),
            fut=sem_ids_fut
        ) 
    

class UserIdEmbedder(nn.Module):
    def __init__(self, num_buckets, embedding_dim) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.emb = nn.Embedding(num_buckets, embedding_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        hashed_indices = x % self.num_buckets
        return self.emb(hashed_indices)
    
BATCH_SIZE = 16

class SemanticIdTokenizer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False
    ) -> None:
        super().__init__()

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=rqvae_codebook_normalize,
            codebook_sim_vq=rqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
        )
        
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
    
    @property
    def sem_ids_dim(self):
        return self.n_layers + 1
    
    @torch.no_grad
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData) -> Tensor:
        cached_ids = None
        dedup_dim = []
        sampler = BatchSampler(
            SequentialSampler(range(len(movie_dataset))), batch_size=512, drop_last=False
        )
        dataloader = DataLoader(movie_dataset, sampler=sampler, shuffle=False, collate_fn=lambda batch: batch[0])
        for batch in dataloader:
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            # Detect in-batch duplicates
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = torch.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if cached_ids is None:
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
            dedup_dim.append(hits)
        # Concatenate new column to deduplicate ids
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        
        return self.cached_ids

    @torch.no_grad
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        prefix_length = sem_id_prefix.shape[-1]
        prefix_cache = self.cached_ids[:, :prefix_length]
        out = torch.zeros(*sem_id_prefix.shape[:-1], dtype=bool, device=sem_id_prefix.device)
        
        # Batch prefixes matching to avoid OOM. 
        batches = math.ceil(sem_id_prefix.shape[0] // BATCH_SIZE)
        for i in range(batches):
            prefixes = sem_id_prefix[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...]
            matches = (prefixes.unsqueeze(-2) == prefix_cache.unsqueeze(-3)).all(axis=-1).any(axis=-1)
            out[i*BATCH_SIZE:(i+1)*BATCH_SIZE,...] = matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1])
    
    @torch.no_grad
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            seq_mask, sem_ids_fut = None, None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
        
        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )
