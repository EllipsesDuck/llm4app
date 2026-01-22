import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict


class ImageSemanticModel(nn.Module):
    def __init__(
        self,
        emb_dim: int = 512,
        num_labels: int = 8,
        backbone: str = "densenet121",
        pretrained: bool = True,
        normalize: bool = True,
        enable_heads: bool = True,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_labels = num_labels
        self.normalize = normalize
        self.enable_heads = enable_heads

        self.backbone, feat_dim = _build_backbone(
            backbone=backbone,
            pretrained=pretrained,
        )

        self.proj = nn.Linear(feat_dim, emb_dim)

        if enable_heads:
            self.head_any = nn.Linear(emb_dim, 1)
            self.head_multi = nn.Linear(emb_dim, num_labels)
        else:
            self.head_any = None
            self.head_multi = None

    @staticmethod
    def _build_backbone(
        backbone: str,
        pretrained: bool,
    ) -> tuple[nn.Module, int]:
        """
        Build backbone and return (model, feat_dim)
        """
        if backbone == "densenet121":
            net = models.densenet121(pretrained=pretrained)
            net.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = net.classifier.in_features
            net.classifier = nn.Identity()
            return net, feat_dim

        elif backbone == "densenet169":
            net = models.densenet169(pretrained=pretrained)
            net.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = net.classifier.in_features
            net.classifier = nn.Identity()
            return net, feat_dim

        elif backbone == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            net.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feat_dim

        elif backbone == "resnet101":
            net = models.resnet101(pretrained=pretrained)
            net.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feat_dim

        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(pretrained=pretrained)
            net.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            feat_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feat_dim

        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone}")


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image into semantic embedding

        Args:
            x: (B, 1, H, W)

        Returns:
            z: (B, emb_dim)
        """
        feat = self.backbone(x)
        z = self.proj(feat)

        if self.normalize:
            z = F.normalize(z, dim=-1)

        return z
    
    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = True,
        return_logits: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward

        Returns dict:
            - embedding
            - logit_any
            - logit_multi
        """
        out = {}

        z = self.encode(x)

        if return_embedding:
            out["embedding"] = z

        if self.enable_heads and return_logits:
            out["logit_any"] = self.head_any(z).squeeze(-1)
            out["logit_multi"] = self.head_multi(z)

        return out
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        y_any: Optional[torch.Tensor] = None,
        y_multi: Optional[torch.Tensor] = None,
        weight_any: float = 1.0,
        weight_multi: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified loss computation
        """
        losses = {}
        total = 0.0

        if y_any is not None:
            loss_any = F.binary_cross_entropy_with_logits(
                outputs["logit_any"], y_any
            )
            losses["loss_any"] = loss_any
            total += weight_any * loss_any

        if y_multi is not None:
            loss_multi = F.binary_cross_entropy_with_logits(
                outputs["logit_multi"], y_multi
            )
            losses["loss_multi"] = loss_multi
            total += weight_multi * loss_multi

        losses["loss_total"] = total
        return losses
