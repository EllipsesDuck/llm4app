import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_sft_loss(logits, labels, pad_id, bos_id):
    B, T, V = logits.shape

    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    loss_mask = (labels != pad_id)
    if bos_id is not None:
        loss_mask = loss_mask & (labels != bos_id)

    loss = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        reduction="none",
    ).view(B, -1)

    loss = loss * loss_mask.to(loss.dtype)

    den = loss_mask.sum().clamp_min(1).to(loss.dtype)
    return loss.sum() / den
