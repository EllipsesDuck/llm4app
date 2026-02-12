# -*- coding: utf-8 -*-
"""
Initialization utilities aligned with TensorFlow VarianceScaling.
"""

import math
import torch
import torch.nn as nn


def variance_scaling_trunc_(tensor: torch.Tensor, scale: float = 2.0, mode: str = "fan_in"):
    """
    PyTorch equivalent of:
        tf.variance_scaling_initializer(scale=scale, mode=mode, distribution='truncated_normal')

    Args:
        tensor: weight tensor to initialize
        scale: TF scale parameter (default 2.0 for GELU/FFN blocks)
        mode: 'fan_in' | 'fan_out' | 'fan_avg'
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    variance = scale / max(1.0, denom)
    std = math.sqrt(variance)

    # TF truncated normal is clipped at ±2σ
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)


def init_ffn_weight_(tensor: torch.Tensor):
    """
    Standard initializer for RankMixer-style FFN / MoE weights.
    """
    variance_scaling_trunc_(tensor, scale=2.0, mode="fan_in")
