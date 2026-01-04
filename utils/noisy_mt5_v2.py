# utils/noisy_mt5.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# ==========================
# ✅ Noisy activation wrapper
# ==========================
class NoisyActivation(nn.Module):
    """
    Wrap an arbitrary activation callable (function or nn.Module).
    Forward behavior:
        y = act(x) + N(0, sigma^2)

    Note:
    - Noise is injected even in model.eval()
      (intended for Monte Carlo / escape-noise experiments).
    """
    def __init__(self, base_act, sigma: float):
        super().__init__()
        self.base_act = base_act
        self.sigma = float(sigma)

    def forward(self, x):
        y = self.base_act(x)
        if self.sigma == 0.0:
            return y
        return y + torch.randn_like(y) * self.sigma


def _is_t5_ffn_module(m: nn.Module) -> bool:
    """
    兼容不同 transformers 版本的 T5/mT5 FFN 实现。
    优先用类名判断，避免 import 路径差异导致 isinstance 失效。
    """
    cls = m.__class__.__name__
    return cls in {
        # 常见（当前主流 transformers）
        "T5DenseActDense",
        "T5DenseGatedActDense",
        # 一些版本/分支可能出现的命名
        "MT5DenseActDense",
        "MT5DenseGatedActDense",
        # 你旧版里写过的（保险起见保留）
        "T5DenseReluDense",
        "T5DenseGatedGeluDense",
    }


def add_activation_noise_to_t5_ffn(model: nn.Module, sigma: float) -> int:
    """
    Traverse the model and replace T5/mT5 FFN activation (module.act) with NoisyActivation.

    If a FFN has already been patched, only update sigma.

    This implements "noise on activation output" (参照 noisy_mgpt 的做法),
    NOT "noise on FFN block output".

    Args:
        model: HuggingFace T5 / mT5 model (AutoModelForSeq2SeqLM)
        sigma: std of Gaussian noise
    Returns:
        number of patched FFN modules.
    """
    sigma = float(sigma)
    patched = 0
    skipped_no_act = 0
    skipped_not_callable = 0

    for m in model.modules():
        if not _is_t5_ffn_module(m):
            continue

        if not hasattr(m, "act"):
            skipped_no_act += 1
            continue

        base_act = getattr(m, "act")

        # 已经 wrap 过：只更新 sigma
        if isinstance(base_act, NoisyActivation):
            base_act.sigma = sigma
            patched += 1
            continue

        # act 可能是 function / nn.Module / callable
        if not callable(base_act):
            skipped_not_callable += 1
            continue

        setattr(m, "act", NoisyActivation(base_act, sigma))
        patched += 1

    print(
        f"[NoiseInject] Patched T5/mT5 FFN activations: {patched}, sigma={sigma} "
        f"(skipped_no_act={skipped_no_act}, skipped_not_callable={skipped_not_callable})"
    )
    return patched
