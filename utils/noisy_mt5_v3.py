# utils/noisy_mt5.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import types


def _is_ffn_module(m: nn.Module) -> bool:
    """
    兼容不同 transformers 版本的 T5/mT5 FFN 实现。
    优先用类名判断，避免 import 路径差异导致 isinstance 失效。
    """
    cls = m.__class__.__name__
    return cls in {
        "T5DenseActDense",
        "T5DenseGatedActDense",
        "MT5DenseActDense",
        "MT5DenseGatedActDense",
    }


def add_activation_noise_to_t5_ffn(model: nn.Module, sigma: float) -> int:
    """
    Inject Gaussian noise into T5/mT5 FFN output by patching FFN forward(),
    but ONLY for decoder FFN modules.

        out = original_forward(...)
        out = out + N(0, sigma^2)   (when sigma > 0)

    Returns: number of patched FFN modules.
    """
    sigma = float(sigma)
    patched = 0

    # === 关键改动：只遍历 decoder ===
    if not hasattr(model, "decoder"):
        raise ValueError("Model does not have a decoder attribute (not a T5/mT5-style model).")

    for m in model.decoder.modules():
        if not _is_ffn_module(m):
            continue

        # already patched -> just update sigma
        if getattr(m, "_noisy_ffn_patched", False):
            m._noisy_ffn_sigma = sigma
            patched += 1
            continue

        orig_forward = m.forward  # capture original forward

        def noisy_forward(self, *args, _orig_forward=orig_forward, **kwargs):
            out = _orig_forward(*args, **kwargs)
            s = getattr(self, "_noisy_ffn_sigma", 0.0)
            if s and s > 0.0:
                out = out + torch.randn_like(out) * s
            return out

        m.forward = types.MethodType(noisy_forward, m)
        m._noisy_ffn_patched = True
        m._noisy_ffn_sigma = sigma
        patched += 1

    print(f"[NoiseInject] Patched T5/mT5 *decoder* FFN blocks: {patched}, sigma={sigma}")
    return patched
