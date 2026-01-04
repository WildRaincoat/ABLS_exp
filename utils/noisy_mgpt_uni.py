# utils/noisy_mgpt.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2MLP


# ==========================
# ✅ Noisy activation wrapper
# ==========================
class NoisyActivation(nn.Module):
    """
    Wrap an arbitrary activation callable (function or nn.Module).
    Forward behavior:
        y = act(x) + U(-sigma, sigma)

    Note:
    - Noise is injected even in model.eval()
      (intended for Monte Carlo / sanity-check experiments).
    """
    def __init__(self, base_act, sigma: float):
        super().__init__()
        self.base_act = base_act
        self.sigma = float(sigma)

    def forward(self, x):
        y = self.base_act(x)
        if self.sigma == 0.0:
            return y
        # ✅ Uniform noise in [-sigma, +sigma]
        noise = (torch.rand_like(y) * 2.0 - 1.0) * self.sigma
        return y + noise


# ==========================
# ✅ Inject noise into GPT2MLP
# ==========================
def add_activation_noise_to_gpt2_mlp(model: nn.Module, sigma: float):
    """
    Traverse the model and replace GPT2MLP.act with NoisyActivation.

    If a GPT2MLP has already been patched, only update sigma.

    Args:
        model: HuggingFace GPT2 / mGPT model
        sigma: half-width of uniform noise
    """
    patched = 0

    for m in model.modules():
        if isinstance(m, GPT2MLP):
            if isinstance(m.act, NoisyActivation):
                # already wrapped → just update sigma
                m.act.sigma = float(sigma)
            else:
                m.act = NoisyActivation(m.act, sigma)
            patched += 1

    print(f"[NoiseInject] Patched GPT2MLP blocks: {patched}, sigma={sigma}")
