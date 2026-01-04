# utils/noisy_qwen.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


# ==========================
# 工具函数：激活函数 + 噪声
# ==========================
class NoisyGELU(nn.GELU):
    """
    y = GELU(x) + N(0, sigma^2)
    """
    def __init__(self, sigma: float, approximate: str = "none"):
        super().__init__(approximate=approximate)
        self.sigma = sigma

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.sigma > 0:
            out = out + torch.randn_like(out) * self.sigma
        return out


class NoisySiLU(nn.SiLU):
    """
    y = SiLU(x) + N(0, sigma^2)
    """
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = super().forward(input)
        if self.sigma > 0:
            out = out + torch.randn_like(out) * self.sigma
        return out


def make_noisy_function(fn, sigma: float):
    """
    用于替换类似 Qwen2MLP 中的 self.act_fn（可能是 silu / swiglu / gelu 等）
    返回：noisy 版本：fn(x) + N(0, sigma^2)
    """
    def noisy_fn(x):
        out = fn(x)
        if sigma > 0:
            out = out + torch.randn_like(out) * sigma
        return out
    return noisy_fn


# ==========================
# MLP-only 注入逻辑
# ==========================
def _is_mlp_like_module(module_name: str, module: nn.Module) -> bool:
    """
    用“保守但通用”的规则识别 FFN/MLP 子模块：
    - 名字包含: mlp / ffn / feed_forward / feedforward
    - 或类名包含: MLP / FFN / FeedForward
    """
    n = (module_name or "").lower()
    cls = module.__class__.__name__.lower()

    name_hit = any(k in n for k in ["mlp", "ffn", "feed_forward", "feedforward"])
    cls_hit = any(k in cls for k in ["mlp", "ffn", "feedforward"])

    return name_hit or cls_hit


def _patch_activations_inside(module: nn.Module, sigma: float):
    """
    在给定 module 的子树内部做你原本的两件事：
    1) 替换 nn.GELU / nn.SiLU -> NoisyGELU / NoisySiLU
    2) 替换函数式 act_fn -> noisy wrapper（只 wrap 一次）
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU):
            setattr(module, name, NoisyGELU(sigma=sigma, approximate=getattr(child, "approximate", "none")))
        elif isinstance(child, nn.SiLU):
            setattr(module, name, NoisySiLU(sigma=sigma))
        else:
            _patch_activations_inside(child, sigma)

    act_fn = getattr(module, "act_fn", None)
    if callable(act_fn) and not isinstance(act_fn, nn.Module):
        if not getattr(module, "_noisy_act_wrapped", False):
            module.act_fn = make_noisy_function(act_fn, sigma)
            module._noisy_act_wrapped = True


def add_noise_to_qwen_mlp_activations(model: nn.Module, sigma: float):
    """
    只对模型中“MLP/FFN 子模块”内部的激活做替换/包裹：
    - 替换 nn.GELU / nn.SiLU
    - 替换函数式 act_fn

    其余结构（包括 attention）保持不变。
    """
    for name, child in list(model.named_modules()):
        # 注意：named_modules() 会包含自身；我们只在命中 MLP 的节点上 patch 它的内部
        if _is_mlp_like_module(name, child):
            _patch_activations_inside(child, sigma)
