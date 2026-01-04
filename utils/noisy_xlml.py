# utils/noisy_xlml.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import types


def _wrap_module_forward_add_noise(m: nn.Module, sigma: float, flag_attr: str):
    """
    给任意 nn.Module 打 monkey-patch：forward 输出 + N(0, sigma^2)
    只 wrap 一次，后续只更新 sigma。
    """
    sigma = float(sigma)

    # 已经 wrap 过：只更新 sigma
    if getattr(m, flag_attr, False):
        setattr(m, f"{flag_attr}_sigma", sigma)
        return

    orig_forward = m.forward

    def noisy_forward(self, *args, _orig=orig_forward, **kwargs):
        out = _orig(*args, **kwargs)
        s = getattr(self, f"{flag_attr}_sigma", 0.0)
        if s and s > 0.0:
            out = out + torch.randn_like(out) * s
        return out

    m.forward = types.MethodType(noisy_forward, m)
    setattr(m, flag_attr, True)
    setattr(m, f"{flag_attr}_sigma", sigma)


def add_noise_to_xlmr_ffn_activations(model: nn.Module, sigma: float) -> int:
    """
    ✅ 只对 XLM-R / RoBERTa 的 FFN(MLP) 激活注入噪声（推荐）

    修复点：
      - callable intermediate_act_fn 不再重复嵌套 noisy(noisy(...))。
        第一次保存原始函数到 m._orig_intermediate_act_fn，以后永远基于原始函数重建 closure。
      - nn.Module 激活：只 wrap 一次，后续更新 sigma。
      - 兜底 generic activation：只 wrap 一次，后续更新 sigma。

    返回 patched_count：本次调用中“更新/设置噪声注入”的数量（便于 sanity check）
    """
    sigma = float(sigma)
    patched = 0

    def make_noisy_fn(orig_fn):
        # 注意：这里闭包捕获的是“本次调用的 sigma”
        def noisy_fn(x):
            y = orig_fn(x)
            if sigma > 0.0:
                y = y + torch.randn_like(y) * sigma
            return y
        return noisy_fn

    for m in model.modules():
        # --------------------------
        # 1) 优先处理 RobertaIntermediate 的 intermediate_act_fn
        # --------------------------
        if hasattr(m, "intermediate_act_fn"):
            act = getattr(m, "intermediate_act_fn", None)
            if act is None:
                continue

            # 保存原始激活（只保存一次！）
            # 这样后续更新 sigma 时不会 noisy(noisy(...))
            if not hasattr(m, "_orig_intermediate_act_fn"):
                m._orig_intermediate_act_fn = act

            orig_act = m._orig_intermediate_act_fn

            # (a) callable function (非 nn.Module)
            if callable(orig_act) and not isinstance(orig_act, nn.Module):
                # 每次调用都用“原始函数”重建 closure 来更新 sigma
                m.intermediate_act_fn = make_noisy_fn(orig_act)
                m._noisy_intermediate_act_wrapped = True
                m._noisy_intermediate_act_sigma = sigma
                patched += 1

            # (b) nn.Module activation
            elif isinstance(orig_act, nn.Module):
                _wrap_module_forward_add_noise(orig_act, sigma, "_noisy_act_module_wrapped")
                patched += 1

            # (c) 极端情况：不可调用/未知类型 -> 不处理
            else:
                pass

        # --------------------------
        # 2) 兜底：对看起来像激活的模块做 patch（保守）
        #    注意：这里可能会命中 FFN 外的激活，但你原实现就这样做的；
        #    如果你想严格只改 FFN，建议把这段关掉。
        # --------------------------
        # cls = m.__class__.__name__
        # if any(k in cls for k in ["GELUActivation", "SiLU", "Activation"]):
        #     _wrap_module_forward_add_noise(m, sigma, "_noisy_generic_act_wrapped")
        #     patched += 1

    print(f"[NoiseInject] Patched/updated XLM-R FFN activations: {patched}, sigma={sigma}")
    return patched
