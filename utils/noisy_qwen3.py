# utils/noisy_qwen.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# =========================================================
# Dynamic sigma controller
# =========================================================
@dataclass
class NoiseController:
    sigma: float = 0.0
    enabled: bool = True

    def set_sigma(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)


# =========================================================
# Identify Qwen3/Qwen2-like MLP blocks
# =========================================================
def _is_mlp_like_module(module_name: str, module: nn.Module) -> bool:
    """
    Conservative matcher:
      - name contains mlp/ffn/feed_forward/feedforward
      - class name contains MLP/FFN/FeedForward
    """
    n = (module_name or "").lower()
    cls = module.__class__.__name__.lower()
    name_hit = any(k in n for k in ["mlp", "ffn", "feed_forward", "feedforward"])
    cls_hit = any(k in cls for k in ["mlp", "ffn", "feedforward"])
    return name_hit or cls_hit


def _has_qwen_style_ffn_fields(m: nn.Module) -> bool:
    """
    Qwen-style FFN (Qwen2/Qwen3 frequently):
      gate_proj, up_proj, down_proj, act_fn (callable)
    Some variants use w1/w2/w3 naming; we handle both.
    """
    if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj"):
        return True
    # alternate naming seen in some LLMs
    if hasattr(m, "w1") and hasattr(m, "w2") and hasattr(m, "w3"):
        return True
    return False


def _get_ffn_parts(m: nn.Module):
    """
    Return (gate_proj, up_proj, down_proj, act_fn) by probing common field names.
    """
    if hasattr(m, "gate_proj") and hasattr(m, "up_proj") and hasattr(m, "down_proj"):
        gate = getattr(m, "gate_proj")
        up = getattr(m, "up_proj")
        down = getattr(m, "down_proj")
    else:
        # fallback
        gate = getattr(m, "w1")
        up = getattr(m, "w3")
        down = getattr(m, "w2")

    act_fn = getattr(m, "act_fn", None)
    if act_fn is None:
        # some models use activation_fn
        act_fn = getattr(m, "activation_fn", None)

    return gate, up, down, act_fn


# =========================================================
# Patch logic
# =========================================================
def attach_noisy_qwen_mlp_activation_patch(
    model: nn.Module,
    controller: Optional[NoiseController] = None,
    verbose: bool = True,
) -> Tuple[NoiseController, Dict[str, callable]]:
    """
    Patch MLP/FFN forward to inject noise RIGHT AFTER activation:
        act = act_fn(gate_proj(x))
        act += N(0, sigma^2)
        out = down_proj(act * up_proj(x))

    Works even when activation is functional/fused (no nn.SiLU module present).
    Compatible with int8 (bnb) and cpu-offload because we do not touch weights.

    Returns:
      controller, restore_map (name -> original_forward)
    """
    if controller is None:
        controller = NoiseController(sigma=0.0, enabled=True)

    restore: Dict[str, callable] = {}
    patched = 0
    skipped = 0

    for name, m in model.named_modules():
        if not _is_mlp_like_module(name, m):
            continue
        if not _has_qwen_style_ffn_fields(m):
            skipped += 1
            continue

        gate, up, down, act_fn = _get_ffn_parts(m)
        if gate is None or up is None or down is None or (not callable(act_fn)):
            skipped += 1
            continue

        # prevent double patch
        if getattr(m, "_noisy_mlp_patched", False):
            continue

        orig_forward = m.forward
        restore[name] = orig_forward

        def make_forward(orig_fwd, gate_proj, up_proj, down_proj, act, ctrl: NoiseController):
            # wrapper keeps signature flexible
            def forward_patched(*args, **kwargs):
                # Most HF MLP forward is forward(hidden_states)
                if len(args) >= 1:
                    x = args[0]
                    rest_args = args[1:]
                else:
                    x = kwargs.get("hidden_states", None)
                    rest_args = ()
                if x is None:
                    # fallback to original if we can't find input tensor
                    return orig_fwd(*args, **kwargs)

                # Qwen-style FFN
                gate_out = gate_proj(x)
                up_out = up_proj(x)

                act_out = act(gate_out)

                if ctrl.enabled and ctrl.sigma > 0:
                    act_out = act_out + torch.randn_like(act_out) * ctrl.sigma

                out = down_proj(act_out * up_out)

                # Preserve extra returns if original forward had them (rare for MLP)
                return out

            return forward_patched

        m.forward = make_forward(orig_forward, gate, up, down, act_fn, controller)
        setattr(m, "_noisy_mlp_patched", True)
        patched += 1

    if verbose:
        print(f"[NoisyQwen] Patched MLP blocks: {patched}, skipped_mlp_like: {skipped}")

    return controller, restore


def restore_patched_mlps(model: nn.Module, restore_map: Dict[str, callable]) -> None:
    """
    Restore original forward methods for patched MLP modules.
    """
    # We need to walk again to find modules by name
    name_to_module = dict(model.named_modules())
    for name, orig_forward in restore_map.items():
        m = name_to_module.get(name, None)
        if m is None:
            continue
        try:
            m.forward = orig_forward
            setattr(m, "_noisy_mlp_patched", False)
        except Exception:
            pass
