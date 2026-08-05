"""Mixed-precision policy for sequence training and cached rollout."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(device: torch.device, precision: str):
    """Return an accelerator autocast context while keeping master weights fp32.

    PPO probabilities, ratios, entropy, KL, returns, and advantages are cast
    explicitly to fp32 by their callers. This context only lowers eligible
    model operations. KV storage has its own independent dtype setting.
    """

    if precision == "fp32":
        return nullcontext()
    if precision not in ("fp16", "bf16"):
        raise ValueError(f"Unknown precision {precision!r}.")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(
            f"Autocast is not configured for device type {device.type!r}."
        )
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)
