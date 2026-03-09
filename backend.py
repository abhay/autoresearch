"""
Unified backend abstraction for autoresearch.
Handles device detection, dtype, autocast, synchronization, memory,
seeding, compilation, peak FLOPS, and attention (FA3 on CUDA, SDPA on MPS/CPU).
"""

import os
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Device detection: CUDA > MPS > CPU, overridable via env var
# ---------------------------------------------------------------------------

_override = os.environ.get("AUTORESEARCH_BACKEND", "").lower()

if _override:
    assert _override in ("cuda", "mps", "cpu"), (
        f"AUTORESEARCH_BACKEND must be cuda, mps, or cpu, got: {_override!r}"
    )
    device_type: str = _override
elif torch.cuda.is_available():
    device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device_type = "mps"
else:
    device_type = "cpu"

device: torch.device = torch.device(device_type)

dtype: torch.dtype = torch.bfloat16

# ---------------------------------------------------------------------------
# CUDA-specific env vars (before any CUDA allocation)
# ---------------------------------------------------------------------------

if device_type == "cuda":
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Peak FLOPS for MFU calculation
# ---------------------------------------------------------------------------

if device_type == "cuda":
    peak_flops: float = 989.5e12   # H100 bf16
elif device_type == "mps":
    peak_flops = 14.0e12           # conservative default
else:
    peak_flops = 1.0e12            # CPU placeholder

# ---------------------------------------------------------------------------
# Attention: FA3 on CUDA, SDPA with sliding window on MPS/CPU
# ---------------------------------------------------------------------------

_fa3 = None

if device_type == "cuda":
    from kernels import get_kernel
    cap = torch.cuda.get_device_capability()
    # varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
    _repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    _fa3 = get_kernel(_repo).flash_attn_interface

# Sliding window mask cache: (T, window_size, device) -> bool tensor
_sliding_window_mask_cache: dict = {}


def _get_sliding_window_mask(T: int, window_size: int, device: torch.device) -> torch.Tensor:
    key = (T, window_size, device)
    if key not in _sliding_window_mask_cache:
        rows = torch.arange(T, device=device).unsqueeze(1)
        cols = torch.arange(T, device=device).unsqueeze(0)
        causal = cols <= rows
        in_window = (rows - cols) < window_size
        _sliding_window_mask_cache[key] = causal & in_window  # (T, T) bool
    return _sliding_window_mask_cache[key]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, window_size: tuple = (-1, -1)) -> torch.Tensor:
    """
    Attention dispatch.
    q, k, v: (B, T, H, D)
    window_size: tuple (left, right) as used by FA3.

    CUDA: FA3 flash_attn_func
    MPS/CPU: F.scaled_dot_product_attention with manual sliding window mask
    """
    if device_type == "cuda":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # MPS / CPU path: SDPA expects (B, H, T, D)
    B, T, H, D = q.shape
    q = q.transpose(1, 2)  # (B, H, T, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    w = window_size[0] if isinstance(window_size, (tuple, list)) else window_size

    if w <= 0 or w >= T:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        mask = _get_sliding_window_mask(T, w, q.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Back to (B, T, H, D)
    y = y.transpose(1, 2)
    return y


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def autocast():
    """Return an AMP autocast context manager for the detected backend."""
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def synchronize():
    """Device synchronization barrier."""
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def peak_memory_mb() -> float:
    """Peak memory allocated in MB."""
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device_type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    return 0.0


def manual_seed(seed: int):
    """Seed torch and device-specific RNG."""
    torch.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed(seed)


def compile_model(model):
    """torch.compile the model. Skipped on MPS (inductor overhead exceeds fusion gains)."""
    if device_type == "mps":
        return model
    return torch.compile(model, dynamic=False)


def maybe_compile(fn):
    """
    torch.compile with fullgraph on CUDA, identity on MPS/CPU.
    Use as a decorator or wrapper for fused optimizer steps etc.
    """
    if device_type == "cuda":
        return torch.compile(fn, dynamic=False, fullgraph=True)
    return fn
