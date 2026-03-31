"""
DLPack bridge: CuPy <-> TileLang (which uses torch internally).
All TileLang kernel calls go through this module.
No torch imports leak into user code.
"""

import torch
import cupy as cp

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kernels import (
    gemm_kernel,
    conv2d_forward,
    conv2d_backward_data,
    conv2d_backward_weight,
)

# Compiled kernel cache (keyed by shape tuple)
_kernel_cache = {}


def _to_torch(x):
    """CuPy array -> torch.Tensor (zero-copy via DLPack)."""
    return torch.from_dlpack(x)


def _to_cupy(x):
    """torch.Tensor -> CuPy array (zero-copy via DLPack)."""
    return cp.from_dlpack(x)


def _ensure_contiguous_torch(cp_array):
    """Convert CuPy fp16 array to contiguous torch tensor."""
    t = _to_torch(cp.ascontiguousarray(cp_array))
    # Ensure truly contiguous strides (handles size-1 dims)
    out = torch.empty(t.shape, device=t.device, dtype=t.dtype)
    out.copy_(t)
    return out


# ============================================================
# GEMM: C = A @ B
# ============================================================

def tilelang_gemm(A_cp, B_cp):
    """
    GEMM using TileLang. A: (M, K), B: (K, N) -> C: (M, N).
    Inputs/outputs are CuPy float16 arrays.
    Pads dimensions to 16 if needed.
    """
    M, K = A_cp.shape
    _, N = B_cp.shape

    # Pad to meet tensor core minimum (16)
    pad_m = (16 - M % 16) % 16 if M < 16 else 0
    pad_n = (16 - N % 16) % 16 if N < 16 else 0
    pad_k = (16 - K % 16) % 16 if K < 16 else 0

    if pad_m or pad_k:
        A_cp = cp.pad(A_cp, ((0, pad_m), (0, pad_k)), mode='constant')
    if pad_k or pad_n:
        B_cp = cp.pad(B_cp, ((0, pad_k), (0, pad_n)), mode='constant')

    Mp, Kp = A_cp.shape
    _, Np = B_cp.shape

    key = ('gemm', Mp, Np, Kp)
    if key not in _kernel_cache:
        _kernel_cache[key] = gemm_kernel(Mp, Np, Kp)

    A_t = _ensure_contiguous_torch(A_cp)
    B_t = _ensure_contiguous_torch(B_cp)
    C_t = _kernel_cache[key](A_t, B_t)
    C_cp = _to_cupy(C_t)

    # Slice back to original dims
    if pad_m or pad_n:
        C_cp = C_cp[:M, :N]
    return C_cp


# ============================================================
# Conv2d forward
# ============================================================

def tilelang_conv_fwd(data_cp, weight_cp, N, C_in, H, W, C_out, KH, KW, stride, padding):
    """
    Conv2d forward using TileLang.
    data_cp: (N, H, W, C_in) float16
    weight_cp: (KH, KW, C_in, C_out) float16
    Returns: (N, OH, OW, C_out) float16
    """
    key = ('conv_fwd', N, C_in, H, W, C_out, KH, KW, stride, padding)
    if key not in _kernel_cache:
        _kernel_cache[key] = conv2d_forward(N, C_in, H, W, C_out, KH, KW, stride, padding)

    data_t = _ensure_contiguous_torch(data_cp)
    weight_t = _ensure_contiguous_torch(weight_cp)
    out_t = _kernel_cache[key](data_t, weight_t)
    return _to_cupy(out_t)


# ============================================================
# Conv2d backward data
# ============================================================

def tilelang_conv_bwd_data(grad_out_cp, weight_cp, N, C_in, H, W, C_out, KH, KW, stride, padding):
    """
    Conv2d backward data using TileLang.
    grad_out_cp: (N, OH, OW, C_out) float16
    weight_cp: (KH, KW, C_in, C_out) float16
    Returns: (N, H, W, C_in) float16
    """
    key = ('conv_bwd_data', N, C_in, H, W, C_out, KH, KW, stride, padding)
    if key not in _kernel_cache:
        _kernel_cache[key] = conv2d_backward_data(N, C_in, H, W, C_out, KH, KW, stride, padding)

    go_t = _ensure_contiguous_torch(grad_out_cp)
    w_t = _ensure_contiguous_torch(weight_cp)
    gi_t = _kernel_cache[key](go_t, w_t)
    return _to_cupy(gi_t)


# ============================================================
# Conv2d backward weight
# ============================================================

def tilelang_conv_bwd_weight(data_cp, grad_out_cp, N, C_in, H, W, C_out, KH, KW, stride, padding):
    """
    Conv2d backward weight using TileLang.
    data_cp: (N, H, W, C_in) float16
    grad_out_cp: (N, OH, OW, C_out) float16
    Returns: (KH, KW, C_in, C_out) float16
    """
    key = ('conv_bwd_weight', N, C_in, H, W, C_out, KH, KW, stride, padding)
    if key not in _kernel_cache:
        _kernel_cache[key] = conv2d_backward_weight(N, C_in, H, W, C_out, KH, KW, stride, padding)

    data_t = _ensure_contiguous_torch(data_cp)
    go_t = _ensure_contiguous_torch(grad_out_cp)
    gw_t = _kernel_cache[key](data_t, go_t)
    return _to_cupy(gw_t)
