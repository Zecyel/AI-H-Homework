"""
PyTorch autograd wrappers for TileLang kernels.
Uses persistent autotune cache — first run benchmarks, subsequent runs are instant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kernels import (
    gemm_kernel,
    conv2d_forward,
    conv2d_backward_data,
    conv2d_backward_weight,
)
from autotune_cache import get_cached_kernel, get_gemm_configs, get_conv_configs

logger = logging.getLogger(__name__)

# In-memory cache of compiled kernels (avoid re-lookup per batch)
_compiled_kernels = {}

# Shapes known to fail — skip instantly
_failed_shapes = set()


def _strict_contiguous(tensor):
    """Ensure tensor has standard contiguous strides (handles size-1 dims)."""
    out = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    out.copy_(tensor)
    return out


_MIN_TILELANG_DIM = 16


def _use_tilelang(M, N, K):
    return M >= _MIN_TILELANG_DIM and N >= _MIN_TILELANG_DIM and K >= _MIN_TILELANG_DIM


def _fallback_gemm(A, B):
    return (A.float() @ B.float()).half()


def _get_gemm(M, N, K, A, B):
    """Get compiled GEMM kernel, using persistent cache. Returns None if cuDNN wins."""
    key = f"gemm_{M}_{N}_{K}"
    if key in _failed_shapes:
        return None
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    # cuDNN reference: plain matmul
    cudnn_fn = lambda: torch.mm(A.float(), B.float())

    compiled = get_cached_kernel(
        gemm_kernel, (M, N, K), get_gemm_configs(), key, (A, B),
        cudnn_fn=cudnn_fn)

    if compiled is None:
        _failed_shapes.add(key)
        return None
    _compiled_kernels[key] = compiled
    return compiled


def _get_conv_fwd(shape_args, data, weight, input_nchw, weight_oihw, stride, padding):
    key = "conv_fwd_" + "_".join(str(x) for x in shape_args)
    if key in _failed_shapes:
        return None
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cudnn_fn = lambda: F.conv2d(input_nchw, weight_oihw, stride=stride, padding=padding)

    compiled = get_cached_kernel(
        conv2d_forward, shape_args, get_conv_configs(), key, (data, weight),
        cudnn_fn=cudnn_fn)

    if compiled is None:
        _failed_shapes.add(key)
        return None
    _compiled_kernels[key] = compiled
    return compiled


def _get_conv_bwd_data(shape_args, grad_out, weight, grad_output_nchw, weight_oihw, stride, padding):
    key = "conv_bwd_data_" + "_".join(str(x) for x in shape_args)
    if key in _failed_shapes:
        return None
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cudnn_fn = lambda: F.conv_transpose2d(grad_output_nchw, weight_oihw, stride=stride, padding=padding)

    compiled = get_cached_kernel(
        conv2d_backward_data, shape_args, get_conv_configs(), key, (grad_out, weight),
        cudnn_fn=cudnn_fn)

    if compiled is None:
        _failed_shapes.add(key)
        return None
    _compiled_kernels[key] = compiled
    return compiled


def _get_conv_bwd_weight(shape_args, data, grad_out, input_nchw, weight_shape, grad_output_nchw, stride, padding):
    key = "conv_bwd_weight_" + "_".join(str(x) for x in shape_args)
    if key in _failed_shapes:
        return None
    if key in _compiled_kernels:
        return _compiled_kernels[key]

    cudnn_fn = lambda: torch.nn.grad.conv2d_weight(input_nchw, weight_shape, grad_output_nchw, stride=stride, padding=padding)

    compiled = get_cached_kernel(
        conv2d_backward_weight, shape_args, get_conv_configs(), key, (data, grad_out),
        cudnn_fn=cudnn_fn)

    if compiled is None:
        _failed_shapes.add(key)
        return None
    _compiled_kernels[key] = compiled
    return compiled


# ============================================================
# Conv2d with TileLang
# ============================================================

class TileLangConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        N, C_in, H, W = input.shape
        C_out, _, KH, KW = weight.shape
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1

        used_tilelang = False
        if _use_tilelang(N * OH * OW, C_out, KH * KW * C_in):
            try:
                input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
                weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())
                shape_args = (N, C_in, H, W, C_out, KH, KW, stride, padding)
                compiled = _get_conv_fwd(shape_args, input_nhwc, weight_hwcf,
                                         input, weight, stride, padding)
                if compiled is not None:
                    output_nhwc = compiled(input_nhwc, weight_hwcf)
                    output = output_nhwc.permute(0, 3, 1, 2).float()
                    used_tilelang = True
            except Exception as e:
                logger.warning(f"TileLang conv2d_forward error: {e}")
        if not used_tilelang:
            output = F.conv2d(input, weight, stride=stride, padding=padding)

        if bias is not None:
            output = output + bias.view(1, -1, 1, 1)

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        N, C_in, H, W = input.shape
        C_out, _, KH, KW = weight.shape
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            used_tilelang = False
            if _use_tilelang(N * H * W, C_in, KH * KW * C_out):
                try:
                    grad_out_nhwc = _strict_contiguous(grad_output.permute(0, 2, 3, 1).half())
                    weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())
                    shape_args = (N, C_in, H, W, C_out, KH, KW, stride, padding)
                    compiled = _get_conv_bwd_data(shape_args, grad_out_nhwc, weight_hwcf,
                                                   grad_output, weight, stride, padding)
                    if compiled is not None:
                        grad_input_nhwc = compiled(grad_out_nhwc, weight_hwcf)
                        grad_input = grad_input_nhwc.permute(0, 3, 1, 2).float()
                        used_tilelang = True
                except Exception as e:
                    logger.warning(f"TileLang conv2d_backward_data error: {e}")
            if not used_tilelang:
                grad_input = F.conv_transpose2d(
                    grad_output, weight, stride=stride, padding=padding)

        if ctx.needs_input_grad[1]:
            used_tilelang = False
            if _use_tilelang(KH * KW * C_in, C_out, N * OH * OW):
                try:
                    input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
                    grad_out_nhwc = _strict_contiguous(grad_output.permute(0, 2, 3, 1).half())
                    shape_args = (N, C_in, H, W, C_out, KH, KW, stride, padding)
                    compiled = _get_conv_bwd_weight(shape_args, input_nhwc, grad_out_nhwc,
                                                     input, weight.shape, grad_output, stride, padding)
                    if compiled is not None:
                        grad_weight_hwcf = compiled(input_nhwc, grad_out_nhwc)
                        grad_weight = grad_weight_hwcf.permute(3, 2, 0, 1).float()
                        used_tilelang = True
                except Exception as e:
                    logger.warning(f"TileLang conv2d_backward_weight error: {e}")
            if not used_tilelang:
                grad_weight = torch.nn.grad.conv2d_weight(
                    input, weight.shape, grad_output, stride=stride, padding=padding)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None


class TileLangConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return TileLangConv2dFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding)


# ============================================================
# Linear with TileLang GEMM
# ============================================================

class TileLangLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input_2d = input.reshape(-1, input.shape[-1])
        M, K = input_2d.shape
        N_out = weight.shape[0]

        input_fp16 = _strict_contiguous(input_2d.half())
        weight_t = _strict_contiguous(weight.t().half())

        used_tilelang = False
        if _use_tilelang(M, N_out, K):
            try:
                compiled = _get_gemm(M, N_out, K, input_fp16, weight_t)
                if compiled is not None:
                    output = compiled(input_fp16, weight_t).float()
                    used_tilelang = True
            except Exception as e:
                logger.warning(f"TileLang GEMM forward error: {e}")
        if not used_tilelang:
            output = (input_fp16.float() @ weight_t.float())

        if bias is not None:
            output = output + bias

        ctx.save_for_backward(input, weight, bias)
        return output.reshape(*input.shape[:-1], N_out)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        input_2d = input.reshape(-1, input.shape[-1])
        M = grad_output_2d.shape[0]
        N_out = weight.shape[0]
        K = weight.shape[1]

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            go_fp16 = _strict_contiguous(grad_output_2d.half())
            w_fp16 = _strict_contiguous(weight.half())
            used_tilelang = False
            if _use_tilelang(M, K, N_out):
                try:
                    compiled = _get_gemm(M, K, N_out, go_fp16, w_fp16)
                    if compiled is not None:
                        gi = compiled(go_fp16, w_fp16)
                        used_tilelang = True
                except Exception as e:
                    logger.warning(f"TileLang GEMM grad_input error: {e}")
            if not used_tilelang:
                gi = _fallback_gemm(go_fp16, w_fp16)
            grad_input = gi.float().reshape(input.shape)

        if ctx.needs_input_grad[1]:
            go_fp16 = _strict_contiguous(grad_output_2d.t().half())
            in_fp16 = _strict_contiguous(input_2d.half())
            used_tilelang = False
            if _use_tilelang(N_out, K, M):
                try:
                    compiled = _get_gemm(N_out, K, M, go_fp16, in_fp16)
                    if compiled is not None:
                        gw = compiled(go_fp16, in_fp16)
                        used_tilelang = True
                except Exception as e:
                    logger.warning(f"TileLang GEMM grad_weight error: {e}")
            if not used_tilelang:
                gw = _fallback_gemm(go_fp16, in_fp16)
            grad_weight = gw.float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias


class TileLangLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return TileLangLinearFunction.apply(x, self.weight, self.bias)


# ============================================================
# CNN Model
# ============================================================

class TileLangCNN(nn.Module):
    """CNN using TileLang autotuned kernels for conv2d and linear."""

    def __init__(self, num_classes=12, dropout_rate=0.3):
        super().__init__()
        self.conv1a = TileLangConv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = TileLangConv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        self.conv2a = TileLangConv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = TileLangConv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        self.conv3 = TileLangConv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = TileLangLinear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate * 1.5)
        self.fc2 = TileLangLinear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(dim=[2, 3])

        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x
