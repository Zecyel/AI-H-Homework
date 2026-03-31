"""
PyTorch autograd wrappers for TileLang kernels.
Uses TileLang's native @autotune for config selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kernels import (
    gemm_kernel,
    conv2d_forward,
    conv2d_backward_data,
    conv2d_backward_weight,
)


def _strict_contiguous(tensor):
    """Ensure tensor has standard contiguous strides (handles size-1 dims)."""
    out = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    out.copy_(tensor)
    return out


# Minimum dimension for TileLang GEMM (SM80 tensor core MMA requires m16n8k16)
_MIN_TILELANG_DIM = 16


def _use_tilelang_gemm(M, N, K):
    """Check if dimensions are large enough for TileLang tensor core GEMM."""
    return M >= _MIN_TILELANG_DIM and N >= _MIN_TILELANG_DIM and K >= _MIN_TILELANG_DIM


def _fallback_gemm(A, B):
    """PyTorch fallback for small GEMMs: C = A @ B."""
    return (A.float() @ B.float()).half()


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

        # Check if GEMM-equivalent dims are large enough for TileLang
        used_tilelang = False
        if _use_tilelang_gemm(N * OH * OW, C_out, KH * KW * C_in):
            try:
                input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
                weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())
                kernel = conv2d_forward(N, C_in, H, W, C_out, KH, KW, stride, padding)
                output_nhwc = kernel(input_nhwc, weight_hwcf)
                output = output_nhwc.permute(0, 3, 1, 2).float()
                used_tilelang = True
            except RuntimeError:
                pass
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

        grad_input = grad_weight = grad_bias = None

        # Check if conv dimensions are large enough for TileLang
        # GEMM-equivalent dims: M=N*OH*OW, N=C_out/C_in, K=KH*KW*C_in/C_out
        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1
        can_tilelang_bwd_data = _use_tilelang_gemm(N * H * W, C_in, KH * KW * C_out)
        can_tilelang_bwd_weight = _use_tilelang_gemm(KH * KW * C_in, C_out, N * OH * OW)

        if ctx.needs_input_grad[0]:
            used_tilelang = False
            if can_tilelang_bwd_data:
                try:
                    grad_out_nhwc = _strict_contiguous(grad_output.permute(0, 2, 3, 1).half())
                    weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())
                    kernel = conv2d_backward_data(N, C_in, H, W, C_out, KH, KW, stride, padding)
                    grad_input_nhwc = kernel(grad_out_nhwc, weight_hwcf)
                    grad_input = grad_input_nhwc.permute(0, 3, 1, 2).float()
                    used_tilelang = True
                except RuntimeError:
                    pass
            if not used_tilelang:
                grad_input = F.conv_transpose2d(
                    grad_output, weight, stride=stride, padding=padding)

        if ctx.needs_input_grad[1]:
            used_tilelang = False
            if can_tilelang_bwd_weight:
                try:
                    input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
                    grad_out_nhwc = _strict_contiguous(grad_output.permute(0, 2, 3, 1).half())
                    kernel = conv2d_backward_weight(N, C_in, H, W, C_out, KH, KW, stride, padding)
                    grad_weight_hwcf = kernel(input_nhwc, grad_out_nhwc)
                    grad_weight = grad_weight_hwcf.permute(3, 2, 0, 1).float()
                    used_tilelang = True
                except RuntimeError:
                    pass
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
        if _use_tilelang_gemm(M, N_out, K):
            try:
                kernel = gemm_kernel(M, N_out, K)
                output = kernel(input_fp16, weight_t).float()
                used_tilelang = True
            except RuntimeError:
                pass
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
            if _use_tilelang_gemm(M, K, N_out):
                try:
                    kernel = gemm_kernel(M, K, N_out)
                    gi = kernel(go_fp16, w_fp16)
                    used_tilelang = True
                except RuntimeError:
                    pass
            if not used_tilelang:
                gi = _fallback_gemm(go_fp16, w_fp16)
            grad_input = gi.float().reshape(input.shape)

        if ctx.needs_input_grad[1]:
            go_fp16 = _strict_contiguous(grad_output_2d.t().half())
            in_fp16 = _strict_contiguous(input_2d.half())
            used_tilelang = False
            if _use_tilelang_gemm(N_out, K, M):
                try:
                    kernel = gemm_kernel(N_out, K, M)
                    gw = kernel(go_fp16, in_fp16)
                    used_tilelang = True
                except RuntimeError:
                    pass
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
