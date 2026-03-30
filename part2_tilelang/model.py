"""
PyTorch autograd wrappers for TileLang kernels.
Provides nn.Module-compatible layers that use TileLang for forward/backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import numpy as np
from functools import lru_cache

from kernels import (
    make_gemm_kernel,
    make_conv2d_forward_kernel,
    make_conv2d_backward_data_kernel,
    make_conv2d_backward_weight_kernel,
)

# ============================================================
# Kernel cache — compile once per shape
# ============================================================

_kernel_cache = {}


def _get_or_compile(key, make_fn, *args, **kwargs):
    if key not in _kernel_cache:
        prim_func = make_fn(*args, **kwargs)
        _kernel_cache[key] = tilelang.compile(prim_func, out_idx=[2], target="auto")
    return _kernel_cache[key]


# ============================================================
# Conv2d with TileLang
# ============================================================

class TileLangConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        input: (N, C_in, H, W) — PyTorch NCHW
        weight: (C_out, C_in, KH, KW) — PyTorch standard
        """
        N, C_in, H, W = input.shape
        C_out, _, KH, KW = weight.shape

        # Convert to NHWC for TileLang (coalesced memory access)
        input_nhwc = input.permute(0, 2, 3, 1).contiguous().half()
        # Convert weight to HWCF: (KH, KW, C_in, C_out)
        weight_hwcf = weight.permute(2, 3, 1, 0).contiguous().half()

        OH = (H + 2 * padding - KH) // stride + 1
        OW = (W + 2 * padding - KW) // stride + 1

        # Pad dimensions to be multiples of block sizes for GEMM efficiency
        key_fwd = ('conv2d_fwd', N, C_in, H, W, C_out, KH, KW, stride, padding)
        kernel_fwd = _get_or_compile(
            key_fwd, make_conv2d_forward_kernel,
            N, C_in, H, W, C_out, KH, KW, stride=stride, padding=padding)

        output_nhwc = torch.empty(N, OH, OW, C_out, device=input.device, dtype=torch.float16)
        kernel_fwd(input_nhwc, weight_hwcf, output_nhwc)

        # Convert output back to NCHW
        output = output_nhwc.permute(0, 3, 1, 2).float()

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

        # Convert to NHWC fp16
        grad_out_nhwc = grad_output.permute(0, 2, 3, 1).contiguous().half()
        input_nhwc = input.permute(0, 2, 3, 1).contiguous().half()
        weight_hwcf = weight.permute(2, 3, 1, 0).contiguous().half()

        if ctx.needs_input_grad[0]:
            key_bd = ('conv2d_bwd_data', N, C_in, H, W, C_out, KH, KW, stride, padding)
            kernel_bd = _get_or_compile(
                key_bd, make_conv2d_backward_data_kernel,
                N, C_in, H, W, C_out, KH, KW, stride=stride, padding=padding)
            grad_input_nhwc = torch.empty(N, H, W, C_in, device=input.device, dtype=torch.float16)
            kernel_bd(grad_out_nhwc, weight_hwcf, grad_input_nhwc)
            grad_input = grad_input_nhwc.permute(0, 3, 1, 2).float()

        if ctx.needs_input_grad[1]:
            key_bw = ('conv2d_bwd_weight', N, C_in, H, W, C_out, KH, KW, stride, padding)
            kernel_bw = _get_or_compile(
                key_bw, make_conv2d_backward_weight_kernel,
                N, C_in, H, W, C_out, KH, KW, stride=stride, padding=padding)
            grad_weight_hwcf = torch.empty(KH, KW, C_in, C_out, device=input.device, dtype=torch.float16)
            kernel_bw(input_nhwc, grad_out_nhwc, grad_weight_hwcf)
            grad_weight = grad_weight_hwcf.permute(3, 2, 0, 1).float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None


class TileLangConv2d(nn.Module):
    """Drop-in replacement for nn.Conv2d using TileLang kernels."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Kaiming init
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
        """
        input: (N, in_features) or (N, *, in_features)
        weight: (out_features, in_features)
        """
        input_2d = input.reshape(-1, input.shape[-1])
        M, K = input_2d.shape
        N_out = weight.shape[0]

        input_fp16 = input_2d.contiguous().half()
        # weight^T: (K, N_out)
        weight_t = weight.t().contiguous().half()

        key = ('linear_fwd', M, N_out, K)
        kernel = _get_or_compile(key, make_gemm_kernel, M, N_out, K)

        output = torch.empty(M, N_out, device=input.device, dtype=torch.float16)
        kernel(input_fp16, weight_t, output)
        output = output.float()

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
            # grad_input = grad_output @ weight
            go_fp16 = grad_output_2d.contiguous().half()
            w_fp16 = weight.contiguous().half()
            key = ('linear_bwd_input', M, K, N_out)
            kernel = _get_or_compile(key, make_gemm_kernel, M, K, N_out)
            gi = torch.empty(M, K, device=input.device, dtype=torch.float16)
            # grad_output (M, N_out) @ weight (N_out, K) -> (M, K)
            kernel(go_fp16, w_fp16.t().contiguous(), gi)
            grad_input = gi.float().reshape(input.shape)

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output^T @ input -> (N_out, K)
            go_fp16 = grad_output_2d.t().contiguous().half()
            in_fp16 = input_2d.contiguous().half()
            key = ('linear_bwd_weight', N_out, K, M)
            kernel = _get_or_compile(key, make_gemm_kernel, N_out, K, M)
            gw = torch.empty(N_out, K, device=input.device, dtype=torch.float16)
            kernel(go_fp16, in_fp16, gw)
            grad_weight = gw.float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias


class TileLangLinear(nn.Module):
    """Drop-in replacement for nn.Linear using TileLang GEMM."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return TileLangLinearFunction.apply(x, self.weight, self.bias)


# ============================================================
# CNN Model using TileLang kernels
# ============================================================

class TileLangCNN(nn.Module):
    """
    CNN for 28x28 grayscale classification using TileLang conv/linear kernels.
    BatchNorm, ReLU, MaxPool, Dropout use PyTorch (these are memory-bound ops
    where custom kernels offer little advantage over cuDNN).

    Architecture matches Part 2 CNN:
        TileLangConv2d(1,32,3,pad=1) -> BN -> ReLU -> TileLangConv2d(32,32,3,pad=1) -> BN -> ReLU -> MaxPool -> Drop
        TileLangConv2d(32,64,3,pad=1) -> BN -> ReLU -> TileLangConv2d(64,64,3,pad=1) -> BN -> ReLU -> MaxPool -> Drop
        TileLangConv2d(64,128,3,pad=1) -> BN -> ReLU -> GAP
        TileLangLinear(128,64) -> ReLU -> Drop -> TileLangLinear(64,12)
    """

    def __init__(self, num_classes=12, dropout_rate=0.3):
        super().__init__()

        # Block 1: 28x28 -> 14x14
        self.conv1a = TileLangConv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = TileLangConv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        # Block 2: 14x14 -> 7x7
        self.conv2a = TileLangConv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = TileLangConv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        # Block 3: 7x7 -> GAP
        self.conv3 = TileLangConv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # FC
        self.fc1 = TileLangLinear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate * 1.5)
        self.fc2 = TileLangLinear(64, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # GAP
        x = x.mean(dim=[2, 3])

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x
