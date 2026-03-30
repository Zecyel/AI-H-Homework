"""
PyTorch autograd wrappers for TileLang kernels.
Loads optimal tile configurations from autotune cache.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from kernels import (
    make_gemm_kernel,
    make_conv2d_forward_kernel,
    make_conv2d_backward_data_kernel,
    make_conv2d_backward_weight_kernel,
)
from autotune import lookup_or_default, get_cache_key, load_cache

# ============================================================
# Kernel cache — compile once per (shape, config)
# ============================================================

_kernel_cache = {}
_autotune_cache = None


def _load_autotune():
    global _autotune_cache
    if _autotune_cache is None:
        _autotune_cache = load_cache()
        if _autotune_cache:
            print(f"[TileLang] Loaded {len(_autotune_cache)} autotuned kernel configs")
        else:
            print("[TileLang] WARNING: No autotune cache found. Run `python autotune.py` first.")
            print("[TileLang] Using conservative default configs (will be slower).")
    return _autotune_cache


def _get_config(name, shape_args):
    """Get the best config for a kernel from autotune cache."""
    cache = _load_autotune()
    key = get_cache_key(name, shape_args)
    if key in cache:
        return cache[key]["config"]
    return {
        "block_M": 32, "block_N": 32, "block_K": 16,
        "num_stages": 2, "threads": 128,
    }


def _compile_kernel(key, make_fn, shape_args, config):
    """Compile kernel with specific config, caching the result."""
    if key not in _kernel_cache:
        full_args = list(shape_args) + [
            config["block_M"], config["block_N"], config["block_K"],
            config["num_stages"], config["threads"],
        ]
        prim_func = make_fn(*full_args)
        _kernel_cache[key] = tilelang.compile(prim_func, out_idx=[2], target="auto")
    return _kernel_cache[key]


def _strict_contiguous(tensor):
    """Ensure tensor has standard contiguous strides (handles size-1 dims)."""
    out = torch.empty(tensor.shape, device=tensor.device, dtype=tensor.dtype)
    out.copy_(tensor)
    return out


# ============================================================
# Conv2d with TileLang
# ============================================================

class TileLangConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        N, C_in, H, W = input.shape
        C_out, _, KH, KW = weight.shape

        input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
        weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())

        shape_args = (N, C_in, H, W, C_out, KH, KW, stride, padding)
        config = _get_config("conv2d_fwd", shape_args)
        key = ('conv2d_fwd', *shape_args, tuple(sorted(config.items())))
        kernel = _compile_kernel(key, make_conv2d_forward_kernel, shape_args, config)

        output_nhwc = kernel(input_nhwc, weight_hwcf)
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

        grad_input = grad_weight = grad_bias = None

        grad_out_nhwc = _strict_contiguous(grad_output.permute(0, 2, 3, 1).half())
        input_nhwc = _strict_contiguous(input.permute(0, 2, 3, 1).half())
        weight_hwcf = _strict_contiguous(weight.permute(2, 3, 1, 0).half())

        conv_shape = (N, C_in, H, W, C_out, KH, KW, stride, padding)

        if ctx.needs_input_grad[0]:
            config = _get_config("conv2d_bwd_data", conv_shape)
            key = ('conv2d_bwd_data', *conv_shape, tuple(sorted(config.items())))
            kernel = _compile_kernel(key, make_conv2d_backward_data_kernel, conv_shape, config)
            grad_input_nhwc = kernel(grad_out_nhwc, weight_hwcf)
            grad_input = grad_input_nhwc.permute(0, 3, 1, 2).float()

        if ctx.needs_input_grad[1]:
            config = _get_config("conv2d_bwd_weight", conv_shape)
            key = ('conv2d_bwd_weight', *conv_shape, tuple(sorted(config.items())))
            kernel = _compile_kernel(key, make_conv2d_backward_weight_kernel, conv_shape, config)
            grad_weight_hwcf = kernel(input_nhwc, grad_out_nhwc)
            grad_weight = grad_weight_hwcf.permute(3, 2, 0, 1).float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None


class TileLangConv2d(nn.Module):
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

        shape_args = (M, N_out, K)
        config = _get_config("gemm_fwd", shape_args)
        key = ('gemm_fwd', *shape_args, tuple(sorted(config.items())))
        kernel = _compile_kernel(key, make_gemm_kernel, shape_args, config)

        output = kernel(input_fp16, weight_t).float()

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
            shape_args = (M, K, N_out)
            config = _get_config("gemm_bwd_input", shape_args)
            key = ('gemm_bwd_input', *shape_args, tuple(sorted(config.items())))
            kernel = _compile_kernel(key, make_gemm_kernel, shape_args, config)
            gi = kernel(go_fp16, w_fp16)
            grad_input = gi.float().reshape(input.shape)

        if ctx.needs_input_grad[1]:
            go_fp16 = _strict_contiguous(grad_output_2d.t().half())
            in_fp16 = _strict_contiguous(input_2d.half())
            shape_args = (N_out, K, M)
            config = _get_config("gemm_bwd_weight", shape_args)
            key = ('gemm_bwd_weight', *shape_args, tuple(sorted(config.items())))
            kernel = _compile_kernel(key, make_gemm_kernel, shape_args, config)
            gw = kernel(go_fp16, in_fp16)
            grad_weight = gw.float()

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, grad_weight, grad_bias


class TileLangLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return TileLangLinearFunction.apply(x, self.weight, self.bias)


# ============================================================
# CNN Model
# ============================================================

class TileLangCNN(nn.Module):
    """
    CNN using TileLang autotuned kernels for conv2d and linear.
    BatchNorm/ReLU/MaxPool/Dropout use PyTorch (memory-bound, no benefit from custom kernels).
    """

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
