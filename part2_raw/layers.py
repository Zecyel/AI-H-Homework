"""
Manual CNN layers with forward/backward pass.
All computation uses CuPy. Conv2d and Linear use TileLang GEMM via bridge.
No PyTorch imports.
"""

import cupy as cp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from bridge import tilelang_conv_fwd, tilelang_conv_bwd_data, tilelang_conv_bwd_weight, tilelang_gemm


# ============================================================
# Conv2d (TileLang implicit im2col GEMM)
# ============================================================

class Conv2d:
    """Conv2d using TileLang kernels. NCHW layout externally, NHWC for TileLang."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.C_in = in_channels
        self.C_out = out_channels
        self.KH = self.KW = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding

        # Kaiming init
        fan_out = out_channels * self.KH * self.KW
        std = np.sqrt(2.0 / fan_out)
        self.weight = cp.array(np.random.randn(out_channels, in_channels, self.KH, self.KW).astype(np.float32) * std)
        self.bias = cp.zeros(out_channels, dtype=cp.float32)
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x):
        """x: (N, C_in, H, W) float32 -> (N, C_out, OH, OW) float32"""
        self.input = x
        N, C_in, H, W = x.shape
        OH = (H + 2 * self.padding - self.KH) // self.stride + 1
        OW = (W + 2 * self.padding - self.KW) // self.stride + 1

        # NCHW -> NHWC for TileLang
        x_nhwc = cp.ascontiguousarray(x.transpose(0, 2, 3, 1)).astype(cp.float16)
        # weight OIHW -> HWIO
        w_hwio = cp.ascontiguousarray(self.weight.transpose(2, 3, 1, 0)).astype(cp.float16)

        out_nhwc = tilelang_conv_fwd(x_nhwc, w_hwio, N, C_in, H, W,
                                      self.C_out, self.KH, self.KW,
                                      self.stride, self.padding)
        # NHWC -> NCHW
        output = cp.ascontiguousarray(out_nhwc.transpose(0, 3, 1, 2)).astype(cp.float32)
        output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def backward(self, grad_output):
        """grad_output: (N, C_out, OH, OW) -> returns (N, C_in, H, W)"""
        N, C_in, H, W = self.input.shape

        # grad_bias
        self.grad_bias = grad_output.sum(axis=(0, 2, 3))

        go_nhwc = cp.ascontiguousarray(grad_output.transpose(0, 2, 3, 1)).astype(cp.float16)
        w_hwio = cp.ascontiguousarray(self.weight.transpose(2, 3, 1, 0)).astype(cp.float16)

        # grad_input
        gi_nhwc = tilelang_conv_bwd_data(go_nhwc, w_hwio, N, C_in, H, W,
                                          self.C_out, self.KH, self.KW,
                                          self.stride, self.padding)
        grad_input = cp.ascontiguousarray(gi_nhwc.transpose(0, 3, 1, 2)).astype(cp.float32)

        # grad_weight
        x_nhwc = cp.ascontiguousarray(self.input.transpose(0, 2, 3, 1)).astype(cp.float16)
        gw_hwio = tilelang_conv_bwd_weight(x_nhwc, go_nhwc, N, C_in, H, W,
                                            self.C_out, self.KH, self.KW,
                                            self.stride, self.padding)
        # HWIO -> OIHW
        self.grad_weight = cp.ascontiguousarray(gw_hwio.transpose(3, 2, 0, 1)).astype(cp.float32)

        return grad_input


# ============================================================
# BatchNorm2d
# ============================================================

class BatchNorm2d:
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = cp.ones((1, num_features, 1, 1), dtype=cp.float32)
        self.beta = cp.zeros((1, num_features, 1, 1), dtype=cp.float32)
        self.running_mean = cp.zeros((1, num_features, 1, 1), dtype=cp.float32)
        self.running_var = cp.ones((1, num_features, 1, 1), dtype=cp.float32)
        self.training = True
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        if self.training:
            self.mean = x.mean(axis=(0, 2, 3), keepdims=True)
            self.var = x.var(axis=(0, 2, 3), keepdims=True)
            self.x_norm = (x - self.mean) / cp.sqrt(self.var + self.eps)
            self.x = x
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var
        else:
            self.x_norm = (x - self.running_mean) / cp.sqrt(self.running_var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        N, C, H, W = grad_output.shape
        m = N * H * W  # number of elements per channel

        self.grad_gamma = (grad_output * self.x_norm).sum(axis=(0, 2, 3), keepdims=True)
        self.grad_beta = grad_output.sum(axis=(0, 2, 3), keepdims=True)

        dx_norm = grad_output * self.gamma
        std_inv = 1.0 / cp.sqrt(self.var + self.eps)
        dvar = (dx_norm * (self.x - self.mean) * (-0.5) * std_inv ** 3).sum(axis=(0, 2, 3), keepdims=True)
        dmean = (-dx_norm * std_inv).sum(axis=(0, 2, 3), keepdims=True) + \
                dvar * (-2.0 * (self.x - self.mean)).mean(axis=(0, 2, 3), keepdims=True)

        dx = dx_norm * std_inv + dvar * 2.0 * (self.x - self.mean) / m + dmean / m
        return dx


# ============================================================
# ReLU
# ============================================================

class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


# ============================================================
# MaxPool2d
# ============================================================

class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.ks = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        OH = H // self.ks
        OW = W // self.ks

        # Reshape to (N, C, OH, ks, OW, ks) and take max
        x_reshaped = x.reshape(N, C, OH, self.ks, OW, self.ks)
        out = x_reshaped.max(axis=(3, 5))

        # Save argmax for backward
        self.input_shape = x.shape
        self.out = out
        self.x = x
        return out

    def backward(self, grad_output):
        N, C, H, W = self.input_shape
        OH, OW = H // self.ks, W // self.ks

        # Upsample grad to input size
        grad_input = cp.zeros(self.input_shape, dtype=grad_output.dtype)

        # Expand output back to input resolution for comparison
        out_expanded = self.out[:, :, :, cp.newaxis, :, cp.newaxis]
        out_expanded = cp.broadcast_to(out_expanded, (N, C, OH, self.ks, OW, self.ks))
        x_reshaped = self.x.reshape(N, C, OH, self.ks, OW, self.ks)

        # Mask where input equals max (only first match matters for gradient)
        mask = (x_reshaped == out_expanded).astype(grad_output.dtype)
        # Normalize mask so gradient is split if there are ties
        mask_sum = mask.sum(axis=(3, 5), keepdims=True)
        mask_sum = cp.maximum(mask_sum, 1.0)
        mask = mask / mask_sum

        grad_expanded = grad_output[:, :, :, cp.newaxis, :, cp.newaxis]
        grad_expanded = cp.broadcast_to(grad_expanded, (N, C, OH, self.ks, OW, self.ks))
        grad_input = (mask * grad_expanded).reshape(N, C, H, W)
        return grad_input


# ============================================================
# Dropout2d (channel-wise)
# ============================================================

class Dropout2d:
    def __init__(self, p=0.3):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # Channel-wise mask: (N, C, 1, 1)
        N, C = x.shape[0], x.shape[1]
        self.mask = (cp.random.rand(N, C, 1, 1) > self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_output):
        if not self.training or self.p == 0:
            return grad_output
        return grad_output * self.mask


# ============================================================
# Dropout (element-wise)
# ============================================================

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        self.mask = (cp.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_output):
        if not self.training or self.p == 0:
            return grad_output
        return grad_output * self.mask


# ============================================================
# Linear (TileLang GEMM)
# ============================================================

class Linear:
    def __init__(self, in_features, out_features):
        std = np.sqrt(2.0 / in_features)
        self.weight = cp.array(np.random.randn(out_features, in_features).astype(np.float32) * std)
        self.bias = cp.zeros(out_features, dtype=cp.float32)
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x):
        """x: (N, in_features) -> (N, out_features)"""
        self.input = x
        # y = x @ W^T + b  =>  y = x @ W^T  where W is (out, in)
        # GEMM: A(N, in) @ B(in, out) = C(N, out)
        A = x.astype(cp.float16)
        B = cp.ascontiguousarray(self.weight.T).astype(cp.float16)  # (in, out)
        output = tilelang_gemm(A, B).astype(cp.float32)
        output = output + self.bias
        return output

    def backward(self, grad_output):
        """grad_output: (N, out_features) -> returns (N, in_features)"""
        # grad_bias
        self.grad_bias = grad_output.sum(axis=0)

        # grad_input = grad_output @ weight  =>  (N, out) @ (out, in) = (N, in)
        A = grad_output.astype(cp.float16)
        B = self.weight.astype(cp.float16)  # (out, in)
        grad_input = tilelang_gemm(A, B).astype(cp.float32)

        # grad_weight = grad_output^T @ input  =>  (out, N) @ (N, in) = (out, in)
        A = cp.ascontiguousarray(grad_output.T).astype(cp.float16)
        B = self.input.astype(cp.float16)
        self.grad_weight = tilelang_gemm(A, B).astype(cp.float32)

        return grad_input


# ============================================================
# Global Average Pooling
# ============================================================

class GlobalAvgPool:
    def forward(self, x):
        """x: (N, C, H, W) -> (N, C)"""
        self.input_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad_output):
        """grad_output: (N, C) -> (N, C, H, W)"""
        N, C, H, W = self.input_shape
        return (grad_output.reshape(N, C, 1, 1) / (H * W)) * cp.ones((N, C, H, W), dtype=grad_output.dtype)


# ============================================================
# CrossEntropyLoss with label smoothing
# ============================================================

class CrossEntropyLoss:
    def __init__(self, label_smoothing=0.0):
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        """
        logits: (N, C) float32
        labels: (N,) int64
        Returns: scalar loss
        """
        self.N, self.C = logits.shape

        # Numerical stability: subtract max
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = cp.log(cp.exp(logits_shifted).sum(axis=1, keepdims=True))
        self.log_probs = logits_shifted - log_sum_exp  # log-softmax

        # One-hot with label smoothing
        one_hot = cp.zeros_like(logits)
        one_hot[cp.arange(self.N), labels] = 1.0
        if self.label_smoothing > 0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + self.label_smoothing / self.C
        self.target = one_hot

        loss = -(one_hot * self.log_probs).sum() / self.N
        self.probs = cp.exp(self.log_probs)  # softmax output for backward
        return float(loss)

    def backward(self):
        """Returns grad w.r.t. logits: (N, C)"""
        return (self.probs - self.target) / self.N
