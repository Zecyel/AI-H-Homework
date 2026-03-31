"""
CNN model assembled from manual layers. No PyTorch.
Same architecture as part2/cnn.py but fully hand-written.
"""

import cupy as cp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from layers import (
    Conv2d, BatchNorm2d, ReLU, MaxPool2d,
    Dropout2d, Dropout, Linear, GlobalAvgPool,
)


class TileLangCNN:
    """
    Hand-written CNN using CuPy + TileLang. No PyTorch.

    Architecture:
        Conv2d(16, 32, 3, pad=1) -> BN -> ReLU -> Conv2d(32, 32, 3, pad=1) -> BN -> ReLU -> MaxPool(2) -> Dropout
        Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> Conv2d(64, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2) -> Dropout
        Conv2d(64, 128, 3, pad=1) -> BN -> ReLU -> GAP
        Linear(128, 64) -> ReLU -> Dropout -> Linear(64, 16) [padded from 12]
    """

    def __init__(self, num_classes=12, dropout_rate=0.3):
        self.num_classes = num_classes
        # Pad input from 1 -> 16 channels for TileLang tensor core compatibility
        self.input_pad_channels = 16
        # Pad num_classes to 16 for GEMM compatibility
        self.output_pad = (16 - num_classes % 16) % 16 if num_classes < 16 else 0
        self.num_classes_padded = num_classes + self.output_pad

        # Build layers in order
        self.layers = []

        # Block 1
        self.conv1a = Conv2d(self.input_pad_channels, 32, 3, padding=1)
        self.bn1a = BatchNorm2d(32)
        self.relu1a = ReLU()
        self.conv1b = Conv2d(32, 32, 3, padding=1)
        self.bn1b = BatchNorm2d(32)
        self.relu1b = ReLU()
        self.pool1 = MaxPool2d(2)
        self.drop1 = Dropout2d(dropout_rate)

        # Block 2
        self.conv2a = Conv2d(32, 64, 3, padding=1)
        self.bn2a = BatchNorm2d(64)
        self.relu2a = ReLU()
        self.conv2b = Conv2d(64, 64, 3, padding=1)
        self.bn2b = BatchNorm2d(64)
        self.relu2b = ReLU()
        self.pool2 = MaxPool2d(2)
        self.drop2 = Dropout2d(dropout_rate)

        # Block 3
        self.conv3 = Conv2d(64, 128, 3, padding=1)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = ReLU()
        self.gap = GlobalAvgPool()

        # Classifier
        self.fc1 = Linear(128, 64)
        self.relu_fc = ReLU()
        self.drop3 = Dropout(dropout_rate * 1.5)
        self.fc2 = Linear(64, self.num_classes_padded)

        # Ordered list for forward/backward traversal
        self.layer_list = [
            self.conv1a, self.bn1a, self.relu1a,
            self.conv1b, self.bn1b, self.relu1b,
            self.pool1, self.drop1,
            self.conv2a, self.bn2a, self.relu2a,
            self.conv2b, self.bn2b, self.relu2b,
            self.pool2, self.drop2,
            self.conv3, self.bn3, self.relu3,
            self.gap,
            self.fc1, self.relu_fc, self.drop3,
            self.fc2,
        ]

    def forward(self, x):
        """x: (N, 1, 28, 28) float32 CuPy array -> (N, num_classes) logits"""
        # Pad channels: 1 -> 16
        if x.shape[1] < self.input_pad_channels:
            pad_width = ((0, 0), (0, self.input_pad_channels - x.shape[1]), (0, 0), (0, 0))
            x = cp.pad(x, pad_width, mode='constant')

        for layer in self.layer_list:
            x = layer.forward(x)

        # Slice padded output back to real num_classes
        if self.output_pad > 0:
            x = x[:, :self.num_classes]
        return x

    def backward(self, grad):
        """grad: (N, num_classes) -> propagate backwards through all layers"""
        # Pad grad to match padded output
        if self.output_pad > 0:
            grad = cp.pad(grad, ((0, 0), (0, self.output_pad)), mode='constant')

        for layer in reversed(self.layer_list):
            grad = layer.backward(grad)
        return grad

    def train_mode(self):
        for layer in self.layer_list:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval_mode(self):
        for layer in self.layer_list:
            if hasattr(layer, 'training'):
                layer.training = False

    def parameters(self):
        """Yield (param, grad, name) tuples for optimizer."""
        param_layers = [
            ('conv1a', self.conv1a), ('conv1b', self.conv1b),
            ('conv2a', self.conv2a), ('conv2b', self.conv2b),
            ('conv3', self.conv3),
            ('fc1', self.fc1), ('fc2', self.fc2),
        ]
        for name, layer in param_layers:
            yield layer.weight, lambda l=layer: l.grad_weight, f'{name}.weight'
            yield layer.bias, lambda l=layer: l.grad_bias, f'{name}.bias'

        bn_layers = [
            ('bn1a', self.bn1a), ('bn1b', self.bn1b),
            ('bn2a', self.bn2a), ('bn2b', self.bn2b),
            ('bn3', self.bn3),
        ]
        for name, layer in bn_layers:
            yield layer.gamma, lambda l=layer: l.grad_gamma, f'{name}.gamma'
            yield layer.beta, lambda l=layer: l.grad_beta, f'{name}.beta'

    def param_count(self):
        total = 0
        for param, _, _ in self.parameters():
            total += param.size
        return total
