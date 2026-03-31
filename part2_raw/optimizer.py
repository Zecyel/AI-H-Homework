"""
Adam optimizer for CuPy arrays. No PyTorch.
"""

import cupy as cp


class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-3):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}  # first moment
        self.v = {}  # second moment

    def step(self):
        self.t += 1
        for param, grad_fn, name in self.model.parameters():
            grad = grad_fn()
            if grad is None:
                continue

            # AdamW: decoupled weight decay
            if self.weight_decay > 0 and 'bias' not in name and 'beta' not in name:
                param -= self.lr * self.weight_decay * param

            if name not in self.m:
                self.m[name] = cp.zeros_like(param)
                self.v[name] = cp.zeros_like(param)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param, grad_fn, name in self.model.parameters():
            # Gradients are overwritten each backward pass, no need to zero
            pass


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=1e-5):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        import math
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self.epoch / self.T_max)) / 2
