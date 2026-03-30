"""
Neural Network framework implemented from scratch using numpy.
Supports flexible layer configuration, multiple activation functions,
and different loss functions for regression and classification tasks.
"""

import numpy as np


# ============================================================
# Activation functions
# ============================================================

class Activation:
    """Base class for activation functions."""
    def forward(self, z):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, z):
        self.mask = (z > 0).astype(z.dtype)
        return z * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid(Activation):
    def forward(self, z):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


class Tanh(Activation):
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)


class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, z):
        self.mask = (z > 0).astype(z.dtype)
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, grad_output):
        return grad_output * np.where(self.mask, 1.0, self.alpha)


class Softmax(Activation):
    """Softmax activation — typically used together with CrossEntropyLoss."""
    def forward(self, z):
        shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        self.out = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.out

    def backward(self, grad_output):
        # When paired with CrossEntropyLoss, the combined gradient
        # is simply (softmax_output - target), which is passed directly.
        return grad_output


class Identity(Activation):
    def forward(self, z):
        return z

    def backward(self, grad_output):
        return grad_output


# ============================================================
# Loss functions
# ============================================================

class Loss:
    def forward(self, predicted, target):
        raise NotImplementedError

    def backward(self, predicted, target):
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error for regression."""
    def forward(self, predicted, target):
        self.predicted = predicted
        self.target = target
        return np.mean((predicted - target) ** 2)

    def backward(self, predicted, target):
        n = predicted.shape[0]
        return 2.0 * (predicted - target) / n


class CrossEntropyLoss(Loss):
    """Cross-entropy loss for classification (expects softmax output)."""
    def forward(self, predicted, target):
        # target: one-hot (N, C)  predicted: softmax probabilities (N, C)
        n = predicted.shape[0]
        clipped = np.clip(predicted, 1e-12, 1.0 - 1e-12)
        loss = -np.sum(target * np.log(clipped)) / n
        return loss

    def backward(self, predicted, target):
        # Combined gradient for softmax + cross-entropy
        n = predicted.shape[0]
        return (predicted - target) / n


# ============================================================
# Layers
# ============================================================

class Linear:
    """Fully-connected layer."""
    def __init__(self, in_features, out_features, weight_init='he'):
        if weight_init == 'he':
            self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        elif weight_init == 'xavier':
            limit = np.sqrt(6.0 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        else:
            self.W = np.random.randn(in_features, out_features) * weight_init

        self.b = np.zeros((1, out_features)) - 0.01  # small negative bias per hint

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T


class Dropout:
    """Dropout regularization layer."""
    def __init__(self, p=0.5):
        self.p = p  # drop probability
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask
        return grad_output


class BatchNorm1d:
    """Batch normalization for 1D inputs."""
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        self.training = True

        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x):
        if self.training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.x = x
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        n = grad_output.shape[0]
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (self.x - self.mean) * (-0.5) * (self.var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * (-1.0 / np.sqrt(self.var + self.eps)), axis=0, keepdims=True) + dvar * np.mean(-2.0 * (self.x - self.mean), axis=0, keepdims=True)

        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2.0 * (self.x - self.mean) / n + dmean / n
        return dx


# ============================================================
# Optimizers
# ============================================================

class SGD:
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                for param_name in ('W', 'b'):
                    param = getattr(layer, param_name)
                    grad = getattr(layer, f'grad_{param_name}')
                    if grad is None:
                        continue

                    if self.weight_decay > 0:
                        grad = grad + self.weight_decay * param

                    key = (i, param_name)
                    if self.momentum > 0:
                        if key not in self.velocities:
                            self.velocities[key] = np.zeros_like(param)
                        self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad
                        param += self.velocities[key]
                    else:
                        param -= self.lr * grad
                    setattr(layer, param_name, param)

            elif isinstance(layer, BatchNorm1d):
                for param_name in ('gamma', 'beta'):
                    param = getattr(layer, param_name)
                    grad = getattr(layer, f'grad_{param_name}')
                    if grad is None:
                        continue
                    key = (i, param_name)
                    if self.momentum > 0:
                        if key not in self.velocities:
                            self.velocities[key] = np.zeros_like(param)
                        self.velocities[key] = self.momentum * self.velocities[key] - self.lr * grad
                        param += self.velocities[key]
                    else:
                        param -= self.lr * grad
                    setattr(layer, param_name, param)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                param_names = ('W', 'b')
            elif isinstance(layer, BatchNorm1d):
                param_names = ('gamma', 'beta')
            else:
                continue

            for param_name in param_names:
                param = getattr(layer, param_name)
                grad = getattr(layer, f'grad_{param_name}')
                if grad is None:
                    continue

                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param

                key = (i, param_name)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                setattr(layer, param_name, param)


# ============================================================
# Learning Rate Scheduler
# ============================================================

class StepLRScheduler:
    def __init__(self, optimizer, step_size=50, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


# ============================================================
# Network
# ============================================================

ACTIVATION_MAP = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'leaky_relu': LeakyReLU,
    'softmax': Softmax,
    'identity': Identity,
}


class Network:
    """
    Flexible neural network with configurable layers.

    Example usage:
        net = Network([
            ('linear', 1, 64),
            ('relu',),
            ('linear', 64, 64),
            ('relu',),
            ('linear', 64, 1),
        ], loss='mse', optimizer='adam', lr=0.001)

        for epoch in range(epochs):
            loss = net.train_step(X, y)
    """

    def __init__(self, layer_specs, loss='mse', optimizer='adam', lr=0.001,
                 momentum=0.9, weight_decay=0.0, weight_init='he'):
        self.layers = []
        for spec in layer_specs:
            name = spec[0].lower()
            if name == 'linear':
                self.layers.append(Linear(spec[1], spec[2], weight_init=weight_init))
            elif name == 'dropout':
                self.layers.append(Dropout(p=spec[1] if len(spec) > 1 else 0.5))
            elif name == 'batchnorm':
                self.layers.append(BatchNorm1d(spec[1]))
            elif name in ACTIVATION_MAP:
                self.layers.append(ACTIVATION_MAP[name]())
            else:
                raise ValueError(f"Unknown layer type: {name}")

        if loss == 'mse':
            self.loss_fn = MSELoss()
        elif loss == 'cross_entropy':
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss: {loss}")

        if optimizer == 'sgd':
            self.optimizer = SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == 'adam':
            self.optimizer = Adam(lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_mode(self):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm1d)):
                layer.training = True

    def eval_mode(self):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm1d)):
                layer.training = False

    def train_step(self, x, y):
        self.train_mode()
        output = self.forward(x)
        loss = self.loss_fn.forward(output, y)
        grad = self.loss_fn.backward(output, y)
        self.backward(grad)
        self.optimizer.step(self.layers)
        return loss

    def predict(self, x):
        self.eval_mode()
        return self.forward(x)

    def fit(self, X, y, epochs=100, batch_size=32, verbose=True, val_data=None,
            scheduler=None):
        n = X.shape[0]
        history = {'train_loss': []}
        if val_data is not None:
            history['val_loss'] = []

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]
                loss = self.train_step(xb, yb)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)

            if scheduler is not None:
                scheduler.step()

            if val_data is not None:
                val_pred = self.predict(val_data[0])
                val_loss = self.loss_fn.forward(val_pred, val_data[1])
                history['val_loss'].append(val_loss)

            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
                msg = f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.6f}"
                if val_data is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.6f}"
                if scheduler is not None:
                    msg += f"  lr={self.optimizer.lr:.6f}"
                print(msg)

        return history
