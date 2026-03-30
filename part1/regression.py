"""
Regression task: fit y = sin(x), x in [-pi, pi].
Uses the neural network framework from nn.py.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from nn import Network, StepLRScheduler


def generate_data(n_train=1000, n_test=500):
    """Generate random samples from [-pi, pi]."""
    X_train = np.random.uniform(-np.pi, np.pi, (n_train, 1))
    y_train = np.sin(X_train)
    X_test = np.random.uniform(-np.pi, np.pi, (n_test, 1))
    y_test = np.sin(X_test)
    return X_train, y_train, X_test, y_test


def main():
    np.random.seed(42)

    X_train, y_train, X_test, y_test = generate_data(n_train=2000, n_test=500)

    # Normalize input to [-1, 1]
    X_train_norm = X_train / np.pi
    X_test_norm = X_test / np.pi

    net = Network([
        ('linear', 1, 64),
        ('relu',),
        ('linear', 64, 64),
        ('relu',),
        ('linear', 64, 32),
        ('relu',),
        ('linear', 32, 1),
    ], loss='mse', optimizer='adam', lr=0.005, weight_init='he')

    scheduler = StepLRScheduler(net.optimizer, step_size=100, gamma=0.5)

    history = net.fit(
        X_train_norm, y_train,
        epochs=300,
        batch_size=64,
        verbose=True,
        val_data=(X_test_norm, y_test),
        scheduler=scheduler,
    )

    # Evaluate
    pred = net.predict(X_test_norm)
    mae = np.mean(np.abs(pred - y_test))
    mse = np.mean((pred - y_test) ** 2)
    print(f"\n=== Results ===")
    print(f"Test MAE: {mae:.6f}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Requirement: avg error < 0.01 => {'PASS' if mae < 0.01 else 'FAIL'}")

    # Optional: save plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        X_plot = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
        X_plot_norm = X_plot / np.pi
        y_plot_pred = net.predict(X_plot_norm)
        y_plot_true = np.sin(X_plot)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(X_plot, y_plot_true, 'b-', label='sin(x)', linewidth=2)
        axes[0].plot(X_plot, y_plot_pred, 'r--', label='Prediction', linewidth=2)
        axes[0].scatter(X_test, y_test, s=5, alpha=0.3, label='Test data')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'Regression: y = sin(x)  (MAE={mae:.6f})')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE Loss')
        axes[1].set_title('Training History')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale('log')

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'regression_result.png')
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
