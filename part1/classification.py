"""
Classification task: 12-class handwritten Chinese character recognition.
Uses the neural network framework from nn.py.
"""

import numpy as np
import os
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from nn import Network, StepLRScheduler


def load_data(data_dir, img_size=28):
    """
    Load BMP images from data_dir/train/{1..12}/.
    Returns X (N, img_size*img_size), y_onehot (N, 12), labels (N,).
    """
    images = []
    labels = []
    num_classes = 12

    for class_id in range(1, num_classes + 1):
        class_dir = os.path.join(data_dir, 'train', str(class_id))
        if not os.path.isdir(class_dir):
            print(f"Warning: {class_dir} not found, skipping.")
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith('.bmp'):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert('L')  # grayscale
            img = img.resize((img_size, img_size))
            arr = np.array(img, dtype=np.float64).flatten()
            images.append(arr)
            labels.append(class_id - 1)  # 0-indexed

    X = np.array(images, dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)

    # Normalize to [0, 1]
    X = X / 255.0

    # One-hot encode
    y_onehot = np.zeros((len(labels), num_classes), dtype=np.float64)
    y_onehot[np.arange(len(labels)), labels] = 1.0

    return X, y_onehot, labels


def augment_batch(X_batch, img_size=28):
    """
    On-the-fly data augmentation for a batch of flattened images.
    Applies random shifts, small rotations, and noise.
    """
    batch_size = X_batch.shape[0]
    augmented = np.empty_like(X_batch)

    for i in range(batch_size):
        img = X_batch[i].reshape(img_size, img_size)

        # Random shift (-2 to +2 pixels)
        dx, dy = np.random.randint(-2, 3), np.random.randint(-2, 3)
        shifted = np.zeros_like(img)
        src_x = slice(max(0, dx), min(img_size, img_size + dx))
        src_y = slice(max(0, dy), min(img_size, img_size + dy))
        dst_x = slice(max(0, -dx), min(img_size, img_size - dx))
        dst_y = slice(max(0, -dy), min(img_size, img_size - dy))
        shifted[dst_y, dst_x] = img[src_y, src_x]

        # Random noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(img_size, img_size) * 0.02
            shifted = np.clip(shifted + noise, 0, 1)

        augmented[i] = shifted.flatten()

    return augmented


def train_test_split(X, y_onehot, labels, test_ratio=0.2, seed=42):
    """Split data into train and test sets, stratified by class."""
    rng = np.random.RandomState(seed)
    num_classes = y_onehot.shape[1]
    train_idx = []
    test_idx = []

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], y_onehot[train_idx], X[test_idx], y_onehot[test_idx], labels[test_idx]


def evaluate(net, X, y_onehot):
    """Return accuracy and loss."""
    pred = net.predict(X)
    pred_labels = np.argmax(pred, axis=1)
    true_labels = np.argmax(y_onehot, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    loss = net.loss_fn.forward(pred, y_onehot)
    return accuracy, loss


def main():
    np.random.seed(42)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'train_data')

    print("Loading data...")
    X, y_onehot, labels = load_data(data_dir)
    print(f"Total samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {y_onehot.shape[1]}")

    X_train, y_train, X_test, y_test, test_labels = train_test_split(
        X, y_onehot, labels, test_ratio=0.2)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    input_dim = X.shape[1]  # 784
    num_classes = 12

    net = Network([
        ('linear', input_dim, 512),
        ('batchnorm', 512),
        ('relu',),
        ('dropout', 0.4),
        ('linear', 512, 256),
        ('batchnorm', 256),
        ('relu',),
        ('dropout', 0.4),
        ('linear', 256, 128),
        ('batchnorm', 128),
        ('relu',),
        ('dropout', 0.3),
        ('linear', 128, num_classes),
        ('softmax',),
    ], loss='cross_entropy', optimizer='adam', lr=0.001,
       weight_init='he', weight_decay=1e-4)

    scheduler = StepLRScheduler(net.optimizer, step_size=40, gamma=0.5)

    print("\nTraining...")
    epochs = 200
    batch_size = 64
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        net.train_mode()
        n = X_train.shape[0]
        indices = np.random.permutation(n)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]

            # Apply data augmentation in [0,1] pixel space
            xb_aug = augment_batch(xb)

            loss = net.train_step(xb_aug, yb)
            epoch_loss += loss
            n_batches += 1

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            train_acc, train_loss = evaluate(net, X_train, y_train)
            test_acc, test_loss = evaluate(net, X_test, y_test)
            best_test_acc = max(best_test_acc, test_acc)
            print(f"Epoch {epoch}/{epochs}  "
                  f"train_loss={epoch_loss/n_batches:.4f} train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
                  f"lr={net.optimizer.lr:.6f}")

    # Final evaluation
    train_acc, _ = evaluate(net, X_train, y_train)
    test_acc, _ = evaluate(net, X_test, y_test)
    best_test_acc = max(best_test_acc, test_acc)
    print(f"\n=== Final Results ===")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Best Test Acc:  {best_test_acc:.4f}")

    # Per-class accuracy
    pred = net.predict(X_test)
    pred_labels = np.argmax(pred, axis=1)
    print("\nPer-class accuracy:")
    for c in range(num_classes):
        mask = test_labels == c
        if mask.sum() > 0:
            acc = np.mean(pred_labels[mask] == c)
            print(f"  Class {c+1}: {acc:.4f} ({mask.sum()} samples)")

    # Optional: save plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, predicted in zip(test_labels, pred_labels):
            cm[true][predicted] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Confusion Matrix (Test Acc={test_acc:.4f})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([str(i+1) for i in range(num_classes)])
        ax.set_yticklabels([str(i+1) for i in range(num_classes)])

        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.colorbar(im)
        plt.tight_layout()
        save_path = os.path.join(script_dir, 'classification_result.png')
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to {save_path}")
    except ImportError:
        print("\nmatplotlib/sklearn not available, skipping plot.")


if __name__ == '__main__':
    main()
