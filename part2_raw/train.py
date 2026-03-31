"""
Training script for hand-written CNN. No PyTorch.
Uses CuPy + TileLang only.
"""

import cupy as cp
import numpy as np
import os
import time
from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(__file__))

from model import TileLangCNN
from layers import CrossEntropyLoss
from optimizer import Adam, CosineAnnealingLR


# ============================================================
# Data loading (PIL -> numpy -> CuPy)
# ============================================================

def load_dataset(data_dir, img_size=28):
    """Load images from directory structure: data_dir/train/{1..12}/*.bmp"""
    images = []
    labels = []
    num_classes = 12

    for class_id in range(1, num_classes + 1):
        class_dir = os.path.join(data_dir, 'train', str(class_id))
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith('.bmp'):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert('L').resize((img_size, img_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(class_id - 1)

    images = np.stack(images)[:, np.newaxis, :, :]  # (N, 1, H, W)
    labels = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(images)} images, {num_classes} classes")
    return images, labels


def augment_batch(images_np, rng):
    """Simple data augmentation on numpy arrays."""
    N, C, H, W = images_np.shape
    augmented = np.empty_like(images_np)

    for i in range(N):
        img = Image.fromarray((images_np[i, 0] * 255).astype(np.uint8), mode='L')

        # Random shift
        dx = rng.randint(-3, 4)
        dy = rng.randint(-3, 4)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)

        # Random rotation
        angle = rng.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=0)

        arr = np.array(img, dtype=np.float32) / 255.0

        # Random noise
        if rng.rand() < 0.5:
            arr = np.clip(arr + rng.randn(*arr.shape).astype(np.float32) * 0.03, 0, 1)

        augmented[i, 0] = arr

    return augmented


def stratified_split(labels, test_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for c in range(labels.max() + 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(train_idx), np.array(test_idx)


# ============================================================
# Training loop
# ============================================================

def train_one_epoch(model, images_np, labels_np, batch_size, criterion, optimizer, rng):
    model.train_mode()
    N = len(images_np)
    indices = rng.permutation(N)
    total_loss = 0.0
    correct = 0
    total = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = indices[start:end]

        # Augment and move to GPU
        batch_images = augment_batch(images_np[batch_idx], rng)
        x = cp.array(batch_images)
        y = cp.array(labels_np[batch_idx])

        # Forward
        logits = model.forward(x)
        loss = criterion.forward(logits, y)

        # Backward
        grad = criterion.backward()
        model.backward(grad)

        # Update
        optimizer.step()

        # Stats
        total_loss += loss * len(batch_idx)
        preds = logits.argmax(axis=1)
        correct += int((preds == y).sum())
        total += len(batch_idx)

    return total_loss / total, correct / total


def evaluate(model, images_np, labels_np, batch_size, criterion):
    model.eval_mode()
    N = len(images_np)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = cp.array(images_np[start:end])
        y = cp.array(labels_np[start:end])

        logits = model.forward(x)
        loss = criterion.forward(logits, y)

        total_loss += loss * len(x)
        preds = logits.argmax(axis=1)
        correct += int((preds == y).sum())
        total += len(x)
        all_preds.extend(cp.asnumpy(preds).tolist())
        all_labels.extend(cp.asnumpy(y).tolist())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    SEED = 42
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    print("=" * 60)
    print("Hand-written CNN with CuPy + TileLang (No PyTorch)")
    print("=" * 60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'train_data')

    # Load data
    img_size = 28
    images, labels = load_dataset(data_dir, img_size)
    train_idx, test_idx = stratified_split(labels, test_ratio=0.2, seed=SEED)

    train_images, train_labels = images[train_idx], labels[train_idx]
    test_images, test_labels = images[test_idx], labels[test_idx]
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    # Model
    num_classes = 12
    model = TileLangCNN(num_classes=num_classes, dropout_rate=0.3)
    print(f"\nParameters: {model.param_count():,}")

    criterion = CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model, lr=1e-3, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # Training
    epochs = 100
    batch_size = 64
    best_test_acc = 0.0

    print("\nTraining...")
    print("=" * 80)

    total_train_time = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_images, train_labels, batch_size, criterion, optimizer, rng)
        test_loss, test_acc, _, _ = evaluate(
            model, test_images, test_labels, 128, criterion)
        scheduler.step()
        elapsed = time.time() - t0
        total_train_time += elapsed

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
                  f"lr={optimizer.lr:.6f}  best={best_test_acc:.4f}  "
                  f"time={elapsed:.1f}s")

    print(f"\nTotal training time: {total_train_time:.1f}s ({total_train_time/epochs:.2f}s/epoch)")

    # Final evaluation
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_images, test_labels, 128, criterion)

    print(f"\n{'=' * 80}")
    print(f"=== Final Results (Hand-written CuPy+TileLang CNN) ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")

    print("\nPer-class accuracy:")
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = np.mean(all_preds[mask] == c)
            print(f"  Class {c+1}: {acc:.4f} ({mask.sum()} samples)")

    # Confusion matrix
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(all_labels, all_preds):
            cm[true][pred] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Raw TileLang CNN (No PyTorch) - Acc={test_acc:.4f}')
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
        plt.colorbar(ax.images[0])
        plt.tight_layout()
        save_path = os.path.join(script_dir, 'raw_tilelang_result.png')
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to {save_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
