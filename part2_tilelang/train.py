"""
Training script for TileLang CNN on handwritten Chinese characters.
Replicates Part 2 results using custom TileLang kernels for conv2d and linear.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from model import TileLangCNN


# ============================================================
# Dataset (same as Part 2)
# ============================================================

class ChineseCharDataset(Dataset):
    def __init__(self, data_dir, img_size=28):
        self.images = []
        self.labels = []
        self.img_size = img_size
        self.num_classes = 12

        for class_id in range(1, self.num_classes + 1):
            class_dir = os.path.join(data_dir, 'train', str(class_id))
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith('.bmp'):
                    continue
                fpath = os.path.join(class_dir, fname)
                img = Image.open(fpath).convert('L')
                self.images.append(img)
                self.labels.append(class_id - 1)

        self.labels = np.array(self.labels, dtype=np.int64)
        print(f"Loaded {len(self.images)} images, {self.num_classes} classes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].copy(), self.labels[idx]


# ============================================================
# Transforms
# ============================================================

class TrainTransform:
    def __init__(self, img_size=28):
        self.img_size = img_size

    def __call__(self, img):
        img = img.resize((self.img_size, self.img_size))
        # Random shift
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
        # Random rotation
        angle = np.random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=0)
        # To tensor
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0)
        # Random noise
        if np.random.rand() < 0.5:
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.03, 0, 1)
        return tensor


class TestTransform:
    def __init__(self, img_size=28):
        self.img_size = img_size

    def __call__(self, img):
        img = img.resize((self.img_size, self.img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0)


class TransformSubset(Dataset):
    def __init__(self, parent, indices, transform):
        self.parent = parent
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.parent[real_idx]
        img = self.transform(img)
        return img, label


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'train_data')

    # Dataset
    img_size = 28
    full_dataset = ChineseCharDataset(data_dir, img_size=img_size)

    # Stratified split
    num_classes = 12
    rng = np.random.RandomState(SEED)
    train_indices = []
    test_indices = []
    for c in range(num_classes):
        idx = np.where(full_dataset.labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * 0.2))
        test_indices.extend(idx[:n_test].tolist())
        train_indices.extend(idx[n_test:].tolist())

    train_dataset = TransformSubset(full_dataset, train_indices, TrainTransform(img_size))
    test_dataset = TransformSubset(full_dataset, test_indices, TestTransform(img_size))
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Model
    model = TileLangCNN(num_classes=num_classes, dropout_rate=0.3).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nTileLang CNN parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # Training
    epochs = 100
    best_test_acc = 0.0
    best_model_path = os.path.join(script_dir, 'best_tilelang_cnn.pth')

    print("\nTraining with TileLang kernels...")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)

        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}  "
                  f"lr={lr:.6f}  best={best_test_acc:.4f}  "
                  f"time={elapsed:.1f}s")

    # Final evaluation with best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n{'=' * 80}")
    print(f"=== Final Results (TileLang CNN, Best Model) ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")

    print("\nPer-class accuracy:")
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = np.mean(all_preds[mask] == c)
            print(f"  Class {c+1}: {acc:.4f} ({mask.sum()} samples)")

    # Confusion matrix plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(all_labels, all_preds):
            cm[true][pred] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'TileLang CNN Confusion Matrix (Acc={test_acc:.4f})')
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
        save_path = os.path.join(script_dir, 'tilelang_cnn_result.png')
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to {save_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
