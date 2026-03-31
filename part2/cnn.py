"""
Part 2: CNN for 12-class handwritten Chinese character classification.
Uses PyTorch with GPU support. No pre-built models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from PIL import Image


# ============================================================
# Dataset
# ============================================================

class ChineseCharDataset(Dataset):
    """Dataset for handwritten Chinese character images."""

    def __init__(self, data_dir, img_size=28, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.img_size = img_size
        self.num_classes = 12

        for class_id in range(1, self.num_classes + 1):
            class_dir = os.path.join(data_dir, 'train', str(class_id))
            if not os.path.isdir(class_dir):
                print(f"Warning: {class_dir} not found, skipping.")
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
        img = self.images[idx].copy()

        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img, dtype=np.float32) / 255.0
            img = torch.tensor(img).unsqueeze(0)  # (1, H, W)

        label = self.labels[idx]
        return img, label


# ============================================================
# Transforms (manual, no torchvision dependency)
# ============================================================

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img.resize((self.size, self.size))


class RandomShift:
    """Random translation by up to max_shift pixels."""
    def __init__(self, max_shift=3):
        self.max_shift = max_shift

    def __call__(self, img):
        dx = np.random.randint(-self.max_shift, self.max_shift + 1)
        dy = np.random.randint(-self.max_shift, self.max_shift + 1)
        return img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)


class RandomRotation:
    """Random rotation by up to max_angle degrees."""
    def __init__(self, max_angle=15):
        self.max_angle = max_angle

    def __call__(self, img):
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        return img.rotate(angle, fillcolor=0)


class RandomScale:
    """Random scaling."""
    def __init__(self, scale_range=(0.85, 1.15)):
        self.scale_range = scale_range

    def __call__(self, img):
        scale = np.random.uniform(*self.scale_range)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))
        # Center crop or pad back to original size
        result = Image.new('L', (w, h), 0)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        if scale >= 1.0:
            # Crop center
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            result = img.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        else:
            result.paste(img, (paste_x, paste_y))
        return result


class GaussianNoise:
    """Add Gaussian noise."""
    def __init__(self, std=0.03):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)


class ToTensor:
    """Convert PIL image to tensor and normalize to [0, 1]."""
    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0)  # (1, H, W)


def get_train_transform(img_size=28):
    return Compose([
        Resize(img_size),
        RandomShift(max_shift=3),
        RandomRotation(max_angle=15),
        RandomScale(scale_range=(0.85, 1.15)),
        ToTensor(),
        GaussianNoise(std=0.03),
    ])


def get_test_transform(img_size=28):
    return Compose([
        Resize(img_size),
        ToTensor(),
    ])


# ============================================================
# CNN Model
# ============================================================

class CNN(nn.Module):
    """
    Custom CNN for 28x28 grayscale image classification.

    Architecture:
        Conv2d(1, 32, 3) -> BN -> ReLU -> Conv2d(32, 32, 3) -> BN -> ReLU -> MaxPool(2) -> Dropout
        Conv2d(32, 64, 3) -> BN -> ReLU -> Conv2d(64, 64, 3) -> BN -> ReLU -> MaxPool(2) -> Dropout
        Conv2d(64, 128, 3) -> BN -> ReLU -> GAP
        FC(128, 64) -> ReLU -> Dropout -> FC(64, num_classes)
    """

    def __init__(self, num_classes=12, dropout_rate=0.3):
        super().__init__()

        # Block 1: 28x28 -> 12x12
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout_rate)

        # Block 2: 12x12 -> 5x5 (actually 14->12 after conv without padding, then pool)
        # With padding=1: 12x12 -> 12x12 -> pool -> 6x6
        # Wait let me recalculate: 28 -> conv(pad=1) -> 28 -> conv(pad=1) -> 28 -> pool(2) -> 14
        # 14 -> conv(pad=1) -> 14 -> conv(pad=1) -> 14 -> pool(2) -> 7

        # Block 2: 14x14 -> 7x7
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout_rate)

        # Block 3: 7x7 -> GAP -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Average Pooling -> FC
        self.fc1 = nn.Linear(128, 64)
        self.drop3 = nn.Dropout(dropout_rate * 1.5)
        self.fc2 = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

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

        # Global Average Pooling
        x = x.mean(dim=[2, 3])  # (B, 128)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'train_data')

    # Create datasets
    img_size = 28
    full_dataset = ChineseCharDataset(data_dir, img_size=img_size)

    # Stratified train/test split
    num_classes = 12
    test_ratio = 0.2
    rng = np.random.RandomState(SEED)
    train_indices = []
    test_indices = []

    for c in range(num_classes):
        idx = np.where(full_dataset.labels == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * test_ratio))
        test_indices.extend(idx[:n_test].tolist())
        train_indices.extend(idx[n_test:].tolist())

    # Create split datasets with appropriate transforms
    train_transform = get_train_transform(img_size)
    test_transform = get_test_transform(img_size)

    class SubsetDataset(Dataset):
        def __init__(self, parent, indices, transform):
            self.parent = parent
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            real_idx = self.indices[idx]
            img = self.parent.images[real_idx].copy()
            label = self.parent.labels[real_idx]
            if self.transform:
                img = self.transform(img)
            else:
                img = np.array(img.resize((self.parent.img_size, self.parent.img_size)),
                               dtype=np.float32) / 255.0
                img = torch.tensor(img).unsqueeze(0)
            return img, label

    train_dataset = SubsetDataset(full_dataset, train_indices, train_transform)
    test_dataset = SubsetDataset(full_dataset, test_indices, test_transform)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Model
    model = CNN(num_classes=num_classes, dropout_rate=0.3).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    # Training loop
    epochs = 100
    best_test_acc = 0.0
    best_model_path = os.path.join(script_dir, 'best_cnn_model.pth')

    print("\nTraining...")
    total_train_time = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        total_train_time += elapsed

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

    print(f"\nTotal training time: {total_train_time:.1f}s ({total_train_time/epochs:.2f}s/epoch)")

    # Load best model and final evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader,
                                                          criterion, device)

    print(f"\n=== Final Results (Best Model) ===")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Per-class accuracy
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
        ax.set_title(f'CNN Confusion Matrix (Test Acc={test_acc:.4f})')
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
        save_path = os.path.join(script_dir, 'cnn_confusion_matrix.png')
        plt.savefig(save_path, dpi=150)
        print(f"\nConfusion matrix saved to {save_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")


if __name__ == '__main__':
    main()
