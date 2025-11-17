# -*- coding: utf-8 -*-

#from google.colab import files

#uploaded = files.upload()

import argparse
import random
import pickle
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# ==========================
#  Reproducibility
# ==========================

def set_seed(seed: int = 4368):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================
#  Dataset
# ==========================

class TinyImageNetDataset(Dataset):
    """
    Generic dataset for the Tiny ImageNet subset stored in a pickle file.

    Assumes the .pkl contains either:
      - A dict with 'images' and 'labels' keys, OR
      - A dict with 'X' and 'y' keys, OR
      - A tuple/list like (images, labels), OR
      - A list of (image, label) pairs.
    """

    def __init__(self, pkl_path: str, transform=None):
        super().__init__()
        self.transform = transform

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # ----- Infer / adjust structure -----
        if isinstance(data, dict) and "images" in data and "labels" in data:
            self.images = data["images"]
            self.labels = data["labels"]

        elif isinstance(data, dict) and "X" in data and "y" in data:
            self.images = data["X"]
            self.labels = data["y"]

        elif isinstance(data, (list, tuple)) and len(data) == 2:
            self.images, self.labels = data

        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple)):
            self.images = [x[0] for x in data]
            self.labels = [x[1] for x in data]

        else:
            raise ValueError(
                f"Unknown data format in {pkl_path}. Please inspect TinyImageNetDataset.__init__()."
            )

        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        label = int(self.labels[idx])

        if isinstance(img, np.ndarray):
            # Expecting (H, W, C)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pass  # already PIL
        else:
            raise TypeError(f"Unsupported image type at index {idx}: {type(img)}")

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def get_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and validation dataloaders with appropriate transforms.
    """

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = TinyImageNetDataset(train_path, transform=train_transform)
    val_dataset = TinyImageNetDataset(val_path, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ==========================
#  Model
# ==========================

class TinyImageNetCNN(nn.Module):
    """
    Sequential-style CNN for 64x64 RGB images.
    Uses only Conv -> BatchNorm -> ReLU, Pool, Dropout, Flatten, Linear.
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
            )

        self.features = nn.Sequential(
            conv_block(3, 32),      # 64x64 -> 32x32
            conv_block(32, 64),     # 32x32 -> 16x16
            conv_block(64, 128),    # 16x16 -> 8x8
            conv_block(128, 256),   # 8x8  -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==========================
#  Training & Evaluation
# ==========================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    train_path: str,
    val_path: str,
    output_weights: str = "model.pth",
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 3e-4,
    max_epochs: int = 50,
    patience: int = 6,
    num_workers: int = 0,
    seed: int = 4368,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = TinyImageNetCNN(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch}/{max_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
            torch.save(best_state, output_weights)
            print(f"  -> New best model saved to {output_weights} (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        best_state = model.state_dict()
        torch.save(best_state, output_weights)
        print("Saved last model state (no improvement over baseline).")

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")


# ==========================
#  Load & Predict (for instructor)
# ==========================

def load_model(weights_path: str = "model.pth") -> nn.Module:
    device = torch.device("cpu")
    model = TinyImageNetCNN(num_classes=200)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model: nn.Module, dataloader: DataLoader, device: torch.device = None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds: List[int] = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())

    return np.array(all_preds, dtype=np.int64)


# ==========================
#  CLI Entry Point
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyImageNet CNN for COSC 4368 Assignment 2")
    parser.add_argument("--train-path", type=str, default="train-70_.pkl",
                        help="Path to training .pkl file")
    parser.add_argument("--val-path", type=str, default="validation-10_.pkl",
                        help="Path to validation .pkl file")
    parser.add_argument("--output-weights", type=str, default="model.pth",
                        help="Where to save best model weights")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=3e-4,
                        help="Weight decay (L2 penalty)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=6,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=4368,
                        help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== COSC 4368 - CNN Design Challenge Training ===")
    print(f"Train file: {args.train_path}")
    print(f"Val file  : {args.val_path}")
    print(f"Saving best weights to: {args.output_weights}")

    train_model(
        train_path=args.train_path,
        val_path=args.val_path,
        output_weights=args.output_weights,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )


#from google.colab import files
#files.download("model.pth")