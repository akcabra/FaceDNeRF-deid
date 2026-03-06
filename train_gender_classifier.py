"""Train a ResNet-18 binary gender classifier on CelebA.

Produces a checkpoint compatible with the GenderClassifier class in
criteria/attr_loss.py (ResNet-18, 2-class output, ~11.2M params).

Usage
-----
    python train_gender_classifier.py --data_dir ./data/celeba

CelebA will be auto-downloaded by torchvision on the first run
(~1.4 GB images + annotations). If automatic download fails,
manually download from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
and place files under <data_dir>/celeba/.

The best model (by validation accuracy) is saved to
    ./networks/gender_classifier.pth
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CelebA


# CelebA attribute index for "Male"
MALE_ATTR_IDX = 20


class CelebAGender(CelebA):
    def _check_integrity(self) -> bool:
        """Skip MD5 check — files from Kaggle have different hashes."""
        for _, _, filename, _ in self.file_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
        return True

    def __getitem__(self, index):
        img, attrs = super().__getitem__(index)
        gender = attrs[MALE_ATTR_IDX].long()  # 0 = female, 1 = male
        return img, gender


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = CelebAGender(root=data_dir, split="train",
                            target_type="attr",
                            transform=train_transform,
                            download=False)
    val_ds = CelebAGender(root=data_dir, split="valid",
                          target_type="attr",
                          transform=val_transform,
                          download=False)
    test_ds = CelebAGender(root=data_dir, split="test",
                           target_type="attr",
                           transform=val_transform,
                           download=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader


def build_model(pretrained: bool = True) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 gender classifier on CelebA")
    parser.add_argument("--data_dir", type=str, default="./data/celeba",
                        help="Root dir for CelebA dataset")
    parser.add_argument("--save_path", type=str,
                        default="./networks/gender_classifier.pth",
                        help="Where to save the best model")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train from scratch (no ImageNet init)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}"
                          if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Loading CelebA dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir, args.batch_size, args.num_workers)
    print(f"  Train: {len(train_loader.dataset):,} images")
    print(f"  Val:   {len(val_loader.dataset):,} images")
    print(f"  Test:  {len(test_loader.dataset):,} images")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(pretrained=not args.no_pretrained).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"[{elapsed:.1f}s]")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f}  Test accuracy={test_acc:.4f}")
    print(f"\nBest checkpoint saved at: {args.save_path}")


if __name__ == "__main__":
    main()
