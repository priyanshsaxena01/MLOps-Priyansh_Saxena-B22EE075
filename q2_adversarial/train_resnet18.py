"""
Q2(i): Train ResNet-18 from scratch on clean CIFAR-10.
Target: >= 72% test accuracy.

Usage:
    python train_resnet18.py --epochs 50 --batch_size 128
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from dataset import get_cifar10_dataloaders
from model import create_resnet18
from utils import setup_wandb, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    parser.add_argument("--weights_dir", type=str, default="./q2_adversarial/weights")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         acc=f"{100.*correct/total:.1f}%")

    return running_loss / total, 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Model
    model = create_resnet18(num_classes=10).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # WandB
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q2_ResNet18_CIFAR10_Training",
        config=vars(args)
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss, "val/loss": val_loss,
            "train/accuracy": train_acc, "val/accuracy": val_acc,
            "lr": optimizer.param_groups[0]['lr'],
        }, step=epoch)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, os.path.join(args.weights_dir, "resnet18_cifar10_best.pth"))
            print(f"  ✓ Best model saved ({val_acc:.2f}%)")

    # Final test
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\n[Test] Accuracy: {test_acc:.2f}%")
    wandb.log({"test/accuracy": test_acc})

    if test_acc < 72.0:
        print(f"[WARNING] Test accuracy {test_acc:.2f}% is below 72% target!")
    else:
        print(f"[SUCCESS] Test accuracy {test_acc:.2f}% meets >= 72% target!")

    # Save final model too
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'test_acc': test_acc,
    }, os.path.join(args.weights_dir, "resnet18_cifar10_final.pth"))

    # Plot
    plot_training_curves(history,
                         os.path.join(args.output_dir, "resnet18_training_curves.png"),
                         "ResNet-18 CIFAR-10")

    # Save history
    with open(os.path.join(args.output_dir, "resnet18_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    wandb.finish()
    print(f"\n[Done] Best Val: {best_val_acc:.2f}%, Test: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
