"""
Q2(ii-b): Adversarial Detector using BIM attack via IBM ART.
Train ResNet-34 binary classifier to detect BIM adversarial images.

Usage:
    python train_detector_bim.py --resnet18_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import BasicIterativeMethod

from dataset import get_cifar10_numpy
from model import create_resnet18, create_resnet34
from utils import setup_wandb, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train BIM adversarial detector")
    parser.add_argument("--resnet18_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--bim_eps", type=float, default=0.03)
    parser.add_argument("--bim_eps_step", type=float, default=0.007)
    parser.add_argument("--bim_max_iter", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    parser.add_argument("--weights_dir", type=str, default="./q2_adversarial/weights")
    return parser.parse_args()


def generate_bim_adversarial(model, x_data, args, device):
    """Generate BIM adversarial examples using ART."""
    criterion = nn.CrossEntropyLoss()
    art_classifier = PyTorchClassifier(
        model=model, loss=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    attack = BasicIterativeMethod(
        estimator=art_classifier,
        eps=args.bim_eps,
        eps_step=args.bim_eps_step,
        max_iter=args.bim_max_iter,
        batch_size=128,
    )

    print(f"[BIM] Generating adversarial examples "
          f"(eps={args.bim_eps}, steps={args.bim_max_iter})...")
    x_adv = attack.generate(x=x_data)
    print(f"[BIM] Generated {x_adv.shape[0]} adversarial samples")

    return x_adv


def create_detector_dataset(x_clean, x_adv, val_split=0.1):
    """Create mixed clean+adversarial dataset. 0=clean, 1=adversarial."""
    n = min(len(x_clean), len(x_adv))
    x_clean = x_clean[:n]
    x_adv = x_adv[:n]

    x_combined = np.concatenate([x_clean, x_adv], axis=0)
    y_combined = np.concatenate([
        np.zeros(n, dtype=np.int64),
        np.ones(n, dtype=np.int64),
    ])

    perm = np.random.RandomState(42).permutation(len(x_combined))
    x_combined = x_combined[perm]
    y_combined = y_combined[perm]

    split_idx = int(len(x_combined) * (1 - val_split))
    x_train, x_val = x_combined[:split_idx], x_combined[split_idx:]
    y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]

    print(f"[Dataset] Train: {len(x_train)} (clean: {(y_train==0).sum()}, "
          f"adv: {(y_train==1).sum()})")
    print(f"[Dataset] Val: {len(x_val)} (clean: {(y_val==0).sum()}, "
          f"adv: {(y_val==1).sum()})")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(y_val).long()
    )

    return train_dataset, val_dataset


def train_detector(model, train_loader, val_loader, args, device):
    """Train the binary detector model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
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

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss, "val/loss": val_loss,
            "train/accuracy": train_acc, "val/accuracy": val_acc,
        }, step=epoch)

        print(f"  Epoch {epoch}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.weights_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, os.path.join(args.weights_dir, f"detector_bim_best.pth"))

    return history, best_val_acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load target model
    target_model = create_resnet18(num_classes=10)
    ckpt = torch.load(args.resnet18_path, map_location='cpu')
    target_model.load_state_dict(ckpt['model_state_dict'])
    target_model = target_model.to(device)
    target_model.eval()
    print(f"[Target Model] Loaded ResNet-18")

    # Load data
    x_train, y_train, x_test, y_test = get_cifar10_numpy(data_dir=args.data_dir)

    # Generate BIM adversarial examples
    x_adv_train = generate_bim_adversarial(target_model, x_train, args, device)

    # Save adversarial examples
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "bim_adv_train.npy"), x_adv_train[:1000])

    # Create detector dataset
    train_dataset, val_dataset = create_detector_dataset(x_train, x_adv_train)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # Create detector model
    detector = create_resnet34(num_classes=2).to(device)

    # WandB
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q2_Detector_BIM",
        config={**vars(args), "attack": "BIM"}
    )

    # Train
    history, best_val_acc = train_detector(
        detector, train_loader, val_loader, args, device
    )

    # Plot
    plot_training_curves(
        history,
        os.path.join(args.output_dir, "detector_bim_curves.png"),
        "BIM Detector (ResNet-34)"
    )

    if best_val_acc >= 70.0:
        print(f"\n[SUCCESS] BIM Detector: {best_val_acc:.2f}% >= 70% target!")
    else:
        print(f"\n[WARNING] BIM Detector: {best_val_acc:.2f}% < 70% target!")

    with open(os.path.join(args.output_dir, "detector_bim_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
