"""
Q1: ViT-S Fine-tuning on CIFAR-100 with/without LoRA.
Runs all experiments: no-LoRA baseline + 9 LoRA combinations (Rank x Alpha).

Usage:
    # Run a single experiment
    python train.py --use_lora --rank 4 --alpha 8 --dropout 0.1 --epochs 10

    # Run all experiments (no-LoRA + 9 LoRA combos)
    python train.py --run_all --epochs 10
"""

import os
import sys
import argparse
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from dataset import get_cifar100_dataloaders
from model import (create_vit_model, freeze_base_model, apply_lora,
                   get_trainable_params, get_lora_params)
from utils import (setup_wandb, log_epoch_metrics, compute_gradient_norms,
                   plot_training_curves, plot_gradient_norms,
                   generate_epoch_table, save_results_json)


def parse_args():
    parser = argparse.ArgumentParser(description="ViT-S CIFAR-100 Fine-tuning with LoRA")

    # Experiment settings
    parser.add_argument("--run_all", action="store_true",
                        help="Run all experiments (baseline + 9 LoRA combos)")
    parser.add_argument("--use_lora", action="store_true",
                        help="Apply LoRA to the model")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout")

    # Training settings
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (32 for 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")

    # Paths
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./q1_vit_lora/outputs",
                        help="Output directory")
    parser.add_argument("--weights_dir", type=str, default="./q1_vit_lora/weights",
                        help="Weights directory")

    return parser.parse_args()


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
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

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def run_experiment(args, use_lora, rank=None, alpha=None, dropout=0.1,
                   experiment_id=0, train_loader=None, val_loader=None):
    """
    Run a single training experiment.

    Args:
        args: Command-line arguments
        use_lora: Whether to apply LoRA
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        experiment_id: Experiment number
        train_loader: Training data loader
        val_loader: Validation data loader

    Returns:
        result dict with metrics and history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}")

    if use_lora:
        exp_name = f"Exp{experiment_id}_LoRA_r{rank}_a{alpha}_d{dropout}"
        print(f"[Experiment {experiment_id}] LoRA: Rank={rank}, Alpha={alpha}, "
              f"Dropout={dropout}")
    else:
        exp_name = f"Exp{experiment_id}_NoLoRA_HeadOnly"
        print(f"[Experiment {experiment_id}] No LoRA (Head-only fine-tuning)")

    print(f"{'=' * 70}")

    # Create model
    model = create_vit_model(num_classes=100, pretrained=True)

    if use_lora:
        model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    else:
        model = freeze_base_model(model)

    model = model.to(device)
    trainable_params, total_params, param_pct = get_trainable_params(model)

    # Setup WandB
    config = {
        "experiment_id": experiment_id,
        "use_lora": use_lora,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    run = setup_wandb(
        project_name="Assignment-5",
        run_name=exp_name,
        config=config
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    all_grad_norms = []
    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Track gradient norms for LoRA weights
        grad_norms = {}
        if use_lora:
            grad_norms = compute_gradient_norms(model)
            all_grad_norms.append(grad_norms)

        # Log to WandB
        log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc, grad_norms)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.weights_dir, exist_ok=True)
            weight_path = os.path.join(args.weights_dir, f"{exp_name}_best.pth")
            if use_lora:
                # Save only LoRA weights + head
                state_dict = {k: v for k, v in model.state_dict().items()
                              if 'lora_' in k or 'head' in k}
            else:
                state_dict = {k: v for k, v in model.state_dict().items()
                              if 'head' in k}
            torch.save({
                'model_state_dict': state_dict,
                'config': config,
                'epoch': epoch,
                'val_acc': val_acc,
            }, weight_path)
            print(f"  ✓ Best model saved ({val_acc:.2f}%)")

    # Plot training curves
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_curves(
        history,
        os.path.join(args.output_dir, f"{exp_name}_curves.png"),
        exp_name
    )

    # Plot gradient norms
    if use_lora and all_grad_norms:
        plot_gradient_norms(
            all_grad_norms,
            os.path.join(args.output_dir, f"{exp_name}_grad_norms.png"),
            exp_name
        )

    # Generate epoch table
    epoch_table = generate_epoch_table(history, exp_name)
    table_path = os.path.join(args.output_dir, f"{exp_name}_epoch_table.md")
    with open(table_path, 'w') as f:
        f.write(epoch_table)

    # Log epoch table to WandB
    wandb.log({"epoch_table": wandb.Table(
        columns=["Epoch", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)"],
        data=[[i + 1, history['train_loss'][i], history['val_loss'][i],
               history['train_acc'][i], history['val_acc'][i]]
              for i in range(len(history['train_loss']))]
    )})

    wandb.finish()

    result = {
        "experiment_id": experiment_id,
        "experiment_name": exp_name,
        "lora": use_lora,
        "rank": rank if use_lora else None,
        "alpha": alpha if use_lora else None,
        "dropout": dropout if use_lora else None,
        "best_val_acc": best_val_acc,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "history": history,
        "weight_path": os.path.join(args.weights_dir, f"{exp_name}_best.pth"),
    }

    print(f"\n[Experiment {experiment_id}] Completed. Best Val Acc: {best_val_acc:.2f}%")
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data once
    train_loader, val_loader, test_loader, class_names = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    all_results = []

    if args.run_all:
        # Experiment 0: No LoRA baseline
        result = run_experiment(
            args, use_lora=False, experiment_id=0,
            train_loader=train_loader, val_loader=val_loader
        )
        all_results.append(result)

        # Experiments 1-9: LoRA with all rank x alpha combinations
        ranks = [2, 4, 8]
        alphas = [2, 4, 8]
        exp_id = 1
        for rank in ranks:
            for alpha in alphas:
                result = run_experiment(
                    args, use_lora=True, rank=rank, alpha=alpha,
                    dropout=args.dropout, experiment_id=exp_id,
                    train_loader=train_loader, val_loader=val_loader
                )
                all_results.append(result)
                exp_id += 1

                # Clear GPU cache between experiments
                torch.cuda.empty_cache()

    else:
        # Run a single experiment
        result = run_experiment(
            args, use_lora=args.use_lora, rank=args.rank, alpha=args.alpha,
            dropout=args.dropout, experiment_id=0,
            train_loader=train_loader, val_loader=val_loader
        )
        all_results.append(result)

    # Save all results
    save_path = os.path.join(args.output_dir, "all_results.json")
    serializable_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items()}
        serializable_results.append(sr)
    save_results_json(serializable_results, save_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    for r in all_results:
        lora_str = f"LoRA(r={r['rank']},a={r['alpha']})" if r['lora'] else "No LoRA"
        print(f"  Exp {r['experiment_id']}: {lora_str} | "
              f"Val Acc: {r['best_val_acc']:.2f}% | "
              f"Params: {r['trainable_params']:,}")


if __name__ == "__main__":
    main()
