"""
Q1: Optuna Hyperparameter Optimization for LoRA on ViT-S.
Searches for the best LoRA rank, alpha, and learning rate.

Usage:
    python optuna_search.py --n_trials 20 --epochs_per_trial 5
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import optuna
from optuna.trial import TrialState
import wandb

from dataset import get_cifar100_dataloaders
from model import create_vit_model, apply_lora, get_trainable_params
from utils import setup_wandb, push_to_huggingface, save_results_json


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna HPO for LoRA")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--epochs_per_trial", type=int, default=5,
                        help="Epochs per trial (shorter for speed)")
    parser.add_argument("--epochs_final", type=int, default=10,
                        help="Epochs for final training with best config")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q1_vit_lora/outputs")
    parser.add_argument("--weights_dir", type=str, default="./q1_vit_lora/weights")
    parser.add_argument("--hf_repo", type=str, default="b22ee075/vit-s-lora-cifar100",
                        help="HuggingFace model repo")
    return parser.parse_args()


def create_objective(train_loader, val_loader, args):
    """Create an Optuna objective function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        # Sample hyperparameters
        rank = trial.suggest_categorical("rank", [2, 4, 8, 16, 32])
        alpha = trial.suggest_categorical("alpha", [2, 4, 8, 16, 32])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        trial_name = f"optuna_trial{trial.number}_r{rank}_a{alpha}"
        print(f"\n[Trial {trial.number}] rank={rank}, alpha={alpha}, "
              f"lr={lr:.6f}, dropout={dropout:.3f}")

        try:
            # Create model
            model = create_vit_model(num_classes=100, pretrained=True)
            model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=0.01
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs_per_trial
            )
            scaler = GradScaler()

            best_val_acc = 0.0

            for epoch in range(1, args.epochs_per_trial + 1):
                # Train
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()

                # Validate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        with autocast():
                            outputs = model(images)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                val_acc = 100. * correct / total
                best_val_acc = max(best_val_acc, val_acc)
                print(f"  Epoch {epoch}: Val Acc = {val_acc:.2f}%")

                # Report to Optuna for pruning
                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Trial {trial.number} ran out of GPU memory. Pruning.")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        finally:
            del model
            torch.cuda.empty_cache()

        return best_val_acc

    return objective


def train_best_config(best_params, train_loader, val_loader, test_loader, args):
    """
    Retrain the model with the best Optuna config for full epochs.
    Push to HuggingFace.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = best_params['rank']
    alpha = best_params['alpha']
    lr = best_params['lr']
    dropout = best_params['dropout']

    exp_name = f"Optuna_Best_r{rank}_a{alpha}"
    print(f"\n{'=' * 70}")
    print(f"[Optuna Best] Retraining with rank={rank}, alpha={alpha}, "
          f"lr={lr:.6f}, dropout={dropout:.3f}")
    print(f"[Optuna Best] Training for {args.epochs_final} epochs")
    print(f"{'=' * 70}")

    run = setup_wandb(
        project_name="Assignment-5",
        run_name=f"Optuna_Best_{exp_name}",
        config=best_params
    )

    model = create_vit_model(num_classes=100, pretrained=True)
    model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_final)
    scaler = GradScaler()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs_final + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        tl = train_loss / train_total
        ta = 100. * train_correct / train_total

        # Validate
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

        vl = val_loss / val_total
        va = 100. * val_correct / val_total

        wandb.log({
            "epoch": epoch, "train/loss": tl, "val/loss": vl,
            "train/accuracy": ta, "val/accuracy": va
        }, step=epoch)

        print(f"  Epoch {epoch}: Train {ta:.2f}% | Val {va:.2f}%")

        if va > best_val_acc:
            best_val_acc = va
            os.makedirs(args.weights_dir, exist_ok=True)
            weight_path = os.path.join(args.weights_dir, f"{exp_name}_best.pth")
            state_dict = {k: v for k, v in model.state_dict().items()
                          if 'lora_' in k or 'head' in k}
            torch.save({
                'model_state_dict': state_dict,
                'config': best_params,
                'epoch': epoch,
                'val_acc': va,
            }, weight_path)

    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = 100. * test_correct / test_total
    wandb.log({"test/accuracy": test_acc})
    print(f"\n[Optuna Best] Test Accuracy: {test_acc:.2f}%")

    # Push to HuggingFace
    weight_path = os.path.join(args.weights_dir, f"{exp_name}_best.pth")
    if os.path.exists(weight_path):
        try:
            push_to_huggingface(
                weight_path,
                repo_id=args.hf_repo,
                commit_message=f"Best Optuna model: rank={rank}, alpha={alpha}, "
                               f"test_acc={test_acc:.2f}%"
            )
        except Exception as e:
            print(f"[HF] Upload failed: {e}")
            print("[HF] You can manually upload later.")

    wandb.finish()

    return {
        'best_params': best_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'weight_path': weight_path,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load data
    train_loader, val_loader, test_loader, _ = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="vit_lora_cifar100",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    objective = create_objective(train_loader, val_loader, args)

    print(f"\n[Optuna] Starting {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    print("\n" + "=" * 70)
    print("OPTUNA RESULTS")
    print("=" * 70)

    pruned = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"  Completed trials: {len(complete)}")
    print(f"  Pruned trials: {len(pruned)}")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best value (Val Acc): {study.best_value:.2f}%")
    print(f"  Best params: {study.best_params}")

    # Save study results
    os.makedirs(args.output_dir, exist_ok=True)
    save_results_json(
        {"best_params": study.best_params, "best_value": study.best_value,
         "n_trials": len(study.trials)},
        os.path.join(args.output_dir, "optuna_results.json")
    )

    # Retrain with best config
    best_result = train_best_config(
        study.best_params, train_loader, val_loader, test_loader, args
    )

    print(f"\n[Done] Best Optuna model: Test Acc = {best_result['test_acc']:.2f}%")
    print(f"[Done] Weights saved to: {best_result['weight_path']}")


if __name__ == "__main__":
    main()
