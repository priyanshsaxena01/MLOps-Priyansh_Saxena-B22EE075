"""
Q1: Test all trained ViT-S models and generate results tables.

Usage:
    python test.py --weights_dir ./q1_vit_lora/weights --output_dir ./q1_vit_lora/outputs
"""

import os
import sys
import argparse
import json
import glob
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
import wandb

from dataset import get_cifar100_dataloaders
from model import create_vit_model, apply_lora, freeze_base_model, get_trainable_params
from utils import (setup_wandb, plot_classwise_histogram,
                   generate_results_table, save_results_json)


def parse_args():
    parser = argparse.ArgumentParser(description="Test ViT-S models on CIFAR-100")
    parser.add_argument("--weights_dir", type=str, default="./q1_vit_lora/weights")
    parser.add_argument("--output_dir", type=str, default="./q1_vit_lora/outputs")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


@torch.no_grad()
def test_model(model, test_loader, device, num_classes=100):
    """
    Test the model and compute overall + class-wise accuracy.

    Returns:
        overall_acc: Overall test accuracy
        class_accs: Per-class accuracy list
    """
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in tqdm(test_loader, desc="Testing", leave=False):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i].item() == label:
                class_correct[label] += 1

    overall_acc = 100. * correct / total
    class_accs = [
        100. * class_correct[i] / max(class_total[i], 1)
        for i in range(num_classes)
    ]

    return overall_acc, class_accs


def load_and_test_model(weight_path, test_loader, device, class_names):
    """Load a saved model checkpoint and test it."""
    checkpoint = torch.load(weight_path, map_location=device)
    config = checkpoint['config']

    use_lora = config['use_lora']
    rank = config.get('rank')
    alpha = config.get('alpha')
    dropout = config.get('dropout', 0.1)

    # Recreate model
    model = create_vit_model(num_classes=100, pretrained=True)

    if use_lora:
        model = apply_lora(model, rank=rank, alpha=alpha, dropout=dropout)
    else:
        model = freeze_base_model(model)

    # Load weights
    model_state = checkpoint['model_state_dict']
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"  Loaded weights: {len(model_state)} tensors "
          f"(missing: {len(missing)}, unexpected: {len(unexpected)})")

    model = model.to(device)
    trainable, total_p, _ = get_trainable_params(model)

    # Test
    overall_acc, class_accs = test_model(model, test_loader, device)

    return {
        'lora': use_lora,
        'rank': rank,
        'alpha': alpha,
        'dropout': dropout,
        'test_acc': overall_acc,
        'trainable_params': trainable,
        'class_accs': class_accs,
        'config': config,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # Load data
    _, _, test_loader, class_names = get_cifar100_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Find all weight files
    weight_files = sorted(glob.glob(os.path.join(args.weights_dir, "*_best.pth")))
    if not weight_files:
        print(f"[Error] No weight files found in {args.weights_dir}")
        sys.exit(1)

    print(f"\n[Test] Found {len(weight_files)} model checkpoints\n")

    # Setup WandB for test results
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q1_Test_Results",
        config={"phase": "testing"}
    )

    all_results = []
    best_result = None
    best_acc = 0

    for wf in weight_files:
        exp_name = os.path.basename(wf).replace("_best.pth", "")
        print(f"\n--- Testing: {exp_name} ---")

        result = load_and_test_model(wf, test_loader, device, class_names)
        result['experiment_name'] = exp_name
        all_results.append(result)

        lora_str = (f"LoRA(r={result['rank']},a={result['alpha']})"
                    if result['lora'] else "No LoRA")
        print(f"  {lora_str} | Test Acc: {result['test_acc']:.2f}% | "
              f"Params: {result['trainable_params']:,}")

        if result['test_acc'] > best_acc:
            best_acc = result['test_acc']
            best_result = result

        # Plot class-wise histogram for each experiment
        plot_classwise_histogram(
            result['class_accs'],
            class_names,
            os.path.join(args.output_dir, f"{exp_name}_classwise.png"),
            exp_name
        )

        torch.cuda.empty_cache()

    # Generate summary table
    table_str = generate_results_table(
        all_results,
        os.path.join(args.output_dir, "test_results_table.md")
    )
    print(f"\n{'=' * 70}")
    print("TEST RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(table_str)

    # Log summary table to WandB
    wandb_data = []
    for r in all_results:
        wandb_data.append([
            "With" if r['lora'] else "Without",
            str(r.get('rank', '-')),
            str(r.get('alpha', '-')),
            str(r.get('dropout', '-')),
            f"{r['test_acc']:.2f}",
            f"{r['trainable_params']:,}"
        ])

    wandb.log({"test_results": wandb.Table(
        columns=["LoRA", "Rank", "Alpha", "Dropout",
                 "Test Accuracy (%)", "Trainable Params"],
        data=wandb_data
    )})

    # Save results
    test_results = [{k: v for k, v in r.items() if k != 'class_accs'}
                    for r in all_results]
    save_results_json(test_results, os.path.join(args.output_dir, "test_results.json"))

    # Also save class-wise accuracies separately
    classwise_results = {r['experiment_name']: r['class_accs'] for r in all_results}
    save_results_json(classwise_results,
                      os.path.join(args.output_dir, "classwise_results.json"))

    print(f"\n[Best] {best_result['experiment_name']} with "
          f"Test Acc: {best_acc:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
