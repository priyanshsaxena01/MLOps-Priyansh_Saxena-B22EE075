"""
Q2(i): Compare FGSM from Scratch vs IBM ART.
Generates visual comparisons and accuracy tables.

Usage:
    python compare_fgsm.py
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from dataset import get_cifar10_numpy, get_cifar10_dataloaders
from model import create_resnet18
from utils import (setup_wandb, plot_adversarial_comparison,
                   plot_eps_vs_accuracy, log_attack_samples_to_wandb)


# Manual FGSM implementation (same as fgsm_scratch.py)
def fgsm_attack_batch(model, images, labels, epsilon, criterion, device):
    """Apply FGSM attack to a batch."""
    images = images.clone().to(device).requires_grad_(True)
    labels = labels.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbed = images + epsilon * images.grad.data.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    return perturbed


def parse_args():
    parser = argparse.ArgumentParser(description="Compare FGSM: Scratch vs ART")
    parser.add_argument("--model_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Epsilon for visual comparison")
    parser.add_argument("--all_epsilons", type=float, nargs='+',
                        default=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = create_resnet18(num_classes=10).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    _, _, test_loader, class_names = get_cifar10_dataloaders(
        batch_size=128, data_dir=args.data_dir
    )
    x_test_np, y_test_np, _, _ = get_cifar10_numpy(data_dir=args.data_dir)

    # ART setup
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod

    criterion = nn.CrossEntropyLoss()
    art_classifier = PyTorchClassifier(
        model=model, loss=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Setup WandB
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q2_FGSM_Comparison",
        config=vars(args)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ===== Accuracy comparison across epsilons =====
    print("\n" + "=" * 60)
    print("FGSM COMPARISON: Scratch vs IBM ART")
    print("=" * 60)

    # Clean accuracy
    clean_preds = art_classifier.predict(x_test_np[:10000], batch_size=128)
    clean_acc = 100. * np.mean(np.argmax(clean_preds, axis=1) == y_test_np[:10000])
    print(f"Clean Accuracy: {clean_acc:.2f}%")

    accs_scratch = []
    accs_art = []

    for eps in args.all_epsilons:
        if eps == 0.0:
            accs_scratch.append(clean_acc)
            accs_art.append(clean_acc)
            continue

        # --- From Scratch ---
        correct_s = 0
        total_s = 0
        for images, labels in test_loader:
            perturbed = fgsm_attack_batch(model, images, labels, eps, criterion, device)
            with torch.no_grad():
                outputs = model(perturbed)
                _, predicted = outputs.max(1)
            total_s += labels.size(0)
            correct_s += predicted.eq(labels.to(device)).sum().item()
        acc_s = 100. * correct_s / total_s
        accs_scratch.append(acc_s)

        # --- IBM ART ---
        attack = FastGradientMethod(estimator=art_classifier, eps=eps, batch_size=128)
        x_adv = attack.generate(x=x_test_np[:10000])
        adv_preds = art_classifier.predict(x_adv, batch_size=128)
        acc_a = 100. * np.mean(np.argmax(adv_preds, axis=1) == y_test_np[:10000])
        accs_art.append(acc_a)

        print(f"  ε={eps:.3f} | Scratch: {acc_s:.2f}% | ART: {acc_a:.2f}%")

    # Plot epsilon vs accuracy
    plot_eps_vs_accuracy(
        args.all_epsilons, clean_acc, accs_scratch, accs_art,
        os.path.join(args.output_dir, "fgsm_eps_vs_accuracy.png")
    )

    # ===== Visual comparison for default epsilon =====
    eps = args.epsilon
    print(f"\n--- Visual comparison at ε={eps} ---")

    # Get a batch for visual comparison
    images_batch, labels_batch = next(iter(test_loader))
    n = args.num_samples

    # Scratch FGSM
    perturbed_scratch = fgsm_attack_batch(
        model, images_batch[:n], labels_batch[:n], eps, criterion, device
    )
    with torch.no_grad():
        pred_orig = model(images_batch[:n].to(device)).argmax(1).cpu().numpy()
        pred_scratch = model(perturbed_scratch).argmax(1).cpu().numpy()

    # ART FGSM
    attack = FastGradientMethod(estimator=art_classifier, eps=eps, batch_size=n)
    x_batch_np = images_batch[:n].numpy()
    x_adv_art = attack.generate(x=x_batch_np)
    pred_art = np.argmax(art_classifier.predict(x_adv_art), axis=1)

    # Plot comparison
    plot_adversarial_comparison(
        images_batch[:n].numpy(),
        perturbed_scratch.cpu().numpy(),
        x_adv_art,
        labels_batch[:n].numpy(),
        pred_orig,
        pred_scratch,
        pred_art,
        class_names,
        os.path.join(args.output_dir, f"fgsm_comparison_eps{eps}.png"),
        num_samples=min(n, 5)
    )

    # ===== Log samples to WandB =====
    log_attack_samples_to_wandb({
        "clean": (images_batch[:10].numpy(), labels_batch[:10].numpy(), pred_orig[:10]),
        "FGSM_scratch": (perturbed_scratch[:10].cpu().numpy(),
                         labels_batch[:10].numpy(), pred_scratch[:10]),
        "FGSM_ART": (x_adv_art[:10], labels_batch[:10].numpy(), pred_art[:10]),
    }, class_names, num_samples=10)

    # ===== Generate comparison table =====
    table_data = []
    for i, eps in enumerate(args.all_epsilons):
        table_data.append([eps, clean_acc, accs_scratch[i], accs_art[i],
                          clean_acc - accs_scratch[i], clean_acc - accs_art[i]])

    wandb.log({"fgsm_comparison": wandb.Table(
        columns=["Epsilon", "Clean Acc (%)", "FGSM Scratch (%)",
                 "FGSM ART (%)", "Drop (Scratch)", "Drop (ART)"],
        data=table_data
    )})

    # Save comparison results
    comparison = {
        "epsilons": args.all_epsilons,
        "clean_accuracy": clean_acc,
        "scratch_accuracies": accs_scratch,
        "art_accuracies": accs_art,
    }
    with open(os.path.join(args.output_dir, "fgsm_comparison_results.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print markdown table
    print("\n### FGSM Comparison Table")
    print("| Epsilon | Clean (%) | Scratch (%) | ART (%) | Drop (Scratch) | Drop (ART) |")
    print("|---------|-----------|-------------|---------|----------------|------------|")
    for i, eps in enumerate(args.all_epsilons):
        print(f"| {eps:.3f} | {clean_acc:.2f} | {accs_scratch[i]:.2f} | "
              f"{accs_art[i]:.2f} | {clean_acc - accs_scratch[i]:.2f} | "
              f"{clean_acc - accs_art[i]:.2f} |")

    wandb.finish()
    print(f"\n[Done] Comparison complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
