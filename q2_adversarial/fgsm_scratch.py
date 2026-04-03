"""
Q2(i): FGSM Attack Implementation from Scratch (without IBM ART).

Usage:
    python fgsm_scratch.py --model_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_cifar10_dataloaders
from model import create_resnet18


def parse_args():
    parser = argparse.ArgumentParser(description="FGSM Attack from Scratch")
    parser.add_argument("--model_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--epsilons", type=float, nargs='+',
                        default=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    return parser.parse_args()


def fgsm_attack(images, epsilon, data_grad):
    """
    Fast Gradient Sign Method (FGSM) attack.

    Perturbs input images by adding epsilon * sign(gradient) to maximize loss.

    Args:
        images: Original images tensor (B, C, H, W)
        epsilon: Perturbation magnitude
        data_grad: Gradient of loss w.r.t. input images

    Returns:
        perturbed_images: Adversarial images clipped to [0, 1]
    """
    # Collect the sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create perturbed image by adjusting input image
    perturbed_images = images + epsilon * sign_data_grad

    # Clip to maintain valid pixel range [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)

    return perturbed_images


def evaluate_with_fgsm(model, test_loader, epsilon, device):
    """
    Evaluate model under FGSM attack at given epsilon.

    Returns:
        accuracy: Accuracy on adversarial examples
        adv_examples: List of (original, adversarial, true_label, pred_label) tuples
    """
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    adv_examples = []

    for images, labels in tqdm(test_loader, desc=f"FGSM ε={epsilon:.3f}", leave=False):
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Zero gradients
        model.zero_grad()

        # Backward pass to get gradients w.r.t. input
        loss.backward()

        # Get gradients
        data_grad = images.grad.data

        # Generate adversarial examples
        perturbed = fgsm_attack(images, epsilon, data_grad)

        # Re-evaluate
        with torch.no_grad():
            adv_outputs = model(perturbed)
            _, adv_predicted = adv_outputs.max(1)

        total += labels.size(0)
        correct += adv_predicted.eq(labels).sum().item()

        # Save some examples for visualization
        if len(adv_examples) < 50:
            for i in range(min(5, images.size(0))):
                adv_examples.append((
                    images[i].detach().cpu().numpy(),
                    perturbed[i].detach().cpu().numpy(),
                    labels[i].item(),
                    adv_predicted[i].item(),
                ))

    accuracy = 100. * correct / total
    return accuracy, adv_examples


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = create_resnet18(num_classes=10).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Model] Loaded from {args.model_path}")

    # Data
    _, _, test_loader, class_names = get_cifar10_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )

    # Run FGSM for each epsilon
    results = {}
    all_examples = {}

    print("\n" + "=" * 50)
    print("FGSM Attack (From Scratch)")
    print("=" * 50)

    for eps in args.epsilons:
        acc, examples = evaluate_with_fgsm(model, test_loader, eps, device)
        results[str(eps)] = acc
        all_examples[str(eps)] = examples
        print(f"  ε = {eps:.3f} | Accuracy = {acc:.2f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fgsm_scratch_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save adversarial examples for visualization
    np.savez(
        os.path.join(args.output_dir, "fgsm_scratch_examples.npz"),
        **{f"eps_{eps}_{field}": np.array([
            ex[idx] for ex in all_examples[str(eps)]
        ]) for eps in args.epsilons for idx, field in
            enumerate(['original', 'adversarial', 'true_label', 'pred_label'])}
    )

    print(f"\n[Done] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
