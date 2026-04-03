"""
Q2(i): FGSM Attack using IBM Adversarial Robustness Toolbox (ART).

Usage:
    python fgsm_art.py --model_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from dataset import get_cifar10_numpy
from model import create_resnet18


def parse_args():
    parser = argparse.ArgumentParser(description="FGSM Attack using IBM ART")
    parser.add_argument("--model_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--epsilons", type=float, nargs='+',
                        default=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load model
    model = create_resnet18(num_classes=10)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"[Model] Loaded from {args.model_path}")

    # Wrap model in ART classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    art_classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Load test data as numpy (ART requires numpy arrays)
    _, _, x_test, y_test = get_cifar10_numpy(data_dir=args.data_dir)

    # Use a subset for speed (full test set = 10000)
    print(f"[Data] Test set: {x_test.shape}")

    # Evaluate on clean data
    clean_preds = art_classifier.predict(x_test, batch_size=128)
    clean_acc = 100. * np.mean(np.argmax(clean_preds, axis=1) == y_test)
    print(f"\n[Clean] Accuracy: {clean_acc:.2f}%")

    # Run FGSM for each epsilon
    results = {}
    all_adv_images = {}
    all_preds = {}

    print("\n" + "=" * 50)
    print("FGSM Attack (IBM ART)")
    print("=" * 50)

    for eps in args.epsilons:
        if eps == 0.0:
            # No attack
            results[str(eps)] = clean_acc
            all_preds[str(eps)] = np.argmax(clean_preds, axis=1)
            print(f"  ε = {eps:.3f} | Accuracy = {clean_acc:.2f}%")
            continue

        # Create FGSM attack
        attack = FastGradientMethod(
            estimator=art_classifier,
            eps=eps,
            eps_step=eps,  # Single step for FGSM
            batch_size=128,
        )

        # Generate adversarial examples
        print(f"  Generating adversarial examples for ε={eps}...")
        x_test_adv = attack.generate(x=x_test)

        # Evaluate
        adv_preds = art_classifier.predict(x_test_adv, batch_size=128)
        adv_acc = 100. * np.mean(np.argmax(adv_preds, axis=1) == y_test)
        results[str(eps)] = adv_acc
        all_preds[str(eps)] = np.argmax(adv_preds, axis=1)

        # Save adversarial images for visualization (first 50)
        all_adv_images[str(eps)] = x_test_adv[:50]

        print(f"  ε = {eps:.3f} | Accuracy = {adv_acc:.2f}%")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fgsm_art_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Save adversarial examples for comparison
    save_dict = {
        'clean_images': x_test[:50],
        'true_labels': y_test[:50],
        'clean_preds': np.argmax(clean_preds[:50], axis=1),
    }
    for eps_str, adv_imgs in all_adv_images.items():
        save_dict[f'adv_images_eps{eps_str}'] = adv_imgs
        save_dict[f'adv_preds_eps{eps_str}'] = all_preds[eps_str][:50]

    np.savez(os.path.join(args.output_dir, "fgsm_art_examples.npz"), **save_dict)

    print(f"\n[Done] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
