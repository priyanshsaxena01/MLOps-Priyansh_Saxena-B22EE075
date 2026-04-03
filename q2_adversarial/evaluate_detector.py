"""
Q2(ii): Evaluate both adversarial detectors (PGD and BIM trained).
Cross-evaluate: PGD detector on BIM samples and vice versa.

Usage:
    python evaluate_detector.py
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

from dataset import get_cifar10_numpy
from model import create_resnet18, create_resnet34
from utils import setup_wandb, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate adversarial detectors")
    parser.add_argument("--resnet18_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--pgd_detector_path", type=str,
                        default="./q2_adversarial/weights/detector_pgd_best.pth")
    parser.add_argument("--bim_detector_path", type=str,
                        default="./q2_adversarial/weights/detector_bim_best.pth")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./q2_adversarial/outputs")
    parser.add_argument("--pgd_eps", type=float, default=0.03)
    parser.add_argument("--bim_eps", type=float, default=0.03)
    return parser.parse_args()


@torch.no_grad()
def evaluate_detector(detector, x_clean, x_adv, device, batch_size=128):
    """Evaluate detector on clean + adversarial samples."""
    detector.eval()

    n = min(len(x_clean), len(x_adv))
    x_combined = np.concatenate([x_clean[:n], x_adv[:n]], axis=0)
    y_true = np.concatenate([np.zeros(n), np.ones(n)])

    dataset = TensorDataset(
        torch.from_numpy(x_combined).float(),
        torch.from_numpy(y_true).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        with autocast():
            outputs = detector(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = 100. * np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm, all_preds, all_labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load target model for generating test adversarial samples
    target_model = create_resnet18(num_classes=10)
    ckpt = torch.load(args.resnet18_path, map_location='cpu')
    target_model.load_state_dict(ckpt['model_state_dict'])
    target_model = target_model.to(device)
    target_model.eval()

    # ART wrapper
    criterion = nn.CrossEntropyLoss()
    art_classifier = PyTorchClassifier(
        model=target_model, loss=criterion,
        optimizer=torch.optim.SGD(target_model.parameters(), lr=0.01),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Load test data
    _, _, x_test, y_test = get_cifar10_numpy(data_dir=args.data_dir)

    # Generate test adversarial examples
    print("\n[Generate] PGD adversarial test set...")
    pgd_attack = ProjectedGradientDescent(
        estimator=art_classifier, eps=args.pgd_eps,
        eps_step=0.007, max_iter=40, batch_size=128
    )
    x_test_pgd = pgd_attack.generate(x=x_test)

    print("[Generate] BIM adversarial test set...")
    bim_attack = BasicIterativeMethod(
        estimator=art_classifier, eps=args.bim_eps,
        eps_step=0.007, max_iter=10, batch_size=128
    )
    x_test_bim = bim_attack.generate(x=x_test)

    # Load detectors
    pgd_detector = create_resnet34(num_classes=2)
    pgd_ckpt = torch.load(args.pgd_detector_path, map_location='cpu')
    pgd_detector.load_state_dict(pgd_ckpt['model_state_dict'])
    pgd_detector = pgd_detector.to(device)

    bim_detector = create_resnet34(num_classes=2)
    bim_ckpt = torch.load(args.bim_detector_path, map_location='cpu')
    bim_detector.load_state_dict(bim_ckpt['model_state_dict'])
    bim_detector = bim_detector.to(device)

    # WandB
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q2_Detector_Evaluation",
        config=vars(args)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ===== Evaluation Matrix =====
    print("\n" + "=" * 60)
    print("ADVERSARIAL DETECTOR EVALUATION")
    print("=" * 60)

    results = {}

    # PGD detector on PGD samples
    acc, cm, _, _ = evaluate_detector(pgd_detector, x_test, x_test_pgd, device)
    results['pgd_on_pgd'] = acc
    print(f"\n  PGD Detector on PGD attack: {acc:.2f}%")
    plot_confusion_matrix(cm, ["Clean", "Adversarial"],
                          os.path.join(args.output_dir, "cm_pgd_on_pgd.png"),
                          "PGD Detector → PGD Attack")

    # PGD detector on BIM samples
    acc, cm, _, _ = evaluate_detector(pgd_detector, x_test, x_test_bim, device)
    results['pgd_on_bim'] = acc
    print(f"  PGD Detector on BIM attack: {acc:.2f}%")
    plot_confusion_matrix(cm, ["Clean", "Adversarial"],
                          os.path.join(args.output_dir, "cm_pgd_on_bim.png"),
                          "PGD Detector → BIM Attack")

    # BIM detector on BIM samples
    acc, cm, _, _ = evaluate_detector(bim_detector, x_test, x_test_bim, device)
    results['bim_on_bim'] = acc
    print(f"  BIM Detector on BIM attack: {acc:.2f}%")
    plot_confusion_matrix(cm, ["Clean", "Adversarial"],
                          os.path.join(args.output_dir, "cm_bim_on_bim.png"),
                          "BIM Detector → BIM Attack")

    # BIM detector on PGD samples
    acc, cm, _, _ = evaluate_detector(bim_detector, x_test, x_test_pgd, device)
    results['bim_on_pgd'] = acc
    print(f"  BIM Detector on PGD attack: {acc:.2f}%")
    plot_confusion_matrix(cm, ["Clean", "Adversarial"],
                          os.path.join(args.output_dir, "cm_bim_on_pgd.png"),
                          "BIM Detector → PGD Attack")

    # Log table to WandB
    wandb.log({"detector_comparison": wandb.Table(
        columns=["Detector", "Attack", "Detection Accuracy (%)"],
        data=[
            ["PGD-trained", "PGD", f"{results['pgd_on_pgd']:.2f}"],
            ["PGD-trained", "BIM", f"{results['pgd_on_bim']:.2f}"],
            ["BIM-trained", "BIM", f"{results['bim_on_bim']:.2f}"],
            ["BIM-trained", "PGD", f"{results['bim_on_pgd']:.2f}"],
        ]
    )})

    # Print summary table
    print("\n### Detector Comparison Table")
    print("| Detector | Attack | Detection Accuracy (%) |")
    print("|----------|--------|------------------------|")
    for key, acc in results.items():
        det, atk = key.split('_on_')
        print(f"| {det.upper()}-trained | {atk.upper()} | {acc:.2f} |")

    # Save results
    with open(os.path.join(args.output_dir, "detector_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    wandb.finish()
    print(f"\n[Done] Evaluation complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
