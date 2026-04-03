"""
Q2: Upload 10 sample images to WandB for each attack type.
Clean, FGSM (scratch), FGSM (ART), PGD, BIM.

Usage:
    python wandb_samples.py --resnet18_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import wandb

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (FastGradientMethod, ProjectedGradientDescent,
                                  BasicIterativeMethod)

from dataset import get_cifar10_numpy
from model import create_resnet18
from utils import setup_wandb, log_attack_samples_to_wandb


def fgsm_attack_np(model, images, labels, epsilon, device):
    """FGSM from scratch on a numpy batch."""
    model.eval()
    images_t = torch.from_numpy(images).float().to(device).requires_grad_(True)
    labels_t = torch.from_numpy(labels).long().to(device)

    outputs = model(images_t)
    loss = nn.CrossEntropyLoss()(outputs, labels_t)
    model.zero_grad()
    loss.backward()

    perturbed = images_t + epsilon * images_t.grad.data.sign()
    perturbed = torch.clamp(perturbed, 0.0, 1.0)

    with torch.no_grad():
        preds = model(perturbed).argmax(1).cpu().numpy()

    return perturbed.detach().cpu().numpy(), preds


def parse_args():
    parser = argparse.ArgumentParser(description="Log attack samples to WandB")
    parser.add_argument("--resnet18_path", type=str,
                        default="./q2_adversarial/weights/resnet18_cifar10_best.pth")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = create_resnet18(num_classes=10)
    ckpt = torch.load(args.resnet18_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # ART wrapper
    criterion = nn.CrossEntropyLoss()
    art_classifier = PyTorchClassifier(
        model=model, loss=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(0.0, 1.0),
        device_type="gpu" if torch.cuda.is_available() else "cpu",
    )

    # Load test data
    _, _, x_test, y_test = get_cifar10_numpy(data_dir=args.data_dir)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    n = args.num_samples
    x_sample = x_test[:n]
    y_sample = y_test[:n]

    # WandB
    run = setup_wandb(
        project_name="Assignment-5",
        run_name="Q2_Attack_Samples",
        config=vars(args)
    )

    # Clean predictions
    clean_preds = np.argmax(art_classifier.predict(x_sample, batch_size=n), axis=1)

    # FGSM from scratch
    x_fgsm_scratch, preds_fgsm_scratch = fgsm_attack_np(
        model, x_sample, y_sample, args.epsilon, device
    )

    # FGSM via ART
    fgsm_art = FastGradientMethod(estimator=art_classifier, eps=args.epsilon)
    x_fgsm_art = fgsm_art.generate(x=x_sample)
    preds_fgsm_art = np.argmax(art_classifier.predict(x_fgsm_art), axis=1)

    # PGD via ART
    pgd = ProjectedGradientDescent(
        estimator=art_classifier, eps=0.03, eps_step=0.007,
        max_iter=40, batch_size=n
    )
    x_pgd = pgd.generate(x=x_sample)
    preds_pgd = np.argmax(art_classifier.predict(x_pgd), axis=1)

    # BIM via ART
    bim = BasicIterativeMethod(
        estimator=art_classifier, eps=0.03, eps_step=0.007,
        max_iter=10, batch_size=n
    )
    x_bim = bim.generate(x=x_sample)
    preds_bim = np.argmax(art_classifier.predict(x_bim), axis=1)

    # Log all to WandB
    log_attack_samples_to_wandb({
        "01_Clean": (x_sample, y_sample, clean_preds),
        "02_FGSM_Scratch": (x_fgsm_scratch, y_sample, preds_fgsm_scratch),
        "03_FGSM_ART": (x_fgsm_art, y_sample, preds_fgsm_art),
        "04_PGD": (x_pgd, y_sample, preds_pgd),
        "05_BIM": (x_bim, y_sample, preds_bim),
    }, class_names, num_samples=n)

    print(f"\n[Done] Logged {n} samples per attack type to WandB")

    wandb.finish()


if __name__ == "__main__":
    main()
