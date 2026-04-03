"""
Utility functions for Q2: plotting, WandB helpers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def setup_wandb(project_name="Assignment-5", run_name=None, config=None,
                entity="priyansh-saxena"):
    """Initialize WandB run."""
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        entity=entity,
        reinit=True
    )
    return run


def plot_adversarial_comparison(original, adv_scratch, adv_art, labels, pred_orig,
                                 pred_scratch, pred_art, class_names, save_path,
                                 num_samples=5):
    """
    Plot side-by-side comparison of original vs adversarial images.

    Args:
        original: Original images (N, 3, 32, 32) in [0, 1]
        adv_scratch: FGSM from scratch adversarial images
        adv_art: FGSM via ART adversarial images
        labels: True labels
        pred_orig: Predictions on original
        pred_scratch: Predictions on scratch adversarial
        pred_art: Predictions on ART adversarial
        class_names: List of class names
        save_path: Path to save figure
        num_samples: Number of samples to show
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))

    for i in range(num_samples):
        # Original
        img_orig = np.transpose(original[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0, 1)
        axes[i, 0].imshow(img_orig)
        true_label = class_names[labels[i]]
        pred_label = class_names[pred_orig[i]]
        color = 'green' if pred_orig[i] == labels[i] else 'red'
        axes[i, 0].set_title(f"Original\nTrue: {true_label}\nPred: {pred_label}",
                              fontsize=10, color=color)
        axes[i, 0].axis('off')

        # FGSM from scratch
        img_scratch = np.transpose(adv_scratch[i], (1, 2, 0))
        img_scratch = np.clip(img_scratch, 0, 1)
        axes[i, 1].imshow(img_scratch)
        pred_label = class_names[pred_scratch[i]]
        color = 'green' if pred_scratch[i] == labels[i] else 'red'
        axes[i, 1].set_title(f"FGSM (Scratch)\nPred: {pred_label}",
                              fontsize=10, color=color)
        axes[i, 1].axis('off')

        # FGSM via ART
        img_art = np.transpose(adv_art[i], (1, 2, 0))
        img_art = np.clip(img_art, 0, 1)
        axes[i, 2].imshow(img_art)
        pred_label = class_names[pred_art[i]]
        color = 'green' if pred_art[i] == labels[i] else 'red'
        axes[i, 2].set_title(f"FGSM (IBM ART)\nPred: {pred_label}",
                              fontsize=10, color=color)
        axes[i, 2].axis('off')

    plt.suptitle("Adversarial Image Comparison: Original vs FGSM (Scratch) vs FGSM (ART)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Adversarial comparison saved to {save_path}")


def plot_eps_vs_accuracy(epsilons, accs_clean, accs_scratch, accs_art, save_path):
    """
    Plot perturbation strength (epsilon) vs accuracy.

    Args:
        epsilons: List of epsilon values
        accs_clean: Clean accuracy (same for all eps)
        accs_scratch: FGSM scratch accuracies per epsilon
        accs_art: FGSM ART accuracies per epsilon
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epsilons, accs_scratch, 'r-o', label='FGSM (Scratch)', linewidth=2, markersize=8)
    ax.plot(epsilons, accs_art, 'b-s', label='FGSM (IBM ART)', linewidth=2, markersize=8)
    ax.axhline(y=accs_clean, color='green', linestyle='--', linewidth=2,
               label=f'Clean Accuracy ({accs_clean:.1f}%)')

    ax.set_xlabel('Perturbation Strength (ε)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('FGSM Attack: Perturbation Strength vs Accuracy', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Epsilon vs accuracy saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path, title=""):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Loss - {title}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy - {title}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"plots/{title}": wandb.Image(save_path)})


def log_attack_samples_to_wandb(images_dict, class_names, num_samples=10):
    """
    Log sample attack images to WandB.

    Args:
        images_dict: dict of {attack_name: (images, labels, preds)}
                     images shape: (N, 3, 32, 32) in [0, 1]
        class_names: List of class names
        num_samples: Number of samples to log
    """
    for attack_name, (images, labels, preds) in images_dict.items():
        wandb_images = []
        for i in range(min(num_samples, len(images))):
            img = np.transpose(images[i], (1, 2, 0))
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            true_label = class_names[labels[i]]
            pred_label = class_names[preds[i]]
            caption = f"True: {true_label}, Pred: {pred_label}"
            wandb_images.append(wandb.Image(img, caption=caption))

        wandb.log({f"samples/{attack_name}": wandb_images})
        print(f"[WandB] Logged {len(wandb_images)} {attack_name} samples")
