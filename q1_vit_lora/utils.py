"""
Utility functions for Q1: plotting, logging, WandB integration, HuggingFace upload.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import torch
from huggingface_hub import HfApi, create_repo


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


def log_epoch_metrics(epoch, train_loss, val_loss, train_acc, val_acc,
                      lora_grad_norms=None):
    """Log training metrics to WandB."""
    metrics = {
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/accuracy": train_acc,
        "val/accuracy": val_acc,
    }

    if lora_grad_norms is not None:
        for name, norm in lora_grad_norms.items():
            short_name = name.replace("base_model.model.", "").replace(".", "/")
            metrics[f"gradients/{short_name}"] = norm

    wandb.log(metrics, step=epoch)


def compute_gradient_norms(model):
    """
    Compute gradient norms for LoRA parameters.

    Args:
        model: PEFT model with LoRA layers

    Returns:
        dict of {param_name: gradient_norm}
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad and param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms


def plot_training_curves(history, save_path, experiment_name=""):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the figure
        experiment_name: Name for the title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'Loss - {experiment_name}', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Accuracy - {experiment_name}', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Also log to wandb
    wandb.log({f"plots/training_curves": wandb.Image(save_path)})
    print(f"[Plot] Training curves saved to {save_path}")


def plot_classwise_histogram(class_accuracies, class_names, save_path,
                              experiment_name=""):
    """
    Plot class-wise test accuracy histogram.

    Args:
        class_accuracies: dict or list of per-class accuracies
        class_names: list of class names
        save_path: Path to save figure
        experiment_name: Name for the title
    """
    if isinstance(class_accuracies, dict):
        accs = [class_accuracies.get(i, 0.0) for i in range(len(class_names))]
    else:
        accs = class_accuracies

    fig, ax = plt.subplots(figsize=(20, 8))
    bars = ax.bar(range(len(accs)), accs, color=sns.color_palette("viridis", len(accs)))

    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Class-wise Test Accuracy - {experiment_name}', fontsize=14)
    ax.set_xticks(range(0, len(accs), 5))
    ax.set_xticklabels(range(0, len(accs), 5), rotation=45)
    ax.axhline(y=np.mean(accs), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(accs):.1f}%')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"plots/classwise_accuracy": wandb.Image(save_path)})
    print(f"[Plot] Class-wise histogram saved to {save_path}")


def plot_gradient_norms(all_grad_norms, save_path, experiment_name=""):
    """
    Plot gradient norms of LoRA weights across training epochs.

    Args:
        all_grad_norms: list of dicts, one per epoch
        save_path: Path to save figure
        experiment_name: Name for the title
    """
    if not all_grad_norms or len(all_grad_norms) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    epochs = range(1, len(all_grad_norms) + 1)

    # Get unique layer names
    layer_names = list(all_grad_norms[0].keys())

    # Simplify names for legend
    for name in layer_names[:10]:  # Limit to 10 layers for readability
        short = name.split(".")[-3] + "." + name.split(".")[-1]
        values = [gn.get(name, 0.0) for gn in all_grad_norms]
        ax.plot(epochs, values, '-o', label=short, linewidth=1.5, markersize=4)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(f'LoRA Gradient Norms - {experiment_name}', fontsize=14)
    ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"plots/gradient_norms": wandb.Image(save_path)})
    print(f"[Plot] Gradient norms saved to {save_path}")


def generate_results_table(results, save_path=None):
    """
    Generate a formatted results table as markdown.

    Args:
        results: list of dicts with keys:
            lora, rank, alpha, dropout, test_acc, trainable_params
        save_path: Optional path to save as text file

    Returns:
        table_str: Formatted markdown table string
    """
    header = "| LoRA | Rank | Alpha | Dropout | Test Accuracy (%) | Trainable Params |"
    separator = "|------|------|-------|---------|-------------------|------------------|"
    rows = [header, separator]

    for r in results:
        lora_str = "With" if r['lora'] else "Without"
        rank_str = str(r.get('rank', '-'))
        alpha_str = str(r.get('alpha', '-'))
        dropout_str = str(r.get('dropout', '-'))
        acc_str = f"{r['test_acc']:.2f}"
        params_str = f"{r['trainable_params']:,}"
        rows.append(f"| {lora_str} | {rank_str} | {alpha_str} | {dropout_str} | "
                     f"{acc_str} | {params_str} |")

    table_str = "\n".join(rows)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"[Table] Results saved to {save_path}")

    return table_str


def generate_epoch_table(history, experiment_name=""):
    """
    Generate per-epoch training table as per assignment spec.

    Args:
        history: dict with train_loss, val_loss, train_acc, val_acc lists

    Returns:
        table_str: Formatted markdown table
    """
    header = ("| Epoch | Training Loss | Validation Loss | "
              "Training Accuracy (%) | Validation Accuracy (%) |")
    separator = ("|-------|---------------|-----------------|"
                 "----------------------|-------------------------|")
    rows = [f"**{experiment_name}**\n", header, separator]

    for i in range(len(history['train_loss'])):
        rows.append(
            f"| {i + 1} | {history['train_loss'][i]:.4f} | "
            f"{history['val_loss'][i]:.4f} | "
            f"{history['train_acc'][i]:.2f} | "
            f"{history['val_acc'][i]:.2f} |"
        )

    return "\n".join(rows)


def push_to_huggingface(model_path, repo_id, token=None, commit_message="Upload best model"):
    """
    Push model weights to HuggingFace Hub.

    Args:
        model_path: Path to the model weights file (.pth)
        repo_id: HuggingFace repo ID (e.g., 'b22ee075/vit-s-lora-cifar100')
        token: HuggingFace API token
        commit_message: Commit message
    """
    api = HfApi()

    if token is None:
        token = os.environ.get("HF_TOKEN")

    try:
        create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"[HF] Repo creation note: {e}")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
    )
    print(f"[HF] Uploaded {model_path} to {repo_id}")


def save_results_json(results, save_path):
    """Save results as JSON for later use in README generation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved to {save_path}")


def load_results_json(load_path):
    """Load results from JSON."""
    with open(load_path, 'r') as f:
        return json.load(f)
