"""
ViT-S Model creation with optional LoRA injection via PEFT.
"""

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType


def create_vit_model(num_classes=100, pretrained=True):
    """
    Create a ViT-Small model pre-trained on ImageNet.

    Args:
        num_classes: Number of output classes (100 for CIFAR-100)
        pretrained: Whether to load ImageNet pre-trained weights

    Returns:
        model: ViT-S model with modified classification head
    """
    model = timm.create_model(
        'vit_small_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    print(f"[Model] ViT-S created with {num_classes} classes")
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def freeze_base_model(model):
    """
    Freeze all parameters except the classification head.
    Used for the no-LoRA baseline.

    Args:
        model: ViT model

    Returns:
        model: Model with frozen backbone, trainable head
    """
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Model] Frozen backbone. Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    return model


def apply_lora(model, rank=8, alpha=8, dropout=0.1):
    """
    Apply LoRA to the ViT model's Q, K, V attention weights using PEFT.

    In timm's ViT, Q/K/V are fused into a single 'qkv' linear layer.
    We target that layer and also keep the classification 'head' trainable.

    Args:
        model: ViT model
        rank: LoRA rank (r)
        alpha: LoRA scaling factor (alpha)
        dropout: LoRA dropout rate

    Returns:
        peft_model: Model with LoRA applied
    """
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["qkv"],          # Fused Q,K,V in timm ViT
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["head"],        # Keep classifier head trainable
    )

    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameter summary
    peft_model.print_trainable_parameters()

    return peft_model


def get_trainable_params(model):
    """
    Get count of trainable and total parameters.

    Returns:
        trainable_params, total_params, percentage
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    return trainable, total, percentage


def get_lora_params(model):
    """
    Get only LoRA parameters (for gradient tracking).

    Returns:
        dict of {name: parameter} for LoRA layers
    """
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_params[name] = param
    return lora_params


def print_model_summary(model):
    """Print a summary of model architecture and parameters."""
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
            params = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            status = "✓ trainable" if trainable > 0 else "✗ frozen"
            print(f"  {name}: {params:,} params [{status}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("--- No LoRA (baseline) ---")
    model = create_vit_model(num_classes=100)
    model = freeze_base_model(model)
    t, tot, pct = get_trainable_params(model)
    print(f"Trainable: {t:,}, Total: {tot:,}, Pct: {pct:.2f}%\n")

    print("--- With LoRA (rank=4, alpha=8) ---")
    model2 = create_vit_model(num_classes=100)
    model2 = apply_lora(model2, rank=4, alpha=8, dropout=0.1)
    t, tot, pct = get_trainable_params(model2)
    print(f"Trainable: {t:,}, Total: {tot:,}, Pct: {pct:.2f}%")

    lora_p = get_lora_params(model2)
    print(f"LoRA parameter groups: {len(lora_p)}")
    for name in list(lora_p.keys())[:5]:
        print(f"  {name}: {lora_p[name].shape}")
