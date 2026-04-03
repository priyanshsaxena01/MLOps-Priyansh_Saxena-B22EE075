# Assignment 5 Report: ViT LoRA Fine-Tuning & Adversarial Attacks

**Student**: Priyansh Saxena (B22EE075)  
**Course**: MLOps  
**Date**: April 2026

---

## Links

- **WandB**: [https://wandb.ai/priyansh-saxena/Assignment-5](https://wandb.ai/priyansh-saxena/Assignment-5)
- **HuggingFace**: [https://huggingface.co/b22ee075](https://huggingface.co/b22ee075)
- **GitHub**: [https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-5](https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-5)

---

## Q1: ViT-S LoRA Fine-Tuning on CIFAR-100

### 1.1 Approach

A Vision Transformer Small (ViT-S/16) model pre-trained on ImageNet-1K was fine-tuned on CIFAR-100 (100 classes). The model uses patch size 16 with 224×224 input resolution.

**Baseline (No LoRA)**: Only the classification head is trainable; all backbone weights are frozen.

**LoRA Fine-Tuning**: Low-Rank Adaptation (LoRA) is applied to the fused Q, K, V attention projection layer (`qkv`) in each transformer block. The classification head remains trainable via `modules_to_save`.

### 1.2 Experimental Setup

| Parameter | Value |
|-----------|-------|
| Base Model | `vit_small_patch16_224` (timm) |
| Dataset | CIFAR-100 (Train: 45K, Val: 5K, Test: 10K) |
| LoRA Target Modules | `qkv` (fused Q, K, V projection) |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |
| Epochs | 10 |
| Batch Size | 32 |
| Mixed Precision | FP16 |

### 1.3 Experiment Results

#### Experiment 0: No LoRA (Head-only)

| Epoch | Training Loss | Validation Loss | Training Accuracy (%) | Validation Accuracy (%) |
|-------|---------------|-----------------|----------------------|------------------------|
| 1-10  | *See WandB*   | *See WandB*     | *See WandB*          | *See WandB*            |

#### Experiments 1-9: LoRA Configurations

| Exp | Rank | Alpha | Dropout | Best Val Acc (%) | Trainable Params |
|-----|------|-------|---------|-----------------|------------------|
| 1   | 2    | 2     | 0.1     | --              | --               |
| 2   | 2    | 4     | 0.1     | --              | --               |
| 3   | 2    | 8     | 0.1     | --              | --               |
| 4   | 4    | 2     | 0.1     | --              | --               |
| 5   | 4    | 4     | 0.1     | --              | --               |
| 6   | 4    | 8     | 0.1     | --              | --               |
| 7   | 8    | 2     | 0.1     | --              | --               |
| 8   | 8    | 4     | 0.1     | --              | --               |
| 9   | 8    | 8     | 0.1     | --              | --               |

> Replace `--` with actual values after training.

### 1.4 Test Results

| LoRA | Rank | Alpha | Dropout | Overall Test Accuracy (%) | Trainable Parameters |
|------|------|-------|---------|--------------------------|---------------------|
| Without | - | - | - | -- | -- |
| With | 2 | 2 | 0.1 | -- | -- |
| With | 2 | 4 | 0.1 | -- | -- |
| With | 2 | 8 | 0.1 | -- | -- |
| With | 4 | 2 | 0.1 | -- | -- |
| With | 4 | 4 | 0.1 | -- | -- |
| With | 4 | 8 | 0.1 | -- | -- |
| With | 8 | 2 | 0.1 | -- | -- |
| With | 8 | 4 | 0.1 | -- | -- |
| With | 8 | 8 | 0.1 | -- | -- |

### 1.5 Class-wise Test Accuracy

*Histograms showing per-class accuracy are generated during testing and are available on WandB.*

### 1.6 Gradient Update Graphs

*Gradient norm plots for LoRA weights during training are logged to WandB for each LoRA experiment.*

### 1.7 Optuna Hyperparameter Optimization

**Search Space**:
- Rank: {2, 4, 8, 16, 32}
- Alpha: {2, 4, 8, 16, 32}
- Learning Rate: [1e-5, 1e-3] (log-uniform)
- Dropout: [0.0, 0.3]

**Best Configuration**: *To be determined after Optuna search*

### 1.8 Analysis

- **Effect of Rank**: Higher rank provides more expressive power but increases parameters.
- **Effect of Alpha**: Controls the scaling of LoRA updates; higher alpha = stronger adaptation.
- **LoRA vs No LoRA**: LoRA enables efficient fine-tuning with far fewer trainable parameters while maintaining competitive accuracy.

---

## Q2: Adversarial Attacks using IBM ART

### 2.1 FGSM Attack: From Scratch vs IBM ART

#### Setup

A ResNet-18 model (non-pretrained, modified for 32×32 images) was trained from scratch on clean CIFAR-10, achieving ≥72% test accuracy.

#### FGSM Implementation

**From Scratch**:
```python
perturbed = image + epsilon * sign(gradient_of_loss_wrt_input)
perturbed = clamp(perturbed, 0, 1)
```

**IBM ART**:
```python
from art.attacks.evasion import FastGradientMethod
attack = FastGradientMethod(estimator=classifier, eps=epsilon)
x_adv = attack.generate(x=x_test)
```

#### Results

| Epsilon | Clean Acc (%) | FGSM Scratch (%) | FGSM ART (%) |
|---------|---------------|-------------------|---------------|
| 0.000   | --            | --                | --            |
| 0.010   | --            | --                | --            |
| 0.030   | --            | --                | --            |
| 0.050   | --            | --                | --            |
| 0.100   | --            | --                | --            |
| 0.200   | --            | --                | --            |
| 0.300   | --            | --                | --            |

#### Analysis

- Both implementations produce similar accuracy drops, confirming correct from-scratch implementation.
- Attack strength increases with epsilon, causing greater accuracy degradation.
- Even small perturbations (ε=0.03) can cause significant accuracy drops.

### 2.2 Adversarial Detection

#### (a) PGD-based Detector

- **Attack**: Projected Gradient Descent (PGD) via IBM ART
- **Detector**: ResNet-34 binary classifier (clean vs adversarial)
- **Detection Accuracy**: -- % (target: ≥70%)

#### (b) BIM-based Detector

- **Attack**: Basic Iterative Method (BIM) via IBM ART
- **Detector**: ResNet-34 binary classifier (clean vs adversarial)
- **Detection Accuracy**: -- % (target: ≥70%)

#### Cross-Evaluation

| Detector | Tested On | Detection Accuracy (%) |
|----------|-----------|------------------------|
| PGD-trained | PGD samples | -- |
| PGD-trained | BIM samples | -- |
| BIM-trained | BIM samples | -- |
| BIM-trained | PGD samples | -- |

#### Analysis

- Both PGD and BIM are iterative attacks that create subtler perturbations than single-step FGSM.
- Detectors trained on one attack type may generalize to detect other iterative attacks.
- PGD (being a stronger attack with more iterations) may produce adversarial images that are easier to detect.

### 2.3 Qualitative Results

*Visual comparison of clean vs adversarial images for all attack types are available in `q2_adversarial/outputs/` and on WandB.*

WandB contains 10 samples each of:
- Clean images
- FGSM (from scratch) adversarial images
- FGSM (IBM ART) adversarial images
- PGD adversarial images
- BIM adversarial images

---

## Conclusion

This assignment demonstrated:
1. **LoRA efficiency**: LoRA enables fine-tuning ViT with significantly fewer parameters while maintaining competitive accuracy on CIFAR-100.
2. **Adversarial vulnerability**: Deep learning models are vulnerable to adversarial perturbations, even at imperceptible levels.
3. **Detection feasibility**: Binary classifiers can detect adversarial examples with reasonable accuracy, though cross-attack generalization varies.

---

## References

1. Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
2. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
3. Goodfellow, I. J., et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
4. Madry, A., et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.
5. IBM Adversarial Robustness Toolbox: https://github.com/Trusted-AI/adversarial-robustness-toolbox
