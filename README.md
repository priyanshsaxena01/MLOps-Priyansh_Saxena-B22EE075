# Assignment 5: ViT LoRA Fine-Tuning & Adversarial Attacks

**Course**: ML-Dl-Ops  
**Student**: Priyansh Saxena (B22EE075)  
**Branch**: Assignment-5

---

## 📋 Table of Contents

- [Overview](#overview)
- [Links](#links)
- [Setup & Installation](#setup--installation)
- [Q1: ViT-S LoRA Fine-Tuning on CIFAR-100](#q1-vit-s-lora-fine-tuning-on-cifar-100)
- [Q2: Adversarial Attacks using IBM ART](#q2-adversarial-attacks-using-ibm-art)
- [Results](#results)
- [Project Structure](#project-structure)

---

## 🔗 Links

| Resource | Link |
|----------|------|
| **WandB Project** | [https://wandb.ai/priyansh-saxena/Assignment-5](https://wandb.ai/priyansh-saxena/Assignment-5) |
| **HuggingFace Model** | [https://huggingface.co/b22ee075/vit-s-lora-cifar100](https://huggingface.co/b22ee075/vit-s-lora-cifar100) |
| **GitHub Repository** | [https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-5](https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-5) |

---

## 🛠️ Setup & Installation

### Prerequisites

- Docker Desktop with WSL2 backend (GPU support enabled)
- NVIDIA GPU drivers (CUDA 12.x compatible)
- NVIDIA Container Toolkit

### Method 1: Using Docker (Recommended - Required per assignment)

```bash
# Clone the repository
git clone -b Assignment-5 https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075.git
cd MLOps-Priyansh_Saxena-B22EE075

# Set environment variables (create .env file)
echo "WANDB_API_KEY=your_wandb_key_here" > .env
echo "HF_TOKEN=your_hf_token_here" >> .env

# Build Docker image
docker build -t assignment5 .

# Run container with GPU
docker run --gpus all -it --rm \
    --shm-size=4g \
    -v $(pwd):/app \
    --env-file .env \
    assignment5

# OR using docker-compose
docker-compose up -d
docker-compose exec assignment5 bash
```

### Method 2: Local Installation (for reference)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.1.0 | Deep learning framework |
| `torchvision` | >=0.16.0 | Vision models and datasets |
| `timm` | >=0.9.12 | ViT-S pretrained model |
| `peft` | >=0.7.0 | LoRA/PEFT implementation |
| `transformers` | >=4.36.0 | HuggingFace integration |
| `optuna` | latest | Hyperparameter optimization |
| `wandb` | latest | Experiment tracking |
| `adversarial-robustness-toolbox` | latest | IBM ART for adversarial attacks |
| `matplotlib` | latest | Plotting |
| `seaborn` | latest | Visualization |
| `scikit-learn` | latest | Metrics |

---

## 📌 Q1: ViT-S LoRA Fine-Tuning on CIFAR-100

### Overview

Fine-tune a Vision Transformer Small (ViT-S/16) pretrained on ImageNet for CIFAR-100 classification:
1. **Baseline**: Fine-tune only the classification head (no LoRA)
2. **LoRA Experiments**: Apply LoRA to Q, K, V attention weights with various hyperparameters
3. **Optuna HPO**: Find the best LoRA configuration
4. **Push to HuggingFace**: Upload best model weights

### Commands to Run

```bash
# ===== Inside Docker container =====

# Login to WandB (if not using env variable)
wandb login

# --- Step 1: Run ALL experiments (baseline + 9 LoRA combos) ---
cd /app
python q1_vit_lora/train.py --run_all --epochs 10 --batch_size 32

# --- Step 2: Test all models ---
python q1_vit_lora/test.py --weights_dir ./q1_vit_lora/weights --output_dir ./q1_vit_lora/outputs

# --- Step 3: Optuna HPO ---
python q1_vit_lora/optuna_search.py --n_trials 20 --epochs_per_trial 5 --epochs_final 10

# --- Run a single experiment (optional) ---
# Without LoRA
python q1_vit_lora/train.py --epochs 10 --batch_size 32

# With LoRA
python q1_vit_lora/train.py --use_lora --rank 4 --alpha 8 --dropout 0.1 --epochs 10 --batch_size 32
```

### Experiment Configuration

| Parameter | Values |
|-----------|--------|
| Model | ViT-S/16 (vit_small_patch16_224) |
| Dataset | CIFAR-100 |
| LoRA Target | Q, K, V attention weights (`qkv` fused layer) |
| Ranks | 2, 4, 8 |
| Alpha | 2, 4, 8 |
| Dropout | 0.1 |
| Epochs | 10 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Batch Size | 32 |

### Q1 Results

#### Training-Validation Table (Example Format: Best Model Exp8_LoRA_r8_a4_d0.1)

Each experiment produces a table like this:

| Epoch | Training Loss | Validation Loss | Training Accuracy (%) | Validation Accuracy (%) |
|-------|---------------|-----------------|----------------------|------------------------|
| 1     | 1.4623        | 0.5643          | 67.97                | 84.38                  |
| 2     | 0.4394        | 0.4461          | 86.91                | 86.62                  |
| 3     | 0.3610        | 0.4015          | 88.97                | 87.84                  |
| 4     | 0.3178        | 0.3766          | 90.12                | 88.28                  |
| 5     | 0.2864        | 0.3709          | 91.02                | 88.36                  |
| 6     | 0.2672        | 0.3603          | 91.63                | 88.70                  |
| 7     | 0.2576        | 0.3561          | 91.85                | 88.96                  |
| 8     | 0.2446        | 0.3535          | 92.31                | 89.16                  |
| 9     | 0.2368        | 0.3532          | 92.56                | 89.12                  |
| 10    | 0.2346        | 0.3518          | 92.58                | 89.18                  |

#### Test Results Summary

| LoRA | Rank | Alpha | Dropout | Test Accuracy (%) | Trainable Parameters |
|------|------|-------|---------|-------------------|---------------------|
| Without | - | - | 0.1 | 77.22 | 38,500 |
| With | 2 | 2 | 0.1 | 89.22 | 75,364 |
| With | 2 | 4 | 0.1 | 89.27 | 75,364 |
| With | 2 | 8 | 0.1 | 89.51 | 75,364 |
| With | 4 | 2 | 0.1 | 89.20 | 112,228 |
| With | 4 | 4 | 0.1 | 89.43 | 112,228 |
| With | 4 | 8 | 0.1 | 89.13 | 112,228 |
| With | 8 | 2 | 0.1 | 89.17 | 185,956 |
| With | 8 | 4 | 0.1 | 89.65 | 185,956 |
| With | 8 | 8 | 0.1 | 89.57 | 185,956 |

> **Note**: Full results available on [WandB](https://wandb.ai/priyansh-saxena/Assignment-5). Best single experiment test accuracy: **89.65%** (Rank 8, Alpha 4).

#### Training Curves

*Training loss and accuracy curves will be saved to `q1_vit_lora/outputs/` and uploaded to WandB.*

#### Class-wise Test Accuracy Histogram

*Class-wise accuracy histograms for each experiment will be saved to `q1_vit_lora/outputs/` and uploaded to WandB.*

#### Gradient Update Graphs on LoRA Weights

*Gradient norm plots during training will be saved for each LoRA experiment and uploaded to WandB.*

#### Optuna Best Configuration

**Best LoRA Configuration found:** Rank (`r`) = 4, Alpha (`a`) = 8, yielding an Optuna Best Test Accuracy of **89.51%**. Model pushed to HuggingFace hub as `Optuna_Best_r4_a8`.

---

## 📌 Q2: Adversarial Attacks using IBM ART

### Q2(i): FGSM Attack - From Scratch vs IBM ART

#### Commands to Run

```bash
# ===== Inside Docker container =====

# --- Step 1: Train ResNet-18 on clean CIFAR-10 ---
python q2_adversarial/train_resnet18.py --epochs 50 --batch_size 128

# --- Step 2: FGSM from scratch ---
python q2_adversarial/fgsm_scratch.py --model_path ./q2_adversarial/weights/resnet18_cifar10_best.pth

# --- Step 3: FGSM using IBM ART ---
python q2_adversarial/fgsm_art.py --model_path ./q2_adversarial/weights/resnet18_cifar10_best.pth

# --- Step 4: Compare results ---
python q2_adversarial/compare_fgsm.py --model_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
```

#### FGSM Comparison Results

| Epsilon | Clean Acc (%) | FGSM Scratch (%) | FGSM ART (%) | Drop (Scratch) | Drop (ART) |
|---------|---------------|-------------------|---------------|----------------|------------|
| 0.000   | 99.37         | 99.37             | 99.37         | 0.00           | 0.00       |
| 0.010   | 99.37         | 35.58             | 38.83         | 63.79          | 60.54      |
| 0.030   | 99.37         | 23.27             | 25.05         | 76.10          | 74.32      |
| 0.050   | 99.37         | 18.28             | 19.78         | 81.09          | 79.59      |
| 0.100   | 99.37         | 12.29             | 12.53         | 87.08          | 86.84      |
| 0.200   | 99.37         | 10.04             | 10.36         | 89.33          | 89.01      |
| 0.300   | 99.37         | 10.00             | 10.32         | 89.37          | 89.05      |

#### Visual Comparison

*Side-by-side comparison images (Original vs FGSM Scratch vs FGSM ART) will be saved to `q2_adversarial/outputs/` and uploaded to WandB.*

### Q2(ii): Adversarial Detection Model

#### Commands to Run

```bash
# ===== Inside Docker container =====

# --- Step 1: Train PGD adversarial detector (ResNet-34) ---
python q2_adversarial/train_detector_pgd.py --resnet18_path ./q2_adversarial/weights/resnet18_cifar10_best.pth --epochs 30

# --- Step 2: Train BIM adversarial detector (ResNet-34) ---
python q2_adversarial/train_detector_bim.py --resnet18_path ./q2_adversarial/weights/resnet18_cifar10_best.pth --epochs 30

# --- Step 3: Evaluate both detectors (cross-evaluation) ---
python q2_adversarial/evaluate_detector.py

# --- Step 4: Upload attack samples to WandB ---
python q2_adversarial/wandb_samples.py --resnet18_path ./q2_adversarial/weights/resnet18_cifar10_best.pth
```

#### Detector Comparison Results

| Detector | Attack | Detection Accuracy (%) |
|----------|--------|------------------------|
| PGD-trained (ResNet-34) | PGD | 99.95 |
| PGD-trained (ResNet-34) | BIM | 99.28 |
| BIM-trained (ResNet-34) | BIM | 99.95 |
| BIM-trained (ResNet-34) | PGD | 99.97 |

> **Target Result**: Achieved near-perfect detection accuracy (~99%), well above the ≥ 70% requirement for each target case!

#### Adversarial Samples on WandB

10 samples of clean and adversarial images for each attack type (FGSM scratch, FGSM ART, PGD, BIM) are uploaded to [WandB](https://wandb.ai/priyansh-saxena/Assignment-5).

---

## 📁 Project Structure

```
Assignment 5/
├── Dockerfile                         # Docker setup with CUDA 12.1
├── docker-compose.yml                 # Docker Compose with GPU support
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
│
├── q1_vit_lora/                       # Q1: ViT LoRA Fine-tuning
│   ├── dataset.py                     # CIFAR-100 data loading
│   ├── model.py                       # ViT-S model + LoRA injection
│   ├── train.py                       # Training (all experiments)
│   ├── test.py                        # Testing & evaluation
│   ├── optuna_search.py               # Optuna HPO
│   ├── utils.py                       # Plotting & logging utilities
│   ├── weights/                       # Model weights
│   └── outputs/                       # Plots, tables, results
│
├── q2_adversarial/                    # Q2: Adversarial Attacks
│   ├── dataset.py                     # CIFAR-10 data loading
│   ├── model.py                       # ResNet-18 & ResNet-34
│   ├── train_resnet18.py              # Train classifier on clean data
│   ├── fgsm_scratch.py                # FGSM from scratch
│   ├── fgsm_art.py                    # FGSM using IBM ART
│   ├── compare_fgsm.py                # Compare & visualize FGSM results
│   ├── train_detector_pgd.py          # PGD adversarial detector
│   ├── train_detector_bim.py          # BIM adversarial detector
│   ├── evaluate_detector.py           # Cross-evaluate detectors
│   ├── wandb_samples.py               # Upload samples to WandB
│   ├── utils.py                       # Plotting utilities
│   ├── weights/                       # Model weights
│   └── outputs/                       # Plots, results
│
└── report                             # Assignment report

```

---

## 📊 Complete Run Order

```bash
# Build and enter Docker container
docker build -t assignment5 .
docker run --gpus all -it --rm --shm-size=4g -v $(pwd):/app --env-file .env assignment5

# Login to services
wandb login
huggingface-cli login

# ========== Q1 ==========
python q1_vit_lora/train.py --run_all --epochs 10 --batch_size 32
python q1_vit_lora/test.py
python q1_vit_lora/optuna_search.py --n_trials 20

# ========== Q2 ==========
python q2_adversarial/train_resnet18.py --epochs 50
python q2_adversarial/fgsm_scratch.py
python q2_adversarial/fgsm_art.py
python q2_adversarial/compare_fgsm.py
python q2_adversarial/train_detector_pgd.py --epochs 30
python q2_adversarial/train_detector_bim.py --epochs 30
python q2_adversarial/evaluate_detector.py
python q2_adversarial/wandb_samples.py
```


---

## 🔧 Hardware

- **GPU**: NVIDIA RTX 2050 (4 GB VRAM)
- **Mixed Precision**: FP16 used throughout for memory efficiency
- **Batch Sizes**: 32 (ViT-S), 128 (ResNet-18), 64 (ResNet-34)

---

*© 2026 Priyansh Saxena (B22EE075) — ML-Dl-Ops Assignment 5*
