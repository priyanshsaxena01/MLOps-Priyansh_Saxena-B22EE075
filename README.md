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

#### Training-Validation Table (Example Format)

Each experiment produces a table like this:

| Epoch | Training Loss | Validation Loss | Training Accuracy (%) | Validation Accuracy (%) |
|-------|---------------|-----------------|----------------------|------------------------|
| 1     | --            | --              | --                   | --                     |
| 2     | --            | --              | --                   | --                     |
| ...   | ...           | ...             | ...                  | ...                    |
| 10    | --            | --              | --                   | --                     |

> **Note**: Actual values will be populated after training. See WandB for live results.

#### Test Results Summary

| LoRA | Rank | Alpha | Dropout | Test Accuracy (%) | Trainable Parameters |
|------|------|-------|---------|-------------------|---------------------|
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

> **Note**: Replace `--` with actual results after training. Full results available on [WandB](https://wandb.ai/priyansh-saxena/Assignment-5).

#### Training Curves

*Training loss and accuracy curves will be saved to `q1_vit_lora/outputs/` and uploaded to WandB.*

#### Class-wise Test Accuracy Histogram

*Class-wise accuracy histograms for each experiment will be saved to `q1_vit_lora/outputs/` and uploaded to WandB.*

#### Gradient Update Graphs on LoRA Weights

*Gradient norm plots during training will be saved for each LoRA experiment and uploaded to WandB.*

#### Optuna Best Configuration

Best LoRA hyperparameters found by Optuna will be logged here after the search completes.

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
| 0.000   | --            | --                | --            | --             | --         |
| 0.010   | --            | --                | --            | --             | --         |
| 0.030   | --            | --                | --            | --             | --         |
| 0.050   | --            | --                | --            | --             | --         |
| 0.100   | --            | --                | --            | --             | --         |
| 0.200   | --            | --                | --            | --             | --         |
| 0.300   | --            | --                | --            | --             | --         |

> **Note**: Replace `--` with actual results after running experiments.

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
| PGD-trained (ResNet-34) | PGD | -- |
| PGD-trained (ResNet-34) | BIM | -- |
| BIM-trained (ResNet-34) | BIM | -- |
| BIM-trained (ResNet-34) | PGD | -- |

> **Target**: ≥ 70% detection accuracy for each case.

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
└── report/                            # Assignment report
    └── report.md                      # Report (converted to PDF)
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

## 📝 Report

The detailed report with all analysis, tables, graphs, and observations is available in `report/report.pdf`.

---

## 🔧 Hardware

- **GPU**: NVIDIA RTX 2050 (4 GB VRAM)
- **Mixed Precision**: FP16 used throughout for memory efficiency
- **Batch Sizes**: 32 (ViT-S), 128 (ResNet-18), 64 (ResNet-34)

---

*© 2026 Priyansh Saxena (B22EE075) — MLOps Assignment 5*
