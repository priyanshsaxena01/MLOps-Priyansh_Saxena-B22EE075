# MLOps-Priyansh_Saxena-B22EE075

## DLOPs Assignment 1
**Deadline:** 24/01/2026  
**Student:** Priyansh Saxena - B22EE075

---

## Project Overview
This repository contains the experiments conducted for Assignment 1. The objective was to train Deep Learning models (ResNet-18, ResNet-50) and Machine Learning classifiers (SVM) on the MNIST and FashionMNIST datasets. The project analyzes the impact of batch sizes, optimizers, learning rates, and hardware acceleration (CPU vs. GPU).

**Colab Notebook:** 

---

## Q1(a). Deep Learning Model Training
**Task:** Train ResNet-18 and ResNet-50 (scratch) on MNIST and FashionMNIST with a 70-10-20 split.

### Q1(a). Experimental Results

#### MNIST Dataset

| Batch Size | Optimizers | Learning Rate | ResNet-18 Accuracy | ResNet-50 Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | 99.03% | 98.73% |
| 16 | SGD | 0.0001 | 97.50% | 96.41% |
| 16 | Adam | 0.001 | 98.71% | 97.94% |
| 16 | Adam | 0.0001 | 98.96% | 98.86% |
| 32 | SGD | 0.001 | 98.76% | 98.70% |
| 32 | Adam | 0.0001 | 98.79% | 97.55% |

#### FashionMNIST Dataset

| Batch Size | Optimizers | Learning Rate | ResNet-18 Accuracy | ResNet-50 Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 16 | SGD | 0.001 | 90.96% | 90.44% |
| 16 | SGD | 0.0001 | 86.64% | 80.25% |
| 16 | Adam | 0.001 | 91.78% | 89.21% |
| 16 | Adam | 0.0001 | 91.81% | 89.60% |
| 32 | SGD | 0.001 | 90.46% | 88.33% |
| 32 | Adam | 0.0001 | 91.24% | 88.78% |

### Analysis Summary
*   **ResNet-18 vs ResNet-50:** ResNet-18 consistently outperformed ResNet-50 on these datasets. The deeper architecture of ResNet-50 is likely overkill for 28x28 grayscale images and harder to train without pre-trained weights.
*   **Optimizers:** Adam generally converged faster than SGD, particularly at lower learning rates.

---

## Q1(b). SVM Classification
**Task:** Train SVM classifiers with `poly` and `rbf` kernels.

| Dataset | Kernel | Test Accuracy (%) | Training Time (ms) |
| :--- | :--- | :--- | :--- |
| MNIST | Poly | 94.60% | 5,547 |
| MNIST | RBF | **95.95%** | 5,591 |
| FashionMNIST | Poly | 81.90% | 5,285 |
| FashionMNIST | RBF | **86.15%** | 5,129 |

### Analysis Summary
*   The **RBF kernel** performed better than the Polynomial kernel on both datasets, handling the non-linear complexities of image data more effectively.
*   SVM training time remains significantly lower than Deep Learning for small subsets, but accuracy is lower compared to ResNet (86% vs 91%).

---

## Q2. CPU vs GPU Analysis (FashionMNIST)
**Task:** Compare performance, training time, and FLOPs on different hardware.

| Compute | Batch Size | Optimizer | LR | ResNet-18 Acc | ResNet-18 Time (ms) | ResNet-18 FLOPs | ResNet-50 Acc | ResNet-50 Time (ms) | ResNet-50 FLOPs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | 16 | SGD | 0.001 | 87.79% | 876,753 | 5e8 | 86.04% | 2,640,757 | 1e9 |
| **CPU** | 16 | Adam | 0.001 | 90.24% | 938,100 | 5e8 | 88.96% | 2,781,353 | 1e9 |
| **GPU** | 16 | SGD | 0.001 | 91.15% | **124,276** | 5e8 | 90.39% | 271,323 | 1e9 |
| **GPU** | 16 | Adam | 0.001 | 92.41% | 138,868 | 5e8 | 90.92% | 299,430 | 1e9 |

*(Note: FLOPs are approximated based on a single forward pass of a 1x28x28 input)*

### Analysis Summary
1.  **Speedup:** GPU training was approximately **7x faster** for ResNet-18 and **10x faster** for ResNet-50 compared to CPU.
2.  **Complexity:** ResNet-50 requires roughly double the FLOPs of ResNet-18, resulting in doubled training time on GPU, without providing a benefit in accuracy for this specific resolution.

---

## Usage
To reproduce these results, clone the repository and run the provided Jupyter Notebook or Python script.

```bash
pip install torch torchvision pandas scikit-learn thop
python assignment1_script.py
