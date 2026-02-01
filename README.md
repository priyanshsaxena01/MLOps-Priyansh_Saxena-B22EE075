# Lab 2 – CNN Analysis on CIFAR-10  
**Lab Worksheet – 31 Jan 2026**

**Name:** Priyansh Saxena  
**Roll Number:** B22EE075  

---

## 📌 Objective
To design and analyze a custom Convolutional Neural Network (CNN) pipeline on the CIFAR-10 dataset using PyTorch. The experiment focuses on:
- Building a **custom dataloader**
- Training a **CNN model (Custom ResNet-9)**
- Measuring **computational efficiency (FLOPs & parameters)**
- Visualizing **gradient flow** and **weight update flow**
- Logging all experiments and visualizations using **Weights & Biases (WandB)**

---

## 🧠 Methodology

### 1. Dataset & Custom Dataloader
- Dataset: **CIFAR-10**
- Framework: `torchvision.datasets.CIFAR10`
- Custom Dataset Class: `ProCIFAR10`

**Training Augmentations (Albumentations):**
- Horizontal Flip (p = 0.5)
- Shift, Scale & Rotate
- CoarseDropout (Cutout)
- Normalization (CIFAR-10 mean & std)

**Batch Size:** 512 (to maximize GPU throughput)

---

### 2. Model Architecture – Custom ResNet-9
A lightweight yet high-performance CNN inspired by ResNet architectures.

**Key Features:**
- Residual connections to avoid vanishing gradients
- Batch Normalization for stable training
- Dropout (0.2) for regularization
- Adaptive pooling for input-size robustness

---

### 3. Training Setup
- **Optimizer:** AdamW  
- **Scheduler:** OneCycleLR  
- **Loss Function:** CrossEntropyLoss with label smoothing (0.1)  
- **Gradient Clipping:** Max norm = 0.1  
- **Epochs:** 30  

---

## ⚙️ Computational Analysis
FLOPs and parameter count were measured using the `thop` library.

| Metric        | Value        |
|--------------|--------------|
| Parameters   | **6.57 M**   |
| FLOPs        | **380.77 M** |

**Observation:**  
The model achieves high accuracy with sub-500M FLOPs, making it suitable for efficient inference and edge deployment scenarios.

---

## 📈 Results

### Final Performance
- **Training Accuracy:** 95.10%
- **Validation Accuracy:** 93.55%
- **Training Loss:** 0.64

### Training Dynamics
- Rapid learning in early epochs (warm-up phase)
- Temporary instability during peak learning rate (OneCycleLR)
- Smooth convergence as learning rate annealed

---

## 🔍 Visualizations & Analysis

All visualizations were logged to **Weights & Biases**.

### 1. Gradient Flow
- Stable gradient norms across all layers
- No vanishing or exploding gradients observed
- Residual connections enabled effective gradient propagation

### 2. Weight Update Flow
- Early layers learned low-level features quickly
- Deeper layers showed sustained updates throughout training
- AdamW + weight decay prevented weight explosion

---

## 🧾 Key Findings
1. **Efficiency:** High accuracy achieved with a compact architecture.
2. **Generalization:** Small gap between training and validation accuracy indicates minimal overfitting.
3. **OneCycleLR:** Enabled super-convergence despite short-lived instability.
4. **Regularization:** Label smoothing and Cutout significantly improved robustness.

---

## ✅ Conclusion
This lab successfully demonstrates an end-to-end CNN experimentation pipeline using modern deep learning practices. A carefully designed ResNet-9, combined with strong data augmentation and training strategies, achieves near state-of-the-art performance on CIFAR-10 with excellent computational efficiency.

---

## 🔗 Links
- **GitHub Repository:**  
  https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Priyansh-Saxena_B22EE075_lab2_worksheet

- **WandB Project:**  
  https://wandb.ai/priyansh-saxena/cifar-10

- **Google Colab Notebook:**  
  https://colab.research.google.com/drive/16TPwZpRBKbAFi1krRktoYLRM7im4iY3o?usp=sharing

