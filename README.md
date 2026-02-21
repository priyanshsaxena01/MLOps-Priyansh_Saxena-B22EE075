# MLOps Minor Assignment: STL-10 Image Classification

This repository contains the complete Dockerized implementation for the MLOps minor assignment, focusing on fine-tuning a ResNet-18 model on a subset of the STL-10 dataset.

## 🚀 Project Overview
* **Dataset:** [Chiranjeev007/STL-10_Subset](https://huggingface.co/datasets/Chiranjeev007/STL-10_Subset)
* **Model:** Pretrained ResNet-18 (from `torchvision.models`)
* **Tracking & Visualizations:** Weights & Biases (WandB)
* **Model Registry:** Hugging Face Hub
* **Environment:** Docker

## 📊 Final Evaluation Results
The model was evaluated on the test set after downloading the best weights from the Hugging Face Hub.

**Overall Test Accuracy:** `79.90%`

### Class-wise Accuracy
| Class | Label | Accuracy |
|---|---|---|
| 0 | Airplane | 85.00% |
| 1 | Bird | 86.00% |
| 2 | Car | 82.00% |
| 3 | Cat | 83.00% |
| 4 | Deer | 82.00% |
| 5 | Dog | 46.00% |
| 6 | Horse | 64.00% |
| 7 | Monkey | 82.00% |
| 8 | Ship | 96.00% |
| 9 | Truck | 93.00% |

## 🔗 Project Links
* **WandB Dashboard (Plots & Confusion Matrix):** [priyansh-saxena/mldlops-minor](https://wandb.ai/priyansh-saxena/mldlops-minor)
* **Hugging Face Model Repo:** [b22ee075/stl10-resnet18-minor](https://huggingface.co/b22ee075/stl10-resnet18-minor)

## 🛠️ How to Run

1. **Build the Docker Image:**
```bash
   docker build -t stl10-classifier .
```

2. **Run the Container (with GPU support):**
```bash
   docker run --rm -it --gpus all \
     -e WANDB_API_KEY="your_wandb_api_key" \
     -e HF_TOKEN="your_hf_token" \
     stl10-classifier
```

## 📁 Repository Structure
* `main.py`: The core script handling data loading, training, WandB logging, HF pushing/pulling, and evaluation.
* `Dockerfile`: Container configuration for reproducible execution.
* `requirements.txt`: Python dependencies (including `filelock>=3.13.0` fix for HF datasets).
* `README.md`: Project documentation.
