# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

**Submitted by:** Priyansh Saxena
**Roll Number:** B22EE075
**Date:** February 18, 2026

---

## 1. Submission Links

- **GitHub Repository:** [https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-3](https://github.com/priyanshsaxena01/MLOps-Priyansh_Saxena-B22EE075/tree/Assignment-3)
- **Hugging Face Model:** [https://huggingface.co/b22ee075/distilbert-goodreads-genre](https://huggingface.co/b22ee075/distilbert-goodreads-genre)

---

## 2. Docker Build Instructions

To reproduce the training and evaluation environment, please use the following commands from the root of the repository.

### Training (GPU Required)

This builds the training image and fine-tunes the model on the Goodreads dataset.

```bash
docker build -t hf-train-img .
docker run --gpus all -it \
  -v ${PWD}:/app \
  -e HF_TOKEN="<YOUR_WRITE_TOKEN>" \
  hf-train-img python src/train.py --push_to_hub --hf_username "b22ee075"
```

### Evaluation (Production Image)

This builds a lightweight production image that pulls the saved model from Hugging Face and runs evaluation immediately on startup.

```bash
docker build -f Dockerfile.eval -t hf-eval-img .
docker run --gpus all hf-eval-img
```

---

## 3. Technical Report

### A. Model Selection

I selected **DistilBERT** (`distilbert-base-cased`) for this classification task.

- **Rationale:** DistilBERT is a distilled version of BERT that retains approximately 97% of BERT's performance while being 40% smaller and 60% faster.
- **Hardware Constraints:** The training was performed on a local machine with an NVIDIA RTX 2050 (4GB VRAM). A full BERT model would likely cause Out-Of-Memory (OOM) errors with standard batch sizes. DistilBERT allowed for a stable training loop with a batch size of 8.

---

### B. Training Summary

The model was fine-tuned to classify book reviews into genres (Poetry, Mystery, Fantasy, etc.).

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Epochs | 5 *(increased from 3 to improve convergence)* |
| Batch Size | 8 per device |
| Gradient Accumulation | 2 steps *(effective batch size of 16)* |
| Learning Rate | 2e-5 *(lowered for stability)* |
| Optimizer | AdamW with weight decay 0.01 |

**Training Dynamics:**

The model started with a loss of ~2.0 and converged to a final training loss of **0.667**. To prevent overfitting, the `load_best_model_at_end=True` parameter was used to save the checkpoint with the highest validation accuracy.

---

### C. Evaluation Comparison

I evaluated the model in two distinct environments to ensure reproducibility.

| Metric | Local Training Loop | Docker Production Container |
|---|---|---|
| Accuracy | 60.62% | 60.50% |
| F1 Score | 0.6048 | 0.6039 |
| Loss | 1.12 | 1.11 |

**Conclusion:** The results are consistent across environments (within a margin of error due to floating-point nondeterminism). This confirms that the Docker container correctly creates the environment and downloads the exact model artifacts pushed to the Hugging Face Hub.

---

### D. Challenges & Solutions

During the implementation, I encountered and resolved two significant issues.

#### 1. Library Version Conflicts (PyTorch vs. Transformers)

- **Issue:** The Docker base image (`pytorch/pytorch:2.0.1`) was incompatible with the latest `transformers` library, which requires PyTorch 2.4+. This caused an `ImportError` regarding `LRScheduler`.
- **Solution:** I pinned the library versions in `requirements.txt` to `transformers==4.38.2` and `accelerate==0.27.2`, ensuring compatibility with the CUDA 11.7 environment provided by the base image.

#### 2. Non-Deterministic Label Mapping

- **Issue:** Initially, the evaluation accuracy in the Docker container dropped to ~10% (random guessing). I discovered that using `set(labels)` to generate IDs resulted in a different order every time the script ran (e.g., `"Poetry"` was ID `0` in training, but `"Mystery"` was ID `0` in evaluation).
- **Solution:** I modified `src/data.py` to use `sorted(list(set(labels)))`. This forced a deterministic alphabetical mapping for the genre labels, restoring accuracy to ~60%.
