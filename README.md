# Assignment 3 Report: End-to-End Hugging Face & Docker

## 1. Model Selection
I selected **DistilBERT (distilbert-base-cased)**.
* **Reasoning:** DistilBERT retains 97% of BERT's performance while being 40% smaller and 60% faster. This was essential for training on a local environment (RTX 2050, 4GB VRAM) while maintaining reasonable training times and memory usage.

## 2. Training Summary
* **Hyperparameters:**
    * Epochs: 5
    * Batch Size: 8 (Training), 16 (Evaluation)
    * Gradient Accumulation: 2 steps (Effective batch size of 16)
    * Learning Rate: 2e-5
* **Performance:** The model converged successfully with a final training loss of ~0.60.
* **Final Accuracy:** ~60.6% on the test set.

## 3. Evaluation Comparison
I compared the model's performance in two environments:
1.  **Local Training Loop:** Accuracy ~60.6%
2.  **Docker Production Container:** Accuracy ~60.5% (Loss: 1.11)

**Observation:** The metrics were consistent across environments. This demonstrates that the Docker container successfully replicated the environment and pulled the correct model artifacts from Hugging Face.

## 4. Challenges & Solutions
* **CUDA Compatibility:** The default `transformers` library required PyTorch 2.4, but the Docker base image used PyTorch 2.0.1.
    * *Solution:* Pinned `transformers==4.38.2` and `accelerate==0.27.2` in `requirements.txt`.
* **Label Inconsistency:** Initially, evaluation accuracy dropped to ~10% (random guessing) because `set()` produced non-deterministic label mappings.
    * *Solution:* Implemented `sorted(list(set(labels)))` in `data.py` to ensure genre IDs (0, 1, 2...) were consistent between training and evaluation.
