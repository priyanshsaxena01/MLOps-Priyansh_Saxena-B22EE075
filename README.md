# Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

> **Name:** Priyansh Saxena 
> **Roll No:** B22EE075 
> **Task:** English → Hindi Neural Machine Translation using a from-scratch PyTorch Transformer  
> **Goal:** Beat the baseline BLEU score (50.13) using ≤ 30 epochs via Ray Tune + Optuna + ASHA

---

## Results at a Glance

| Metric | Baseline (100 epochs) | Tuned Model (30 epochs) | Improvement |
|---|---|---|---|
| Training Time | ~2 hr 1 min | ~53.4 min | **−56%** |
| Final Loss | 0.0963 | 0.3788 | — |
| **BLEU Score** | **50.13** | **73.70** | **+23.57 pts (+47%)** |
| Baseline first matched | — | **Epoch 15** | 85 epochs saved |

The baseline BLEU of 50.13 was surpassed at **epoch 15** (BLEU = 70.45), saving 85 epochs of compute.

---

## Repository Structure

```
assignment-4/
├── b22ee075_ass_4_tuned_en_to_hi_final.ipynb  ← Ray Tune implementation (main submission)
├── en_to_hi_new.ipynb                          ← Baseline notebook (unchanged)
├── b22ee075_ass_4_best_model.pth               ← Best model weights (302 MB, see note below)
├── b22ee075_ass_4_report.pdf                   ← 2-page assignment report
├── README.md                                   ← This file
└── best_model_artifacts/
    ├── best_config.json                        ← Serialised best hyperparameters
    ├── en_vocab.pkl                            ← English vocabulary object
    └── hi_vocab.pkl                            ← Hindi vocabulary object
```

> **Note:** `b22ee075_ass_4_best_model.pth` (302.6 MB) exceeds GitHub's 100 MB limit. It is hosted on HuggingFace Hub at [`b22ee075/en-hi-transformer`](https://huggingface.co/b22ee075/en-hi-transformer).

---

## Setup & Installation

### Requirements

```bash
pip install torch torchvision torchaudio
pip install "ray[tune]" optuna
pip install nltk pandas matplotlib tqdm
pip install huggingface_hub
```

Or in Google Colab (single cell):

```bash
!pip install -q "ray[tune]" optuna huggingface_hub
```

### Dataset

The model uses the **English-Hindi parallel corpus** (`English-Hindi.tsv`) containing 13,186 sentence pairs.  
Download it from the course Google Drive:  
[https://drive.google.com/drive/folders/19fhUruT03NkYp6u0uax-zjaZ6dZHUjzz](https://drive.google.com/drive/folders/19fhUruT03NkYp6u0uax-zjaZ6dZHUjzz)

Place `English-Hindi.tsv` in the same directory as the notebooks before running.

---

## Part 1 — Baseline

Run `en_to_hi_new.ipynb` from start to finish **without any changes**. On a T4 GPU this takes ~2 hours.

**Baseline hyperparameters (hardcoded):**

| Parameter | Value |
|---|---|
| Learning Rate | 1e-4 |
| Batch Size | 64 |
| Attention Heads | 8 |
| FFN Dimension (d_ff) | 2048 |
| Dropout | 0.1 |
| Encoder/Decoder Layers | 6 |
| d_model | 512 |
| Epochs | 100 |

---

## Part 2 — Ray Tune Refactoring

The core change is wrapping the training loop in a `train_tune(config)` function that:

1. Reads all hyperparameters from the `config` dict instead of globals.
2. Reports loss each epoch with `ray.train.report({"loss": epoch_loss})`.
3. Uses `.reshape()` instead of `.view()` after the Transformer forward pass (the tensor is non-contiguous after MultiHeadAttention's internal transpose operations).

### Search Space (6 Hyperparameters)

```python
search_space = {
    "lr":         tune.loguniform(1e-5, 1e-3),    # log scale
    "batch_size": tune.choice([32, 64, 128]),
    "num_heads":  tune.choice([4, 8]),             # must divide d_model=512
    "d_ff":       tune.choice([1024, 2048, 4096]),
    "dropout":    tune.uniform(0.05, 0.35),
    "num_layers": tune.choice([2, 3, 4, 6]),
    "num_epochs": 30,                              # trial cap
}
```

### Tuner Configuration

```python
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

algo = OptunaSearch(metric="loss", mode="min")

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=30,           # max epochs per trial
    grace_period=3,     # min epochs before pruning
    reduction_factor=3  # aggressive pruning
)

tuner = tune.Tuner(
    tune.with_resources(train_tune, resources={"cpu": 1, "gpu": 1}),
    tune_config=tune.TuneConfig(
        search_alg=algo,
        scheduler=scheduler,
        num_samples=10,
    ),
    run_config=tune.RunConfig(stop={"training_iteration": 30}),
    param_space=search_space,
)
results = tuner.fit()
```

---

## Part 3 — Best Configuration & Final Training

### Best Hyperparameters (found by Optuna)

```json
{
  "lr": 0.00013507,
  "batch_size": 32,
  "num_heads": 4,
  "d_ff": 4096,
  "dropout": 0.29935,
  "num_layers": 6,
  "num_epochs": 30
}
```

| Key Change vs Baseline | Effect |
|---|---|
| LR: 1e-4 → 1.35e-4 | Slightly faster convergence |
| Batch: 64 → 32 | Higher gradient noise = implicit regularisation |
| Heads: 8 → 4 | Fewer, wider attention subspaces |
| d_ff: 2048 → 4096 | More representational capacity |
| Dropout: 0.10 → 0.30 | Stronger regularisation; prevents overfitting on 13K pairs |

### Training Progress (Best Model)

| Epoch | Loss | BLEU (%) |
|---|---|---|
| 1 | 5.40 | — |
| 5 | 3.19 | 26.77 |
| 10 | 2.04 | — |
| **15** | **1.19** | **70.45 ✓ baseline matched** |
| 20 | 0.71 | 70.56 |
| 25 | 0.49 | 66.25 |
| **30** | **0.38** | **73.70** |

---

## Loading the Best Model

```python
import torch, pickle
from huggingface_hub import hf_hub_download

# Download from HuggingFace Hub
model_path    = hf_hub_download(repo_id="b22ee075/en-hi-transformer", filename="b22ee075_ass_4_best_model.pth")
en_vocab_path = hf_hub_download(repo_id="b22ee075/en-hi-transformer", filename="en_vocab.pkl")
hi_vocab_path = hf_hub_download(repo_id="b22ee075/en-hi-transformer", filename="hi_vocab.pkl")

with open(en_vocab_path, 'rb') as f: en_vocab = pickle.load(f)
with open(hi_vocab_path, 'rb') as f: hi_vocab = pickle.load(f)

model = Transformer(
    src_vocab=len(en_vocab), tgt_vocab=len(hi_vocab),
    d_model=512, n_layers=6, n_heads=4, d_ff=4096,
    max_len=50, dropout=0.0
)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
```

### Sample Translations

| English | Hindi (Model Output) |
|---|---|
| I love you. | मैं तुमसे प्यार करता हूँ। |
| What is your name? | आपका नाम क्या है? |
| How are you? | आप कैसे हैं? |
| The weather is nice today. | मौसम आज अच्छा है। |

---

## Key Technical Notes

- **`.reshape()` vs `.view()`**: The Transformer's MultiHeadAttention module internally transposes tensors, making them non-contiguous. `.view()` raises a RuntimeError on non-contiguous tensors; `.reshape()` handles this transparently.
- **ASHA pruning**: 8 of 10 trials were pruned after exactly 3 epochs (grace period), saving ~80% of potential compute.
- **GPU allocation**: `ray.init(num_gpus=1)` + `tune.with_resources({"gpu": 1})` is required to expose the Colab T4 to Ray workers.
- **BLEU evaluation set**: 5 fixed sentence pairs used for fast in-training BLEU checks. A larger held-out set would reduce score variance.

---

## Grading Rubric Compliance

| Criterion | Status |
|---|---|
| Baseline Execution & Metrics Documented | ✅ Loss=0.0963, BLEU=50.13, Time=~2hr 1min |
| Code Refactored for Ray Tune | ✅ `train_tune(config)` with `ray.train.report` |
| ≥ 4 Hyperparameters + OptunaSearch | ✅ 6 hyperparameters, OptunaSearch + ASHAScheduler |
| Efficiency Goal: Beat Baseline BLEU in ≤ 30 epochs | ✅ BLEU=73.70 at epoch 30; baseline first matched at epoch 15 |
| Report (PDF) | ✅ `b22ee075_ass_4_report.pdf` |

---

*Environment: Google Colab, NVIDIA T4 GPU, Python 3.10, PyTorch 2.x, Ray 2.x, Optuna 3.x*
