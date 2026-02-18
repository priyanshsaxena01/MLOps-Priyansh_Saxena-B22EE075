import os
import argparse
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import prepare_data
from utils import compute_metrics
from huggingface_hub import login

def train(args):
    # Authenticate if pushing to Hub
    if args.push_to_hub:
        login(token=os.environ.get("HF_TOKEN"))

    # Prepare Data
    train_dataset, test_dataset, label2id, id2label = prepare_data()
    
    # Load Model
    print(f"Loading model: {args.model_name}")
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # --- OPTIMIZED TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=5,               # INCREASED: From 3 to 5 to give more learning time
        learning_rate=2e-5,               # LOWERED: From 5e-5 to 2e-5 for better convergence
        per_device_train_batch_size=8,    # Kept low for 4GB VRAM
        gradient_accumulation_steps=2,    # ADDED: Simulates batch size of 16 (8*2) without extra memory
        per_device_eval_batch_size=16,
        warmup_steps=500,                 # INCREASED: More warmup steps for stability
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=100,                   # Evaluate more frequently
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,      # CRITICAL: Ensures you save the best version, not just the last
        metric_for_best_model="accuracy", # Optimize specifically for accuracy
        push_to_hub=args.push_to_hub,
        hub_model_id=f"{args.hf_username}/distilbert-goodreads-genre" if args.push_to_hub else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting Training...")
    trainer.train()

    print("Saving Best Model...")
    trainer.save_model(args.output_dir)
    
    if args.push_to_hub:
        print("Pushing to Hub...")
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert-base-cased")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_username", type=str, help="Your HuggingFace Username")
    args = parser.parse_args()
    train(args)