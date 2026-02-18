import argparse
import torch
from transformers import DistilBertForSequenceClassification, Trainer
from data import prepare_data
from utils import compute_metrics
import json

def evaluate(args):
    # Prepare only test data
    _, test_dataset, label2id, id2label = prepare_data()

    print(f"Loading model from: {args.model_path}")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_path)
    
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Evaluating...")
    metrics = trainer.evaluate()
    
    print("Evaluation Results:", metrics)
    
    with open("eval_results.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF Model ID")
    args = parser.parse_args()
    evaluate(args)