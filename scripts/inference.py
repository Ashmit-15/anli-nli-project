"""
inference.py

Load the fine-tuned transformer model (roberta_anli_r2) and run
inference on a (premise, hypothesis) pair.

This is what you'll call from Docker.
"""

import argparse
from typing import Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.data_loading import LABEL2NAME


MODEL_DIR = "models/roberta_anli_r2"



def load_model_and_tokenizer(model_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(
    premise: str,
    hypothesis: str,
    model_dir: str = MODEL_DIR,
) -> Tuple[str, List[float]]:
    """
    Run a single prediction.

    Returns:
        label_name: str
        probabilities: list of floats (length 3)
    """
    tokenizer, model = load_model_and_tokenizer(model_dir)

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = int(torch.argmax(probs).item())
    label_name = LABEL2NAME[pred_idx]
    return label_name, probs.tolist()


def main():
    parser = argparse.ArgumentParser(description="Run ANLI R2 NLI inference.")
    parser.add_argument("--premise", type=str, required=True, help="Premise text.")
    parser.add_argument("--hypothesis", type=str, required=True, help="Hypothesis text.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=MODEL_DIR,
        help="Directory where the fine-tuned model is saved.",
    )
    args = parser.parse_args()

    label, probs = predict(args.premise, args.hypothesis, model_dir=args.model_dir)
    print(f"Premise:    {args.premise}")
    print(f"Hypothesis: {args.hypothesis}")
    print(f"\nPrediction: {label}")
    print(f"Probabilities [entailment, neutral, contradiction]:")
    print(probs)


if __name__ == "__main__":
    main()
