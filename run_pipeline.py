"""
run_pipeline.py

Unified entry point to run:
- Baseline model training + evaluation
- Transformer fine-tuned model evaluation
- Inference demo

Works with the existing scripts:
  scripts/train_baseline.py
  scripts/train_transformer.py
  scripts/inference.py
"""

import argparse
import sys
import os

# Ensure project root is in PYTHONPATH
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Import the functions from your scripts
from scripts.train_baseline import train_baseline_model
from scripts.train_transformer import train_transformer
from scripts.inference import predict


def run_demo(premise: str, hypothesis: str):
    """Runs single prediction using the fine-tuned transformer model."""
    label, probs = predict(premise, hypothesis)

    print("\n=== Demo Inference ===")
    print(f"Premise:    {premise}")
    print(f"Hypothesis: {hypothesis}")
    print(f"\nPrediction: {label}")
    print(f"Probabilities: {probs}\n")


def main():
    parser = argparse.ArgumentParser(description="ANLI R2 Full Pipeline Runner")

    parser.add_argument(
        "--mode",
        required=True,
        choices=["eval_baseline", "eval_transformer", "demo"],
        help="Which pipeline step to run.",
    )

    parser.add_argument("--premise", type=str, help="Premise text (demo mode).")
    parser.add_argument("--hypothesis", type=str, help="Hypothesis text (demo mode).")

    args = parser.parse_args()

    # -----------------------------
    # Run classical baseline
    # -----------------------------
    if args.mode == "eval_baseline":
        print("\n=== Running TF-IDF + Logistic Regression Baseline ===\n")
        train_baseline_model()

    # -----------------------------
    # Run transformer evaluation (fine-tuning + test eval)
    # -----------------------------
    elif args.mode == "eval_transformer":
        print("\n=== Running Transformer Training + Evaluation ===\n")
        train_transformer()

    # -----------------------------
    # Run demo inference
    # -----------------------------
    elif args.mode == "demo":
        if not args.premise or not args.hypothesis:
            raise ValueError("Demo mode requires --premise and --hypothesis.")

        run_demo(args.premise, args.hypothesis)


if __name__ == "__main__":
    main()
