ANLI R2 – Natural Language Inference (3-Way Classification)
End-to-End Machine Learning Pipeline

This project solves a 3-class NLI task using the ANLI R2 dataset:

entailment

neutral

contradiction

The pipeline includes:

Data loading

EDA

Classical ML baselines

Transformer fine-tuning

Evaluation

Docker model packaging

🚀 1. Dataset

Using facebook/anli (round 2):

Train: 45,460 examples

Dev: 1,000

Test: 1,000

Only the R2 split is used.

🚀 2. Models Trained
✔ Classical ML Baselines

TF-IDF + Logistic Regression

TF-IDF + Linear SVM

TF-IDF + XGBoost

✔ Transformer Fine-Tuning

Fine-tuned DistilRoBERTa-base on ANLI R2

Saved model (models/roberta_anli_r2/)



3. Repository Structure
src/               # core modules (loading, preprocessing, evaluation)
scripts/           # training and inference scripts
notebooks/         # EDA, baselines, transformer training
models/            # saved transformer model (optional)
run_pipeline.py    # unified CLI pipeline
Dockerfile         # containerization
requirements.txt   # dependencies

4. How to Run (Local)
Baseline models:
python run_pipeline.py --mode eval_baseline

Transformer evaluation (trained model required):
python run_pipeline.py --mode demo --premise "A" --hypothesis "B"

🚀 5. Docker (Optional)

The Dockerfile is provided as part of the assignment.
It is configured to run inference via:

CMD ["python", "run_pipeline.py", "--mode", "demo", ...]