# ANLI Round 2 – Natural Language Inference (NLI) Project  
## End-to-End ML + Transformer Fine-Tuning + Docker Deployment

Comprehensive exploration and production-ready implementation of **Natural Language Inference** models on the **Adversarial NLI (ANLI) Round 2 dataset** — covering everything from **EDA** → **ML Baselines** → **Transformer Fine-Tuning** → **Deployment with Docker**.

---

## 🎯 Project Overview

**Task:** Natural Language Inference (NLI)  
**Dataset:** Adversarial NLI (ANLI) Round 2  
**Goal:** Predict the relationship between *premise* and *hypothesis*:

- Entailment  
- Neutral  
- Contradiction  

**Dataset Size (ANLI R2):**

| Split | Count |
|-------|--------|
| Train | 45,460 |
| Dev   | 1,000  |
| Test  | 1,000  |

ANLI is **intentionally adversarial** and significantly harder than SNLI/MNLI, making it a strong benchmark for model robustness.

---

## 📊 Results Summary

| Model                         | Accuracy | Macro F1 | Notes                          |
|------------------------------|----------|----------|--------------------------------|
| **DistilRoBERTa (fine-tuned)** | XX%      | XX       | Best model; fine-tuned for NLI |
| XGBoost                       | ~38%     | ~0.33    | Strongest ML baseline          |
| Linear SVM                    | ~36%     | ~0.33    | Good baseline                  |
| Logistic Regression           | ~35%     | ~0.33    | Baseline                       |
| DistilRoBERTa (no fine-tune) | ~33%     | ~0.24    | Zero-shot baseline             |

> Replace **XX%** with your actual final results.

---

## 🧠 High-Level Workflow

✔ Exploratory Data Analysis  
✔ Preprocessing & feature construction  
✔ Traditional ML baselines  
✔ Transformer fine-tuning (DistilRoBERTa)  
✔ Evaluation on Dev & Test  
✔ Docker-based inference pipeline  

---

## 📁 Repository Structure

```plaintext
.
├── notebooks/
│   ├── eda.ipynb                 # Data exploration
│   ├── baseline_ml.ipynb         # LogReg, SVM, XGBoost
│   └── RoBerta.ipynb             # Transformer fine-tuning
│
├── src/
│   ├── data_loading.py           # Load ANLI R2 dataset
│   ├── preprocessing.py          # Text construction & tokenization
│   ├── evaluation.py             # Metrics & reports
│
├── scripts/
│   ├── train_baseline.py         # Train TF-IDF + ML models
│   ├── train_transformer.py      # Train DistilRoBERTa
│   └── inference.py              # Run inference on text pairs
│
├── models/                       # (Ignored in Git; local only)
│   └── roberta_anli_r2/          # Saved fine-tuned transformer model
│
├── run_pipeline.py               # Unified CLI pipeline
├── Dockerfile                    # Inference container
├── requirements.txt
├── README.md
└── .dockerignore




🚀 Two Ways to Use This Project
1️⃣ Jupyter Notebooks (Exploration)

Best for experimentation and understanding the pipeline.

Includes:

notebooks/eda.ipynb → label distribution, text length, examples

notebooks/baseline_ml.ipynb → TF-IDF + Logistic/SVM/XGBoost

notebooks/RoBerta.ipynb → full DistilRoBERTa fine-tuning

Great for learning and showcasing methodology.

2️⃣ Production Pipeline (Scripts + Automation)

Everything modular, script-based, and deployable.

Run classical ML baselines:
python run_pipeline.py --mode eval_baseline

Run transformer inference:
python run_pipeline.py --mode demo \
    --premise "A man is playing music" \
    --hypothesis "A man is playing guitar"

Train models manually:
python scripts/train_baseline.py
python scripts/train_transformer.py

📦 Docker Deployment (Inference-Ready Container)

This project includes a Dockerfile that packages:

The inference script

All dependencies

Model loading

A default demo prediction

Build the image:
docker build -t anli-nli .

Run inference:
docker run --rm anli-nli

Example CMD in Dockerfile:
CMD ["python", "run_pipeline.py", "--mode", "demo",
     "--premise", "A man is playing music",
     "--hypothesis", "A man is playing guitar"]


Even if the interviewer doesn’t run Docker, including it demonstrates deployment capability.
