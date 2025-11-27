"""
train_transformer.py

Fine-tune a transformer model (DistilRoBERTa by default) on ANLI R2
for 3-way NLI classification.
"""

from typing import Dict

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.data_loading import load_anli_r2, LABEL2NAME
from src.preprocessing import get_tokenizer, tokenize_batch


def tokenize_splits(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    model_name: str = "distilroberta-base",
    max_length: int = 256,
):
    tokenizer = get_tokenizer(model_name)

    def _fn(batch):
        return tokenize_batch(batch, tokenizer, max_length=max_length)

    tokenized_train = train.map(_fn, batched=True)
    tokenized_val = val.map(_fn, batched=True)
    tokenized_test = test.map(_fn, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_val = tokenized_val.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenizer, tokenized_train, tokenized_val, tokenized_test


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


def train_transformer(
    model_name: str = "distilroberta-base",
    output_dir: str = "./checkpoints_roberta",
    num_train_epochs: int = 3,
):
    # 1. Load data
    train, val, test = load_anli_r2()

    # 2. Tokenize
    tokenizer, tokenized_train, tokenized_val, tokenized_test = tokenize_splits(
        train, val, test, model_name=model_name
    )

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    # 4. Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_steps=100,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    print("Starting transformer training...")
    trainer.train()
    print("Training complete.")

    # 7. Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(tokenized_test)
    print("Test metrics:", test_results)

    # Detailed classification report on test set
    raw_preds = trainer.predict(tokenized_test)
    preds = np.argmax(raw_preds.predictions, axis=-1)
    labels = raw_preds.label_ids

    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME.keys())]
    print("\nClassification report (test):")
    print(classification_report(labels, preds, target_names=target_names))
    print("Confusion matrix (test):\n", confusion_matrix(labels, preds))

    # 8. Save final model (for inference / Docker)
    save_dir = "roberta_anli_r2"
    print(f"\nSaving fine-tuned model to: {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    train_transformer()
