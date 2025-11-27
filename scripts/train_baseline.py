"""
train_baseline.py

Train a TF-IDF + Logistic Regression baseline model on ANLI R2.
"""

import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_loading import load_anli_r2, LABEL2NAME
from src.preprocessing import combine_premise_hypothesis
from src.evaluation import evaluate_and_print


def prepare_xy(split):
    X = [combine_premise_hypothesis(example) for example in split]
    y = split["label"]
    return X, y


def train_baseline_model(save_path: str = "baseline_tfidf_model.joblib"):
    # 1. Load data
    train, val, test = load_anli_r2()

    # 2. Prepare X, y
    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(val)
    X_test, y_test = prepare_xy(test)

    # 3. Define pipeline
    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=100_000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 4. Train
    print("Training TF-IDF + Logistic Regression baseline...")
    clf.fit(X_train, y_train)
    print("Training complete.")

    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME.keys())]

    # 5. Evaluate on validation set
    print("\nValidation set performance:")
    val_pred = clf.predict(X_val)
    evaluate_and_print(y_val, val_pred, target_names, prefix="Validation")

    # 6. Evaluate on test set (for final reporting)
    print("\nTest set performance:")
    test_pred = clf.predict(X_test)
    evaluate_and_print(y_test, test_pred, target_names, prefix="Test")

    # 7. Save model
    joblib.dump(clf, save_path)
    print(f"\nSaved baseline model to: {save_path}")


if __name__ == "__main__":
    train_baseline_model()
