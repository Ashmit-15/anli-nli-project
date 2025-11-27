"""
evaluation.py
Evaluation helpers for baseline and transformer models.
"""

from typing import List, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_and_print(
    y_true,
    y_pred,
    target_names: List[str],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Print classification report and confusion matrix, 
    and return structured data.
    """

    if prefix:
        print(f"\n===== {prefix} =====")

    # Print the human-readable report
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Return the structured report for logging
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True
    )

    return {
        "classification_report": report_dict,
        "confusion_matrix": cm
    }
