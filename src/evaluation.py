"""
Model evaluation utilities for sentiment classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Not Recommended", "Recommended"]))
    
    return y_pred


def plot_confusion_matrix(y_test, y_pred, model_name="Model"):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Rec", "Recommended"],
        yticklabels=["Not Rec", "Recommended"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def compare_models(results_dict, y_test):
    """Compare multiple models in a summary table.
    
    Args:
        results_dict: {"Model Name": y_pred, ...}
        y_test: true labels
    """
    rows = []
    for name, y_pred in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        })
    
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\nModel Comparison:")
    print(df.to_string(index=False, float_format="%.4f"))
    return df