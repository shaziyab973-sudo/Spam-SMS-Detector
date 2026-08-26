import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from typing import Dict, Any, Tuple


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Evaluates trained model performance on test dataset and saves metrics.json.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # Predict probabilities and apply threshold
    y_probs = model.predict(X_test, verbose=0).ravel()
    y_preds = (y_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_preds))
    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_probs))

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold_used": threshold
    }

    metrics_path = os.path.join(reports_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\n================ MODEL EVALUATION REPORT ================")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score : {metrics['f1_score'] * 100:.2f}%")
    print(f"ROC-AUC  : {metrics['roc_auc'] * 100:.2f}%")
    print("========================================================")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_preds, target_names=['Legit (Ham)', 'Spam']))

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_preds)
    plot_confusion_matrix(cm, reports_dir=reports_dir)

    return metrics


def plot_confusion_matrix(cm: np.ndarray, reports_dir: str = "reports") -> str:
    """Generates and saves a confusion matrix heatmap plot."""
    os.makedirs(reports_dir, exist_ok=True)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Legit (Ham)', 'Spam'],
        yticklabels=['Legit (Ham)', 'Spam'],
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Confusion Matrix - Spam SMS Detection", fontsize=13, pad=12, weight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved confusion matrix heatmap plot to {cm_path}")
    return cm_path


def plot_training_history(history, reports_dir: str = "reports") -> str:
    """Generates and saves Training vs Validation Loss & Accuracy curves."""
    os.makedirs(reports_dir, exist_ok=True)
    history_path = os.path.join(reports_dir, "training_history.png")

    hist_dict = history.history if hasattr(history, 'history') else history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Plot
    axes[0].plot(hist_dict.get('loss', []), label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(hist_dict.get('val_loss', []), label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].set_title('Training vs Validation Loss', fontsize=12, weight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy Plot
    axes[1].plot(hist_dict.get('accuracy', []), label='Train Accuracy', color='#2ca02c', linewidth=2)
    axes[1].plot(hist_dict.get('val_accuracy', []), label='Val Accuracy', color='#d62728', linewidth=2, linestyle='--')
    axes[1].set_title('Training vs Validation Accuracy', fontsize=12, weight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(history_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved training history curves plot to {history_path}")
    return history_path
