import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)


def evaluate_model(model, X_valid, y_valid, model_name):
    """
    Evaluate a classification model using both
    classification and probability-based metrics.
    """

    # Class prediction: 0 or 1
    y_pred = model.predict(X_valid)

    # Probability of positive class
    y_prob = model.predict_proba(X_valid)[:, 1]

    # Classification metrics
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)

    # Ranking metrics
    roc_auc = roc_auc_score(y_valid, y_prob)
    pr_auc = average_precision_score(y_valid, y_prob)

    # Probability calibration metric
    brier = brier_score_loss(y_valid, y_prob)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Brier    : {brier:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def compare_models(results):
    """
    Print a comparison table for multiple models.
    """

    print("\n" + "=" * 85)
    print("MODEL COMPARISON")
    print("=" * 85)

    print(
        f"{'Metric':<15}"
        f"{'Logistic Regression':<22}"
        f"{'Random Forest':<18}"
        f"{'LightGBM':<15}"
    )

    print("-" * 85)

    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC-AUC", "roc_auc"),
        ("PR-AUC", "pr_auc"),
        ("Brier Score", "brier_score"),
    ]

    for display_name, key in metrics:
        print(
            f"{display_name:<15}"
            f"{results['Logistic Regression'][key]:<22.4f}"
            f"{results['Random Forest'][key]:<18.4f}"
            f"{results['LightGBM'][key]:<15.4f}"
        )