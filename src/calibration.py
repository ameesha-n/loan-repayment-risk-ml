import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay,
)
from sklearn.metrics import brier_score_loss


def calibrate_model(model, X_train, y_train, method="sigmoid"):
    """
    Calibrate a classification model's predicted probabilities.

    method:
        sigmoid -> Platt scaling
        isotonic -> Isotonic regression
    """

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=3,
        n_jobs=-1,
    )

    calibrated_model.fit(X_train, y_train)

    return calibrated_model


def evaluate_calibration(
    model,
    calibrated_model,
    X_valid,
    y_valid,
    model_name="Model",
):
    """
    Compare raw and calibrated probability predictions.
    """

    raw_probabilities = model.predict_proba(X_valid)[:, 1]

    calibrated_probabilities = (
        calibrated_model.predict_proba(X_valid)[:, 1]
    )

    raw_brier = brier_score_loss(
        y_valid,
        raw_probabilities,
    )

    calibrated_brier = brier_score_loss(
        y_valid,
        calibrated_probabilities,
    )

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} CALIBRATION RESULTS")
    print("=" * 70)

    print(f"\nRaw Brier Score       : {raw_brier:.4f}")
    print(f"Calibrated Brier Score: {calibrated_brier:.4f}")

    improvement = raw_brier - calibrated_brier

    print(f"Brier Improvement     : {improvement:.4f}")

    if improvement > 0:
        print("Calibration improved probability quality.")
    elif improvement < 0:
        print("Calibration worsened probability quality.")
    else:
        print("Calibration produced no change.")

    return {
        "raw_probabilities": raw_probabilities,
        "calibrated_probabilities": calibrated_probabilities,
        "raw_brier": raw_brier,
        "calibrated_brier": calibrated_brier,
        "brier_improvement": improvement,
    }


def plot_calibration(
    model,
    calibrated_model,
    X_valid,
    y_valid,
    model_name="Model",
):
    """
    Plot reliability curves for raw and calibrated probabilities.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    CalibrationDisplay.from_estimator(
        model,
        X_valid,
        y_valid,
        n_bins=10,
        name=f"{model_name} - Raw",
        ax=ax,
    )

    CalibrationDisplay.from_estimator(
        calibrated_model,
        X_valid,
        y_valid,
        n_bins=10,
        name=f"{model_name} - Calibrated",
        ax=ax,
    )

    ax.set_title(f"{model_name} Probability Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")

    ax.grid()

    plt.tight_layout()

    plt.show()