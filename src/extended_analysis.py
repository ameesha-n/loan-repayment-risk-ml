
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import load_application_data
from preprocessing import build_preprocessor
from models import train_lightgbm
from calibration import calibrate_model
from explainability import (
    compute_shap_values,
    plot_global_importance,
    plot_local_explanation,
)

RESULTS_DIR = ROOT_DIR / "results"
RANDOM_STATE = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load the extended dataset and create the split
    # --------------------------------------------------
    print("Loading extended dataset...")
    df = load_application_data(extended=True)

    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining applicants: {len(X_train):,}")
    print(f"Validation applicants: {len(X_valid):,}")
    print(f"Training default rate: {y_train.mean():.4f}")
    print(f"Validation default rate: {y_valid.mean():.4f}")

    # --------------------------------------------------
    # 2. Preprocess without fitting on validation data
    # --------------------------------------------------
    print("\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    feature_names = preprocessor.get_feature_names_out().tolist()

    # --------------------------------------------------
    # 3. Train the raw LightGBM model
    # --------------------------------------------------
    print("\nTraining extended LightGBM...")
    model = train_lightgbm(X_train_processed, y_train)

    raw_probabilities = model.predict_proba(
        X_valid_processed
    )[:, 1]

    raw_brier = brier_score_loss(y_valid, raw_probabilities)

    # --------------------------------------------------
    # 4. Calibrate using training data only
    # --------------------------------------------------
    print("\nCalibrating extended LightGBM...")
    calibrated_model = calibrate_model(
        model,
        X_train_processed,
        y_train,
        method="sigmoid",
    )

    calibrated_probabilities = calibrated_model.predict_proba(
        X_valid_processed
    )[:, 1]

    calibrated_brier = brier_score_loss(
        y_valid,
        calibrated_probabilities,
    )

    baseline_probability = y_valid.mean()
    baseline_brier = brier_score_loss(
        y_valid,
        np.full(len(y_valid), baseline_probability),
    )

    # --------------------------------------------------
    # 5. Print probability diagnostics
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("EXTENDED LIGHTGBM: PROBABILITY DIAGNOSTICS")
    print("=" * 65)

    print(f"Actual validation default rate : {y_valid.mean():.4f}")
    print(f"Raw mean predicted probability : {raw_probabilities.mean():.4f}")
    print(
        "Calibrated mean probability   : "
        f"{calibrated_probabilities.mean():.4f}"
    )

    print("\nProbability ranges")
    print(
        f"Raw model       : {raw_probabilities.min():.6f} "
        f"to {raw_probabilities.max():.6f}"
    )
    print(
        f"Calibrated model: {calibrated_probabilities.min():.6f} "
        f"to {calibrated_probabilities.max():.6f}"
    )

    print("\nBrier scores (lower is better)")
    print(f"Raw LightGBM          : {raw_brier:.4f}")
    print(f"Calibrated LightGBM   : {calibrated_brier:.4f}")
    print(f"Constant-rate baseline: {baseline_brier:.4f}")
    print(f"Raw-to-calibrated gain: {raw_brier - calibrated_brier:.4f}")

    # --------------------------------------------------
    # 6. Save row-level validation predictions
    # --------------------------------------------------
    predictions_path = RESULTS_DIR / "extended_validation_predictions.csv"

    pd.DataFrame({
        "actual_target": y_valid.to_numpy(),
        "raw_probability": raw_probabilities,
        "calibrated_probability": calibrated_probabilities,
    }).to_csv(predictions_path, index=False)

    print(f"\nValidation predictions saved: {predictions_path}")

    # --------------------------------------------------
    # 7. Reliability curves
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    CalibrationDisplay.from_estimator(
        model,
        X_valid_processed,
        y_valid,
        n_bins=10,
        name="Raw LightGBM",
        ax=ax,
    )

    CalibrationDisplay.from_estimator(
        calibrated_model,
        X_valid_processed,
        y_valid,
        n_bins=10,
        name="Calibrated LightGBM",
        ax=ax,
    )

    ax.set_title("Extended LightGBM Probability Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    calibration_path = RESULTS_DIR / "extended_calibration_curve.png"
    fig.savefig(calibration_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Calibration plot saved: {calibration_path}")

    # --------------------------------------------------
    # 8. Save calibration metrics
    # --------------------------------------------------
    metrics_path = RESULTS_DIR / "extended_calibration_metrics.csv"

    pd.DataFrame([{
        "model": "Extended LightGBM",
        "validation_default_rate": y_valid.mean(),
        "raw_mean_probability": raw_probabilities.mean(),
        "calibrated_mean_probability": calibrated_probabilities.mean(),
        "raw_brier_score": raw_brier,
        "calibrated_brier_score": calibrated_brier,
        "constant_rate_baseline_brier": baseline_brier,
        "brier_improvement": raw_brier - calibrated_brier,
    }]).to_csv(metrics_path, index=False)

    print(f"Calibration metrics saved: {metrics_path}")

    # --------------------------------------------------
    # 9. SHAP explanations for the raw LightGBM model
    # --------------------------------------------------
    print("\nComputing SHAP explanations...")

    explainer, shap_values, X_sample = compute_shap_values(
        model,
        X_valid_processed,
        sample_size=500,
    )

    plot_global_importance(
        shap_values,
        feature_names,
        output_path=str(RESULTS_DIR / "extended_shap_summary.png"),
    )

    plot_local_explanation(
        shap_values,
        sample_index=0,
        output_path=str(RESULTS_DIR / "extended_shap_local.png"),
    )

    print("\nExtended analysis completed successfully.")


if __name__ == "__main__":
    main()
