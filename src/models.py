import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_application_data
from preprocessing import prepare_data, build_preprocessor
from evaluation import evaluate_model, compare_models
from calibration import (
    calibrate_model,
    evaluate_calibration,
    plot_calibration,
)
from explainability import (
    compute_shap_values,
    plot_global_importance,
    plot_local_explanation,
)


RANDOM_STATE = 42


def train_logistic_regression(X_train_processed, y_train):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train_processed, y_train)

    return model


def train_random_forest(X_train_processed, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train_processed, y_train)

    return model


def train_lightgbm(X_train_processed, y_train):
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train_processed, y_train)

    return model


if __name__ == "__main__":

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================

    print("Loading dataset...")

    df = load_application_data()

    print(f"Dataset shape: {df.shape}")


    # ============================================================
    # 2. TRAIN / VALIDATION SPLIT
    # ============================================================

    X_train, X_valid, y_train, y_valid = prepare_data(df)

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")


    # ============================================================
    # 3. PREPROCESSING
    # ============================================================

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    print("\nPreprocessing completed.")

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed validation shape: "
        f"{X_valid_processed.shape}"
    )


    # ============================================================
    # 4. TRAIN LOGISTIC REGRESSION
    # ============================================================

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train,
    )

    print("Logistic Regression trained successfully.")


    # ============================================================
    # 5. TRAIN RANDOM FOREST
    # ============================================================

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train,
    )

    print("Random Forest trained successfully.")


    # ============================================================
    # 6. TRAIN LIGHTGBM
    # ============================================================

    print("\nTraining LightGBM...")

    lightgbm_model = train_lightgbm(
        X_train_processed,
        y_train,
    )

    print("LightGBM trained successfully.")


    # ============================================================
    # 7. EVALUATE BASELINE MODELS
    # ============================================================

    print("\nEvaluating baseline models...")

    logistic_results = evaluate_model(
        logistic_model,
        X_valid_processed,
        y_valid,
        "Logistic Regression",
    )

    random_forest_results = evaluate_model(
        random_forest_model,
        X_valid_processed,
        y_valid,
        "Random Forest",
    )

    lightgbm_results = evaluate_model(
        lightgbm_model,
        X_valid_processed,
        y_valid,
        "LightGBM",
    )


    # ============================================================
    # 8. MODEL COMPARISON
    # ============================================================

    results = {
        "Logistic Regression": logistic_results,
        "Random Forest": random_forest_results,
        "LightGBM": lightgbm_results,
    }

    compare_models(results)


    # ============================================================
    # 9. CALIBRATE LIGHTGBM
    # ============================================================

    print("\n" + "=" * 70)
    print("CALIBRATING LIGHTGBM")
    print("=" * 70)

    print("\nUsing sigmoid calibration (Platt scaling)...")

    calibrated_lightgbm = calibrate_model(
        lightgbm_model,
        X_train_processed,
        y_train,
        method="sigmoid",
    )

    print("Calibration completed successfully.")


    # ============================================================
    # 10. EVALUATE CALIBRATION
    # ============================================================

    calibration_results = evaluate_calibration(
        lightgbm_model,
        calibrated_lightgbm,
        X_valid_processed,
        y_valid,
        model_name="LightGBM",
    )


    # ============================================================
    # 11. PLOT CALIBRATION CURVE
    # ============================================================

    print("\nGenerating calibration curve...")

    plot_calibration(
        lightgbm_model,
        calibrated_lightgbm,
        X_valid_processed,
        y_valid,
        model_name="LightGBM",
    )


    # ============================================================
    # 12. CALIBRATION SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)

    print(
        f"\nRaw LightGBM Brier Score        : "
        f"{calibration_results['raw_brier']:.4f}"
    )

    print(
        f"Calibrated LightGBM Brier Score: "
        f"{calibration_results['calibrated_brier']:.4f}"
    )

    print(
        f"Brier Score Improvement         : "
        f"{calibration_results['brier_improvement']:.4f}"
    )


    # ============================================================
    # 13. SHAP EXPLAINABILITY
    # ============================================================

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY")
    print("=" * 70)

    # Get the feature names after preprocessing.
    # One-hot encoding expands the original features,
    # so we use the transformed feature names here.
    feature_names = preprocessor.get_feature_names_out()

    print(
        f"\nNumber of transformed features: "
        f"{len(feature_names)}"
    )

    # Calculate SHAP values using a sample of validation data.
    # We use the raw LightGBM model because TreeSHAP is
    # designed for tree-based models.
    explainer, shap_values, X_shap = compute_shap_values(
        lightgbm_model,
        X_valid_processed,
        sample_size=500,
    )

    print("SHAP values calculated successfully.")


    # ============================================================
    # 14. GLOBAL SHAP EXPLANATION
    # ============================================================

    print("\nGenerating global SHAP importance plot...")

    plot_global_importance(
        shap_values,
        feature_names,
    )


    # ============================================================
    # 15. LOCAL SHAP EXPLANATION
    # ============================================================

    print("\nGenerating individual applicant explanation...")

    plot_local_explanation(
        shap_values,
        sample_index=0,
    )


    # ============================================================
    # 16. FINAL MESSAGE
    # ============================================================

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 70)

    print("\nGenerated files:")

    print("  results/calibration_curve.png")
    print("  results/shap_summary.png")
    print("  results/shap_local.png")

    print("\nExperiment completed successfully!")