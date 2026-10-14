import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

sys.path.insert(0, str(SRC_DIR))

from preprocessing import build_preprocessor
from models import train_lightgbm
from calibration import calibrate_model
from feature_engineering import (
    aggregate_bureau_data,
    aggregate_previous_applications,
    aggregate_pos_cash,
    aggregate_installments,
    aggregate_credit_card,
)


def add_auxiliary_features(applications):
    result = applications.copy()

    feature_functions = [
        aggregate_bureau_data,
        aggregate_previous_applications,
        aggregate_pos_cash,
        aggregate_installments,
        aggregate_credit_card,
    ]

    for function in feature_functions:
        print(f"Generating {function.__name__}...")
        features = function()
        result = result.merge(
            features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )
        del features

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / "application_train_extended.csv"
    test_path = DATA_DIR / "application_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run feature engineering and merging first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if not test_df["SK_ID_CURR"].is_unique:
        raise ValueError("Duplicate IDs in application_test.csv.")

    print("Generating auxiliary features for test applicants...")
    test_extended = add_auxiliary_features(test_df)

    y_train = train_df["TARGET"]
    X_train = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    test_ids = test_extended["SK_ID_CURR"].copy()
    X_test = test_extended.drop(columns=["SK_ID_CURR"])

    missing = set(X_train.columns) - set(X_test.columns)
    extra = set(X_test.columns) - set(X_train.columns)

    if missing or extra:
        raise ValueError(
            f"Feature mismatch. Missing: {sorted(missing)[:10]}; "
            f"extra: {sorted(extra)[:10]}"
        )

    X_test = X_test[X_train.columns]

    print(f"Training applicants: {len(X_train):,}")
    print(f"Test applicants: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")

    print("\nFitting preprocessing on labeled training data...")
    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nTraining raw LightGBM model...")
    model = train_lightgbm(X_train_processed, y_train)

    print("\nGenerating raw test probabilities...")
    raw_probabilities = model.predict_proba(X_test_processed)[:, 1]

    print("\nFitting sigmoid calibration with 3-fold cross-validation...")
    calibrated_model = calibrate_model(
        model,
        X_train_processed,
        y_train,
        method="sigmoid",
    )

    print("\nGenerating calibrated test probabilities...")
    calibrated_probabilities = calibrated_model.predict_proba(
        X_test_processed
    )[:, 1]

    output = pd.DataFrame({
        "SK_ID_CURR": test_ids.to_numpy(),
        "raw_probability": raw_probabilities,
        "calibrated_probability": calibrated_probabilities,
    })

    output_path = RESULTS_DIR / "test_predictions.csv"
    output.to_csv(output_path, index=False)

    print("\nTEST PREDICTIONS COMPLETED")
    print(f"Saved: {output_path}")
    print(f"Prediction rows: {len(output):,}")
    print(f"Unique applicant IDs: {output['SK_ID_CURR'].is_unique}")
    print(
        f"Raw mean probability: {raw_probabilities.mean():.4f}"
    )
    print(
        "Calibrated mean probability: "
        f"{calibrated_probabilities.mean():.4f}"
    )
    print(
        f"Calibrated range: {calibrated_probabilities.min():.6f} "
        f"to {calibrated_probabilities.max():.6f}"
    )


if __name__ == "__main__":
    main()