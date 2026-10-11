
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

from data_loader import load_application_data
from preprocessing import build_preprocessor
from models import (
    train_logistic_regression,
    train_random_forest,
    train_lightgbm,
)
from evaluation import evaluate_model

RANDOM_STATE = 42
RESULTS_DIR = ROOT_DIR / "results"


def prepare_features(df):
    """Separate target and remove the applicant identifier."""
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def run_experiment(df, dataset_name):
    """Train and evaluate all three models on one dataset."""
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    X_train, X_valid, y_train, y_valid = prepare_features(df)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    print(f"Training shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_valid_processed.shape}")

    trainers = {
        "Logistic Regression": train_logistic_regression,
        "Random Forest": train_random_forest,
        "LightGBM": train_lightgbm,
    }

    results = []

    for model_name, trainer in trainers.items():
        print(f"\nTraining {model_name} on {dataset_name}...")

        model = trainer(X_train_processed, y_train)

        metrics = evaluate_model(
            model,
            X_valid_processed,
            y_valid,
            f"{dataset_name} - {model_name}",
        )

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "brier_score": metrics["brier_score"],
        })

        del model

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading original dataset...")
    original_df = load_application_data()

    print("\nLoading extended dataset...")
    extended_df = load_application_data(extended=True)

    if len(original_df) != len(extended_df):
        raise ValueError("Original and extended row counts differ.")

    if not original_df["SK_ID_CURR"].equals(
        extended_df["SK_ID_CURR"]
    ):
        raise ValueError(
            "Applicant order or IDs differ between datasets."
        )

    if not original_df["TARGET"].equals(extended_df["TARGET"]):
        raise ValueError(
            "Targets differ between original and extended datasets."
        )

    print("\nDataset alignment checks passed.")

    original_results = run_experiment(
        original_df,
        "Baseline",
    )

    extended_results = run_experiment(
        extended_df,
        "Extended",
    )

    results_df = pd.DataFrame(
        original_results + extended_results
    )

    results_path = RESULTS_DIR / "dataset_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 100)
    print("BASELINE VS EXTENDED DATASET COMPARISON")
    print("=" * 100)

    print(
        results_df[
            [
                "dataset",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
