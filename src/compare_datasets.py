
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


def prepare_features(df, train_ids, valid_ids):
    """Select features using fixed applicant IDs for a fair comparison."""
    indexed = df.set_index("SK_ID_CURR")
    if not indexed.index.is_unique:
        raise ValueError("Duplicate applicant IDs found.")

    if "TARGET" not in indexed.columns:
        raise ValueError("TARGET column is missing.")

    y = indexed["TARGET"]
    X = indexed.drop(columns=["TARGET"])

    missing_ids = set(train_ids).union(valid_ids) - set(indexed.index)
    if missing_ids:
        raise ValueError(
            f"Dataset is missing {len(missing_ids)} expected applicant IDs."
        )

    X_train = X.loc[train_ids].drop(columns=["SK_ID_CURR"], errors="ignore")
    X_valid = X.loc[valid_ids].drop(columns=["SK_ID_CURR"], errors="ignore")
    y_train = y.loc[train_ids]
    y_valid = y.loc[valid_ids]

    return X_train, X_valid, y_train, y_valid


def run_experiment(df, dataset_name, train_ids, valid_ids):
    """Train and evaluate all models using the same applicant split."""
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    X_train, X_valid, y_train, y_valid = prepare_features(
        df, train_ids, valid_ids
    )

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

    # Validate row counts and applicant IDs.
    original = original_df.set_index("SK_ID_CURR")
    extended = extended_df.set_index("SK_ID_CURR")

    if not original.index.is_unique or not extended.index.is_unique:
        raise ValueError("Duplicate applicant IDs found.")
    if len(original) != len(extended):
        raise ValueError("Original and extended row counts differ.")

    if set(original.index) != set(extended.index):
        raise ValueError("Applicant IDs differ between datasets.")

    # Ensure the target labels match for every applicant.
    if not original["TARGET"].sort_index().equals(
        extended["TARGET"].sort_index()
    ):
        raise ValueError("Targets differ between datasets.")

    print("\nDataset alignment checks passed.")

    # Create the split ONCE and reuse the same applicant IDs.
    train_ids, valid_ids = train_test_split(
        original.index.to_numpy(),
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=original["TARGET"].to_numpy(),
    )

    print(f"Training applicants: {len(train_ids):,}")
    print(f"Validation applicants: {len(valid_ids):,}")

    original_results = run_experiment(
        original_df, "Baseline", train_ids, valid_ids
    )

    extended_results = run_experiment(
        extended_df, "Extended", train_ids, valid_ids
    )

    results_df = pd.DataFrame(original_results + extended_results)
    results_path = RESULTS_DIR / "dataset_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 90)
    print("BASELINE VS EXTENDED DATASET COMPARISON")
    print("=" * 90)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
