
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

FEATURE_FILES = [
    "bureau_features.csv",
    "previous_application_features.csv",
    "pos_cash_features.csv",
    "installment_features.csv",
    "credit_card_features.csv",
]


def build_merged_dataset():
    """Merge auxiliary features into the original training applications."""

    application_path = DATA_DIR / "application_train.csv"

    if not application_path.exists():
        raise FileNotFoundError(
            f"Application dataset not found: {application_path}"
        )

    applications = pd.read_csv(application_path)

    original_rows = len(applications)

    if not applications["SK_ID_CURR"].is_unique:
        raise ValueError(
            "Application dataset contains duplicate applicant IDs."
        )

    if "TARGET" not in applications.columns:
        raise ValueError("TARGET column is missing from application data.")

    print(f"Original application rows: {original_rows:,}")
    print(f"Original application columns: {len(applications.columns)}")

    for filename in FEATURE_FILES:
        path = PROCESSED_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {path}\n"
                "Run src/feature_engineering.py first."
            )

        print(f"\nLoading {filename}...")
        features = pd.read_csv(path)

        if "SK_ID_CURR" not in features.columns:
            raise ValueError(
                f"SK_ID_CURR is missing from {filename}"
            )

        if not features["SK_ID_CURR"].is_unique:
            raise ValueError(
                f"Duplicate applicant IDs found in {filename}"
            )

        feature_columns = [
            column
            for column in features.columns
            if column != "SK_ID_CURR"
        ]

        overlapping_columns = (
            set(feature_columns) & set(applications.columns)
        )

        if overlapping_columns:
            raise ValueError(
                f"Duplicate feature names in {filename}: "
                f"{sorted(overlapping_columns)}"
            )

        applications = applications.merge(
            features,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

        if len(applications) != original_rows:
            raise ValueError(
                f"Row count changed after merging {filename}."
            )

        print(f"Added features: {len(feature_columns)}")
        print(f"Current dataset shape: {applications.shape}")

        del features

    # Final validation
    if len(applications) != original_rows:
        raise ValueError("Final row count does not match original data.")

    if not applications["SK_ID_CURR"].is_unique:
        raise ValueError("Final applicant IDs are not unique.")

    if "TARGET" not in applications.columns:
        raise ValueError("TARGET was lost during merging.")

    if applications["TARGET"].isna().any():
        raise ValueError("TARGET contains missing values.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "application_train_extended.csv"
    applications.to_csv(output_path, index=False)

    added_features = len(applications.columns) - 122

    print("\n" + "=" * 60)
    print("MERGED DATASET VALIDATION")
    print("=" * 60)
    print(f"Rows: {len(applications):,}")
    print(f"Columns: {len(applications.columns)}")
    print(f"Added auxiliary features: {added_features}")
    print(f"Target preserved: {'TARGET' in applications.columns}")
    print(
        f"Unique applicant IDs: "
        f"{applications['SK_ID_CURR'].is_unique}"
    )
    print(f"Saved to: {output_path}")
    print("\nMerge completed successfully!")


if __name__ == "__main__":
    build_merged_dataset()
