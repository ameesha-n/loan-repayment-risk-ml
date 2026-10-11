
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_application_data(extended=False):
    """
    Load the original or extended Home Credit training dataset.

    Args:
        extended: If True, load the dataset containing engineered
                  auxiliary features.

    Returns:
        A pandas DataFrame containing applicant features and TARGET.
    """
    if extended:
        data_path = PROCESSED_DIR / "application_train_extended.csv"
    else:
        data_path = DATA_DIR / "application_train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    if "TARGET" not in df.columns:
        raise ValueError("TARGET column is missing from the dataset.")

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("SK_ID_CURR column is missing from the dataset.")

    print(f"Loaded dataset: {data_path.name}")
    print(f"Dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    print("\n--- Original dataset ---")
    original = load_application_data()

    print("\n--- Extended dataset ---")
    extended = load_application_data(extended=True)
