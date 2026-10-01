from pathlib import Path
import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "application_train.csv"
)


def load_application_data():
    """Load the Home Credit application training dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


if __name__ == "__main__":
    df = load_application_data()

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head())