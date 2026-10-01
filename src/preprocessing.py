import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_loader import load_application_data


RANDOM_STATE = 42


def prepare_data(df):
    """
    Separate features and target and create
    stratified training and validation sets.
    """

    # TARGET is what we want the model to predict.
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # Keep the same proportion of TARGET=0 and TARGET=1
    # in both training and validation datasets.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_valid, y_train, y_valid


def build_preprocessor(X_train):
    """
    Build separate preprocessing pipelines for
    numerical and categorical features.
    """

    # Find numerical columns.
    numerical_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Find categorical columns.
    categorical_features = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    # Numerical preprocessing:
    # 1. Replace missing values with the median.
    # 2. Standardize the numerical features.
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Categorical preprocessing:
    # 1. Replace missing values with the most common category.
    # 2. Convert categories into one-hot encoded columns.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    # Apply the appropriate pipeline to each type of feature.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    return preprocessor


if __name__ == "__main__":

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = load_application_data()

    # --------------------------------------------------
    # 2. Split data
    # --------------------------------------------------

    X_train, X_valid, y_train, y_valid = prepare_data(df)

    # --------------------------------------------------
    # 3. Create preprocessing pipeline
    # --------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    print("Preprocessing pipeline created successfully!")

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")

    print(f"\nTraining positive class: {y_train.sum()}")
    print(f"Validation positive class: {y_valid.sum()}")

    print("\nTraining positive percentage:")
    print(f"{y_train.mean() * 100:.2f}%")

    print("\nValidation positive percentage:")
    print(f"{y_valid.mean() * 100:.2f}%")

    # --------------------------------------------------
    # 4. Fit preprocessing ONLY on training data
    # --------------------------------------------------

    X_train_processed = preprocessor.fit_transform(X_train)

    # --------------------------------------------------
    # 5. Transform validation data using the
    #    preprocessing learned from training data
    # --------------------------------------------------

    X_valid_processed = preprocessor.transform(X_valid)

    # --------------------------------------------------
    # 6. Display transformed data information
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("TRANSFORMED DATA")
    print("=" * 60)

    print(f"Training shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_valid_processed.shape}")

    print("\nPreprocessing completed successfully!")