import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_application_data
from preprocessing import prepare_data, build_preprocessor
from evaluation import evaluate_model, compare_models


RANDOM_STATE = 42


def train_logistic_regression(X_train_processed, y_train):
    """
    Train Logistic Regression with class balancing.
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    model.fit(X_train_processed, y_train)

    return model


def train_random_forest(X_train_processed, y_train):
    """
    Train Random Forest with class balancing.
    """
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
    """
    Train LightGBM with class balancing.
    """
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

    # --------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------

    print("Loading dataset...")

    df = load_application_data()

    print(f"Dataset shape: {df.shape}")


    # --------------------------------------------------
    # 2. TRAIN / VALIDATION SPLIT
    # --------------------------------------------------

    X_train, X_valid, y_train, y_valid = prepare_data(df)

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")


    # --------------------------------------------------
    # 3. PREPROCESSING
    # --------------------------------------------------

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


    # --------------------------------------------------
    # 4. LOGISTIC REGRESSION
    # --------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train,
    )

    print("Logistic Regression trained successfully.")


    # --------------------------------------------------
    # 5. RANDOM FOREST
    # --------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train,
    )

    print("Random Forest trained successfully.")


    # --------------------------------------------------
    # 6. LIGHTGBM
    # --------------------------------------------------

    print("\nTraining LightGBM...")

    lightgbm_model = train_lightgbm(
        X_train_processed,
        y_train,
    )

    print("LightGBM trained successfully.")


    # --------------------------------------------------
    # 7. EVALUATE MODELS
    # --------------------------------------------------

    print("\nEvaluating models...")

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


    # --------------------------------------------------
    # 8. MODEL COMPARISON
    # --------------------------------------------------

    results = {
        "Logistic Regression": logistic_results,
        "Random Forest": random_forest_results,
        "LightGBM": lightgbm_results,
    }

    compare_models(results)


    # --------------------------------------------------
    # 9. COMPLETION MESSAGE
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION COMPLETED!")
    print("=" * 70)