import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_application_data
from preprocessing import prepare_data, build_preprocessor


RANDOM_STATE = 42


# ============================================================
# MODEL TRAINING
# ============================================================

def train_logistic_regression(X_train_processed, y_train):
    """
    Train Logistic Regression baseline.
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
    Train Random Forest classifier.
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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_valid_processed, y_valid, model_name):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X_valid_processed)

    # Probability of TARGET = 1
    y_prob = model.predict_proba(X_valid_processed)[:, 1]

    print("\n" + "=" * 60)
    print(f"{model_name.upper()} RESULTS")
    print("=" * 60)

    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred)
    recall = recall_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    roc_auc = roc_auc_score(y_valid, y_prob)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("Loading dataset...")

    df = load_application_data()

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Train / validation split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = prepare_data(df)

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_valid.shape[0]}")

    # --------------------------------------------------------
    # 3. Build preprocessing pipeline
    # --------------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    # IMPORTANT:
    # Fit preprocessing ONLY on training data.
    # Then use the learned transformations on validation data.

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    print("\nPreprocessing completed.")

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_valid_processed.shape}")

    # --------------------------------------------------------
    # 4. Train Logistic Regression
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train,
    )

    print("Logistic Regression trained successfully.")

    # --------------------------------------------------------
    # 5. Train Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train,
    )

    print("Random Forest trained successfully.")

    # --------------------------------------------------------
    # 6. Evaluate Logistic Regression
    # --------------------------------------------------------

    logistic_results = evaluate_model(
        logistic_model,
        X_valid_processed,
        y_valid,
        "Logistic Regression",
    )

    # --------------------------------------------------------
    # 7. Evaluate Random Forest
    # --------------------------------------------------------

    random_forest_results = evaluate_model(
        random_forest_model,
        X_valid_processed,
        y_valid,
        "Random Forest",
    )

    # --------------------------------------------------------
    # 8. Model comparison
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Metric':<15}"
        f"{'Logistic Regression':<22}"
        f"{'Random Forest':<15}"
    )

    print("-" * 52)

    print(
        f"{'Accuracy':<15}"
        f"{logistic_results['accuracy']:<22.4f}"
        f"{random_forest_results['accuracy']:<15.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{logistic_results['precision']:<22.4f}"
        f"{random_forest_results['precision']:<15.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{logistic_results['recall']:<22.4f}"
        f"{random_forest_results['recall']:<15.4f}"
    )

    print(
        f"{'F1 Score':<15}"
        f"{logistic_results['f1']:<22.4f}"
        f"{random_forest_results['f1']:<15.4f}"
    )

    print(
        f"{'ROC-AUC':<15}"
        f"{logistic_results['roc_auc']:<22.4f}"
        f"{random_forest_results['roc_auc']:<15.4f}"
    )

    print("\nModel comparison completed!")