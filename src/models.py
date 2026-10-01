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

from lightgbm import LGBMClassifier

sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import load_application_data
from preprocessing import prepare_data, build_preprocessor


RANDOM_STATE = 42


# ============================================================
# MODEL TRAINING
# ============================================================

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


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_valid_processed, y_valid, model_name):

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
    # 3. Preprocessing
    # --------------------------------------------------------

    print("\nBuilding preprocessing pipeline...")

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_valid_processed = preprocessor.transform(X_valid)

    print("\nPreprocessing completed.")

    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed validation shape: {X_valid_processed.shape}")

    # --------------------------------------------------------
    # 4. Logistic Regression
    # --------------------------------------------------------

    print("\nTraining Logistic Regression...")

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train,
    )

    print("Logistic Regression trained successfully.")

    # --------------------------------------------------------
    # 5. Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    random_forest_model = train_random_forest(
        X_train_processed,
        y_train,
    )

    print("Random Forest trained successfully.")

    # --------------------------------------------------------
    # 6. LightGBM
    # --------------------------------------------------------

    print("\nTraining LightGBM...")

    lightgbm_model = train_lightgbm(
        X_train_processed,
        y_train,
    )

    print("LightGBM trained successfully.")

    # --------------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 8. Model Comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Metric':<15}"
        f"{'Logistic Regression':<22}"
        f"{'Random Forest':<18}"
        f"{'LightGBM':<15}"
    )

    print("-" * 70)

    print(
        f"{'Accuracy':<15}"
        f"{logistic_results['accuracy']:<22.4f}"
        f"{random_forest_results['accuracy']:<18.4f}"
        f"{lightgbm_results['accuracy']:<15.4f}"
    )

    print(
        f"{'Precision':<15}"
        f"{logistic_results['precision']:<22.4f}"
        f"{random_forest_results['precision']:<18.4f}"
        f"{lightgbm_results['precision']:<15.4f}"
    )

    print(
        f"{'Recall':<15}"
        f"{logistic_results['recall']:<22.4f}"
        f"{random_forest_results['recall']:<18.4f}"
        f"{lightgbm_results['recall']:<15.4f}"
    )

    print(
        f"{'F1 Score':<15}"
        f"{logistic_results['f1']:<22.4f}"
        f"{random_forest_results['f1']:<18.4f}"
        f"{lightgbm_results['f1']:<15.4f}"
    )

    print(
        f"{'ROC-AUC':<15}"
        f"{logistic_results['roc_auc']:<22.4f}"
        f"{random_forest_results['roc_auc']:<18.4f}"
        f"{lightgbm_results['roc_auc']:<15.4f}"
    )

    print("\nModel comparison completed!")
