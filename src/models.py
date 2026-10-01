import sys
from pathlib import Path

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


def train_logistic_regression(X_train, y_train, preprocessor):
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    X_train_processed = preprocessor.fit_transform(X_train)

    model.fit(X_train_processed, y_train)

    return model, X_train_processed


def evaluate_model(model, preprocessor, X_valid, y_valid):
    X_valid_processed = preprocessor.transform(X_valid)

    y_pred = model.predict(X_valid_processed)
    y_prob = model.predict_proba(X_valid_processed)[:, 1]

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)

    print(f"\nAccuracy : {accuracy_score(y_valid, y_pred):.4f}")
    print(f"Precision: {precision_score(y_valid, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_valid, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_valid, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_valid, y_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))

    return y_pred, y_prob


if __name__ == "__main__":
    df = load_application_data()

    X_train, X_valid, y_train, y_valid = prepare_data(df)

    preprocessor = build_preprocessor(X_train)

    print("Training Logistic Regression...")

    model, X_train_processed = train_logistic_regression(
        X_train,
        y_train,
        preprocessor,
    )

    print(f"\nProcessed training shape: {X_train_processed.shape}")

    evaluate_model(
        model,
        preprocessor,
        X_valid,
        y_valid,
    )