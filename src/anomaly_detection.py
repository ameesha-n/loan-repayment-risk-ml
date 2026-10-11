
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_loader import load_application_data

RESULTS_DIR = ROOT_DIR / "results"
RANDOM_STATE = 42


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading extended dataset...")
    df = load_application_data(extended=True)

    applicant_ids = df["SK_ID_CURR"].copy()
    target = df["TARGET"].copy()

    # Use numeric features only for the first version.
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    X = X.select_dtypes(include=["number"])

    # Remove columns that contain no usable information.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    print(f"Applicants: {len(X):,}")
    print(f"Numeric features: {X.shape[1]:,}")

    # Impute missing values and scale features.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    print("\nFitting Isolation Forest...")
    detector = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    predictions = detector.fit_predict(X_scaled)
    anomaly_scores = -detector.score_samples(X_scaled)

    results = pd.DataFrame({
        "SK_ID_CURR": applicant_ids.to_numpy(),
        "TARGET": target.to_numpy(),
        "anomaly_label": np.where(predictions == -1, "Anomaly", "Normal"),
        "anomaly_score": anomaly_scores,
    })

    results = results.sort_values(
        "anomaly_score",
        ascending=False,
    )

    output_path = RESULTS_DIR / "anomaly_detection_results.csv"
    results.to_csv(output_path, index=False)

    anomaly_mask = predictions == -1
    anomaly_count = int(anomaly_mask.sum())

    print("\n" + "=" * 55)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 55)
    print(f"Applicants analysed: {len(results):,}")
    print(f"Anomalies flagged: {anomaly_count:,}")
    print(f"Anomaly rate: {anomaly_mask.mean():.2%}")
    print(f"Default rate among all applicants: {target.mean():.2%}")

    if anomaly_count:
        anomaly_default_rate = target.to_numpy()[anomaly_mask].mean()
        normal_default_rate = target.to_numpy()[~anomaly_mask].mean()

        print(f"Default rate among flagged anomalies: {anomaly_default_rate:.2%}")
        print(f"Default rate among normal applicants: {normal_default_rate:.2%}")

    # Plot anomaly score distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(anomaly_scores, bins=50)
    ax.set_title("Isolation Forest Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Number of applicants")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    plot_path = RESULTS_DIR / "anomaly_score_distribution.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nResults saved: {output_path}")
    print(f"Plot saved: {plot_path}")
    print("\nAnomaly detection completed.")


if __name__ == "__main__":
    main()
