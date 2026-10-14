
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

PREDICTIONS_FILE = RESULTS_DIR / "extended_validation_predictions.csv"
EXPLANATIONS_FILE = RESULTS_DIR / "individual_shap_explanations.csv"


def explain_applicant(applicant_id):
    if not PREDICTIONS_FILE.exists() or not EXPLANATIONS_FILE.exists():
        print("Required results are missing.")
        print("Run: python3 src/extended_analysis.py")
        return

    predictions = pd.read_csv(PREDICTIONS_FILE)
    explanations = pd.read_csv(EXPLANATIONS_FILE)

    prediction = predictions.loc[
        predictions["SK_ID_CURR"] == applicant_id
    ]

    explanation = explanations.loc[
        explanations["SK_ID_CURR"] == applicant_id
    ]

    if prediction.empty:
        print(f"Applicant ID {applicant_id} was not found in validation predictions.")
        return

    row = prediction.iloc[0]

    print("\n" + "=" * 60)
    print(f"LOAN REPAYMENT RISK EXPLANATION")
    print("=" * 60)
    print(f"Applicant ID: {applicant_id}")
    print(f"Raw model default probability: {row['raw_probability']:.2%}")
    print(
        "Calibrated default probability: "
        f"{row['calibrated_probability']:.2%}"
    )

    print("\nNote: SHAP values below explain the raw LightGBM model output.")
    print("They do not directly explain the calibrated probability.")

    if explanation.empty:
        print("\nNo individual SHAP explanation is available for this applicant.")
        print("The current SHAP analysis covers 500 validation applicants.")
        return

    shap_row = explanation.iloc[0]

    increasing = []
    decreasing = []

    for rank in range(1, 11):
        feature = shap_row.get(f"feature_{rank}")
        value = shap_row.get(f"shap_value_{rank}")

        if pd.isna(feature) or pd.isna(value):
            continue

        item = (str(feature), float(value))

        if value > 0:
            increasing.append(item)
        elif value < 0:
            decreasing.append(item)

    print("\nFactors pushing the raw model output HIGHER:")
    if increasing:
        for feature, value in sorted(
            increasing, key=lambda item: item[1], reverse=True
        ):
            print(f"  + {feature}: {value:.5f}")
    else:
        print("  None among the ten strongest features.")

    print("\nFactors pushing the raw model output LOWER:")
    if decreasing:
        for feature, value in sorted(
            decreasing, key=lambda item: item[1]
        ):
            print(f"  - {feature}: {value:.5f}")
    else:
        print("  None among the ten strongest features.")

    print("\nInterpretation:")
    print(
        "Positive SHAP values increase the model output for default; "
        "negative values decrease it."
    )
    print(
        "These are model explanations, not proof of causation or "
        "the actual reason a lender would reject an application."
    )


def main():
    if len(sys.argv) > 1:
        try:
            applicant_id = int(sys.argv[1])
        except ValueError:
            print("Applicant ID must be an integer.")
            sys.exit(1)
    else:
        try:
            applicant_id = int(input("Enter applicant ID (SK_ID_CURR): "))
        except ValueError:
            print("Applicant ID must be an integer.")
            sys.exit(1)

    explain_applicant(applicant_id)


if __name__ == "__main__":
    main()
