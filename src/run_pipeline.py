
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

STEPS = [
    ("Feature engineering", "feature_engineering.py"),
    ("Merge engineered features", "merge_features.py"),
    ("Baseline vs extended comparison", "compare_datasets.py"),
    ("Extended calibration and SHAP analysis", "extended_analysis.py"),
    ("Anomaly detection", "anomaly_detection.py"),
]


def main():
    print("=" * 65)
    print("LOAN REPAYMENT RISK ML - FULL PIPELINE")
    print("=" * 65)

    for step_name, script_name in STEPS:
        script_path = SRC_DIR / script_name

        print(f"\n\n{'=' * 65}")
        print(f"STEP: {step_name}")
        print(f"RUNNING: {script_name}")
        print("=" * 65, flush=True)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT_DIR,
            check=False,
        )

        if result.returncode != 0:
            print(
                f"\nPipeline stopped: {step_name} failed "
                f"with exit code {result.returncode}."
            )
            sys.exit(result.returncode)

    print("\n" + "=" * 65)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print("\nCheck the results/ directory for metrics, plots, and CSV outputs.")


if __name__ == "__main__":
    main()
