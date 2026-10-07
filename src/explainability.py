import os

import matplotlib.pyplot as plt
import shap


def compute_shap_values(model, X_data, sample_size=500):
    """
    Compute SHAP values for a sample of the data.

    We use a sample instead of the entire validation set
    to keep SHAP computation efficient.
    """

    # Limit the number of samples used for explanation
    if X_data.shape[0] > sample_size:
        X_sample = X_data[:sample_size]
    else:
        X_sample = X_data

    print(f"Computing SHAP values for {X_sample.shape[0]} samples...")

    # TreeExplainer is optimized for tree-based models such as LightGBM
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer(X_sample)

    return explainer, shap_values, X_sample


def plot_global_importance(
    shap_values,
    feature_names,
    output_path="results/shap_summary.png",
):
    """
    Generate a global SHAP feature-importance plot.

    Features with larger mean absolute SHAP values
    have a greater overall influence on predictions.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Attach readable feature names
    shap_values.feature_names = feature_names

    plt.figure(figsize=(10, 8))

    shap.plots.bar(
        shap_values,
        max_display=15,
        show=False,
    )

    plt.title("Top 15 Features - SHAP Importance")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Global SHAP plot saved to: {output_path}")


def plot_local_explanation(
    shap_values,
    sample_index=0,
    output_path="results/shap_local.png",
):
    """
    Generate a SHAP explanation for one individual applicant.

    Positive SHAP values push the prediction toward default.
    Negative SHAP values push the prediction away from default.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    single_prediction = shap_values[sample_index]

    plt.figure(figsize=(10, 8))

    shap.plots.bar(
        single_prediction,
        max_display=10,
        show=False,
    )

    plt.title(
        f"SHAP Explanation for Applicant {sample_index}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Local SHAP plot saved to: {output_path}")