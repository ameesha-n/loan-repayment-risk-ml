# Loan Repayment Risk Prediction

## Project Overview

This project uses machine learning to predict the likelihood of loan applicants experiencing payment difficulties. It compares multiple classification models, investigates probability calibration, and uses SHAP to explain model predictions.

The project focuses on three key aspects of credit risk modelling:

- **Predictive performance:** Comparing classification models using appropriate evaluation metrics.
- **Probability reliability:** Investigating whether predicted probabilities correspond to observed outcomes.
- **Model interpretability:** Using SHAP to explain global model behaviour and individual predictions.

The project also explores auxiliary credit-history features and performs anomaly detection on applicant data.

## Problem Statement

Financial institutions need reliable ways to assess the risk associated with loan applications. Machine learning can identify patterns in applicant data that are associated with repayment difficulties.

However, credit risk modelling presents several challenges:

- **Class imbalance:** Only a small proportion of applicants experience payment difficulties.
- **Probability calibration:** Predicted probabilities may not accurately reflect observed outcomes.
- **Interpretability:** Understanding which features influence predictions is important when evaluating model behaviour.
- **Data quality and timing:** Historical credit information must be aligned with the information available when an application is evaluated.

This project explores these challenges through model comparison, feature engineering, calibration, explainability, and anomaly detection.

## Dataset

The project uses the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk/data).

The primary dataset is `application_train.csv`.

| Property | Value |
|---|---:|
| Number of applicants | 307,511 |
| Number of columns | 122 |
| Target variable | `TARGET` |
| No recorded payment difficulties (0) | 282,686 (91.93%) |
| Payment difficulties (1) | 24,825 (8.07%) |

The target variable indicates whether an applicant experienced payment difficulties:

- `0` — No recorded payment difficulties
- `1` — Payment difficulties recorded

The dataset is imbalanced, so accuracy alone is insufficient for evaluating model performance.

### Auxiliary datasets

The extended pipeline uses additional Home Credit tables containing historical credit and repayment information:

- `bureau.csv`
- `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

These tables are aggregated at the applicant level and merged with the main application data.

**Data availability:** The datasets are not included in this repository. Download the required CSV files from the Kaggle competition and place them in the `data/` directory.

## Project Structure

```text
loan-repayment-risk-ml/
├── data/
│   ├── application_train.csv
│   ├── application_test.csv
│   ├── bureau.csv
│   ├── bureau_balance.csv
│   ├── previous_application.csv
│   ├── POS_CASH_balance.csv
│   ├── installments_payments.csv
│   ├── credit_card_balance.csv
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── calibration.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── merge_features.py
│   ├── compare_datasets.py
│   ├── extended_analysis.py
│   ├── anomaly_detection.py
│   ├── explain_applicant.py
│   ├── predict_test.py
│   └── run_pipeline.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

Generated datasets and results are excluded from version control where configured in `.gitignore`.

## Methodology

### 1. Exploratory Data Analysis

The initial analysis examines:

- Dataset dimensions and data types
- Target distribution and class imbalance
- Missing values
- Duplicate records
- Numerical and categorical features

The initial analysis identified 307,511 applicants and 122 columns in the training dataset.

### 2. Data Preprocessing

The preprocessing pipeline includes:

- Stratified training-validation splitting
- Missing-value imputation
- Numerical feature scaling where required
- One-hot encoding of categorical variables
- Consistent transformations of training and validation data

The original application dataset is split as follows:

| Dataset | Samples |
|---|---:|
| Training | 246,008 |
| Validation | 61,503 |

The baseline preprocessing pipeline produces 245 transformed features.

### 3. Auxiliary Feature Engineering

The extended pipeline aggregates information from historical credit and repayment tables.

Examples include:

- Number and characteristics of previous credit records
- Previous application amounts and application outcomes
- POS cash balance history and delinquency indicators
- Installment payment delays and payment shortfalls
- Credit-card balances, credit limits, drawings, and delinquency indicators

The aggregated features are merged with the application dataset to investigate whether historical credit information improves predictive performance.

**Temporal limitation:** The current feature-engineering implementation aggregates historical records without explicitly enforcing a cutoff for every record based on the current application's prediction date. The temporal availability of all auxiliary records has not yet been fully verified. Consequently, temporal leakage has not been ruled out, and the reported results should be considered preliminary until this limitation is investigated.

### 4. Machine Learning Models

Three classification algorithms are compared.

**Logistic Regression**

A baseline model that provides a relatively simple approach to binary classification.

**Random Forest**

An ensemble learning algorithm that combines multiple decision trees to capture nonlinear patterns.

**LightGBM**

A gradient-boosting algorithm designed for efficient learning on structured and tabular datasets.

Both baseline models using application features and extended models using auxiliary features are evaluated.

## Model Evaluation

The models are evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases.
- **PR-AUC:** Summarises precision-recall performance and is particularly useful for imbalanced datasets.
- **Brier score:** Measures the mean squared error of predicted probabilities; lower is better.

### Baseline Results

The following results come from the saved validation evaluation in `results/dataset_comparison.csv`.

| Metric | Logistic Regression | Random Forest | LightGBM |
|---|---:|---:|---:|
| Accuracy | 0.6899 | 0.7239 | 0.7059 |
| Precision | 0.1618 | 0.1680 | 0.1695 |
| Recall | 0.6796 | 0.6123 | 0.6779 |
| F1-score | 0.2614 | 0.2637 | 0.2713 |
| ROC-AUC | 0.7487 | 0.7370 | 0.7604 |
| PR-AUC | 0.2279 | 0.2137 | 0.2504 |
| Brier score | 0.2026 | 0.1953 | 0.1911 |

### Extended Results

The extended models use the engineered auxiliary credit-history features in addition to the application features.

| Metric | Logistic Regression | Random Forest | LightGBM |
|---|---:|---:|---:|
| Accuracy | 0.7013 | 0.7533 | 0.7240 |
| Precision | 0.1685 | 0.1825 | 0.1815 |
| Recall | 0.6866 | 0.5911 | 0.6892 |
| F1-score | 0.2706 | 0.2789 | 0.2874 |
| ROC-AUC | 0.7625 | 0.7485 | 0.7783 |
| PR-AUC | 0.2424 | 0.2284 | 0.2787 |
| Brier score | 0.1969 | 0.1840 | 0.1821 |

### Results Interpretation

Extended LightGBM achieved the highest ROC-AUC, PR-AUC, recall, and F1-score among the six model configurations in the saved comparison.

Extended Random Forest achieved the highest accuracy and precision.

The extended LightGBM model is therefore the strongest candidate in this experiment based on ranking performance and the balance between precision and recall.

The additional features improved the recorded LightGBM ROC-AUC from 0.7604 to 0.7783 and PR-AUC from 0.2504 to 0.2787.

These results are based on the current validation procedure. The potential temporal leakage limitation must be investigated before drawing strong conclusions about generalisation.

## Probability Calibration

A model can rank applicants effectively without producing probabilities that accurately represent observed event frequencies.

To investigate probability reliability, sigmoid calibration (Platt scaling) was applied to the extended LightGBM model.

The saved validation calibration metrics are recorded in `results/extended_calibration_metrics.csv`.

| Metric | Result |
|---|---:|
| Validation default rate | 0.0807 |
| Raw mean predicted probability | 0.3922 |
| Calibrated mean predicted probability | 0.0804 |
| Raw Brier score | 0.1821 |
| Calibrated Brier score | 0.0663 |
| Constant-rate baseline Brier score | 0.0742 |

The calibrated mean predicted probability is close to the observed validation default rate. The recorded calibrated Brier score is also lower than the constant-rate baseline Brier score.

However, the calibration evaluation procedure needs further methodological verification before the improvement can be attributed solely to calibration. In particular, the fitting and evaluation procedures must be clearly separated to avoid overly optimistic conclusions.

Calibration improves probability estimates; it does not automatically improve classification accuracy or ranking performance.

The calibration curves are saved as:

- `results/calibration_curve.png`
- `results/extended_calibration_curve.png`

## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to explain LightGBM predictions at both global and individual levels.

### Global Feature Importance

The global SHAP summary plot ranks features according to their mean absolute SHAP values, highlighting features with the largest average influence on model output.

Features prominent in the current analysis include:

- `EXT_SOURCE_3`
- `EXT_SOURCE_2`
- `EXT_SOURCE_1`
- `AMT_GOODS_PRICE`
- `AMT_CREDIT`
- `DAYS_EMPLOYED`
- `DAYS_BIRTH`

The extended SHAP summary plot is saved at `results/extended_shap_summary.png`.

### Individual Prediction Explanation

Local SHAP explanations illustrate how individual features contribute to a selected applicant's prediction.

Positive SHAP values push the raw model output towards the positive class, while negative SHAP values push it towards the negative class.

The project saves:

- `results/extended_shap_local.png`
- `results/individual_shap_explanations.csv`

The individual explanation CSV contains explanations for a sample of 500 validation applicants. The script `src/explain_applicant.py` can be used to look up an applicant's available explanation.

SHAP explains the behaviour of the fitted model; it does not establish causal relationships between applicant characteristics and repayment outcomes. The current SHAP explanations describe the raw model output, not the calibrated probability directly.

## Anomaly Detection

The project also includes an anomaly detection stage using Isolation Forest to identify unusual applicant records based on numerical features.

The outputs include:

- `results/anomaly_detection_results.csv`
- `results/anomaly_score_distribution.png`

Anomaly detection identifies unusual feature patterns; it does not automatically identify fraud or establish that an applicant is likely to default.

## Test Applicant Predictions

The project includes `src/predict_test.py` to generate predictions for the separate `application_test.csv` dataset.

The current run processed 48,744 test applicants and generated:

`results/test_predictions.csv`

The file contains:

- `SK_ID_CURR` — applicant identifier
- `raw_probability` — raw LightGBM predicted probability
- `calibrated_probability` — sigmoid-calibrated predicted probability

The current output contains 48,744 unique applicant IDs and no missing values in these columns.

The test dataset does not contain `TARGET` labels. Therefore, accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and test-set calibration quality cannot be calculated from this dataset.

The predictions demonstrate that the pipeline can generate applicant-level risk estimates; they do not establish that every prediction is correct.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/ameesha-n/loan-repayment-risk-ml.git
cd loan-repayment-risk-ml
```

### 2. Install Dependencies

Using a Python virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 3. Add the Datasets

Download the required CSV files from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/data).

Place the training, test, and auxiliary CSV files inside the `data/` directory. The main pipeline requires the training data and auxiliary files. Generating test predictions additionally requires `application_test.csv`.

### 4. Run the Full Pipeline

From the project root, run:

```bash
python3 src/run_pipeline.py
```

This executes the following stages in order:

1. Feature engineering
2. Merging engineered features
3. Baseline versus extended model comparison
4. Extended calibration and SHAP analysis
5. Anomaly detection

The script stops if a stage fails.

### 5. Generate Test Predictions

After the required data and processed features are available, run:

```bash
python3 src/predict_test.py
```

This separately generates raw and calibrated predictions for applicants in `application_test.csv`.

### 6. Explain an Individual Applicant

For applicants included in the saved sample of SHAP explanations, run:

```bash
python3 src/explain_applicant.py
```

Follow the script's prompts or arguments, as implemented, to select an applicant.

### Generated Outputs

The main generated outputs are saved in `results/`, including:

- `dataset_comparison.csv`
- `extended_calibration_metrics.csv`
- `calibration_curve.png`
- `extended_calibration_curve.png`
- `shap_summary.png`
- `shap_local.png`
- `extended_shap_summary.png`
- `extended_shap_local.png`
- `individual_shap_explanations.csv`
- `extended_validation_predictions.csv`
- `anomaly_detection_results.csv`
- `anomaly_score_distribution.png`
- `test_predictions.csv`

Generated outputs may be excluded from version control through `.gitignore`.

## Limitations and Future Work

The current project has several limitations:

- Results are based on the current training-validation split.
- Potential temporal leakage in auxiliary feature aggregation has not been fully ruled out.
- Calibration performance and evaluation methodology require further verification.
- Test-set labels are unavailable, so test prediction accuracy cannot be measured.
- Classification thresholds should account for the costs of false positives and false negatives.
- Fairness and subgroup performance require further investigation.
- Additional hyperparameter tuning and independent testing could improve robustness.
- Real-world use would require further validation, governance, and monitoring.

Future work should prioritise verifying temporal alignment of historical features, validating calibration on independent evaluation data, and assessing model robustness before drawing stronger conclusions.

## Conclusion

This project develops a loan repayment risk prediction pipeline using Logistic Regression, Random Forest, and LightGBM.

The extended LightGBM model achieved the strongest ROC-AUC and PR-AUC among the evaluated configurations in the current validation experiment. Probability calibration investigates the reliability of predicted probabilities, while SHAP provides global feature importance and individual prediction explanations. Anomaly detection adds an additional analysis of unusual applicant records.

Together, these components demonstrate a workflow for evaluating predictive performance, probability reliability, and model interpretability in an imbalanced credit risk classification problem.

The reported results remain preliminary until the temporal alignment of auxiliary features and the calibration evaluation procedure have been fully verified.

**Disclaimer:** This is an educational machine learning project. Its predictions are not intended to be used as the sole basis for real-world lending decisions.
