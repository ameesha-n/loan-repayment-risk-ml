# Loan Repayment Risk Prediction

## Project Overview

This project uses machine learning to predict the likelihood of loan applicants experiencing payment difficulties. It compares multiple classification models, investigates probability calibration, and uses SHAP to explain model predictions.

The project focuses on three key aspects of credit risk modelling:

* Predictive performance
* Reliability of predicted probabilities
* Model interpretability

## Problem Statement

Financial institutions need reliable ways to assess the risk associated with loan applications. Machine learning can identify patterns in applicant data that are associated with repayment difficulties.

However, this task presents several challenges:

* **Class imbalance:** Only a small proportion of applicants experience payment difficulties.
* **Probability calibration:** Predicted probabilities may not accurately reflect observed outcomes.
* **Interpretability:** Understanding which features influence predictions is important when evaluating model behaviour.

This project explores these challenges through model comparison, calibration, and explainability.

## Dataset

The project uses the **Home Credit Default Risk** dataset, specifically `application_train.csv`.

| Property                               |            Value |
| -------------------------------------- | ---------------: |
| Number of applicants                   |          307,511 |
| Number of columns                      |              122 |
| Target variable                        |         `TARGET` |
| No recorded payment difficulties (`0`) | 282,686 (91.93%) |
| Payment difficulties (`1`)             |   24,825 (8.07%) |

The target variable indicates whether an applicant experienced payment difficulties.

* `0` — No recorded payment difficulties
* `1` — Payment difficulties recorded

The dataset is imbalanced, so accuracy alone is not sufficient to evaluate model performance.

**Dataset source:** [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)

The dataset is not included in this repository. Download `application_train.csv` and place it inside the `data/` directory.

## Project Structure

```text
loan-repayment-risk-ml/
├── data/
│   └── README.md
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── calibration.py
│   └── explainability.py
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis

The initial analysis examines:

* Dataset dimensions and data types
* Target distribution and class imbalance
* Missing values
* Duplicate records
* Numerical and categorical features

The dataset contains 307,511 rows and 122 columns, with no duplicate rows identified in the initial analysis.

### 2. Data Preprocessing

The preprocessing pipeline includes:

* Stratified training-validation split
* Missing-value imputation
* Numerical feature scaling where required
* One-hot encoding of categorical variables
* Consistent transformation of training and validation data

The dataset is split as follows:

| Dataset    | Samples |
| ---------- | ------: |
| Training   | 246,008 |
| Validation |  61,503 |

After preprocessing, the feature matrix contains 245 transformed features.

### 3. Machine Learning Models

Three classification models are compared.

**Logistic Regression**

A baseline model that provides a relatively simple approach to binary classification.

**Random Forest**

An ensemble learning algorithm that combines multiple decision trees to capture nonlinear patterns.

**LightGBM**

A gradient-boosting algorithm designed for efficient learning on structured and tabular datasets.

## Model Evaluation

The models are evaluated using the following metrics:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted positive cases that are actually positive.
* **Recall:** Proportion of actual positive cases identified.
* **F1-score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Measures how well the model ranks positive cases above negative cases.
* **PR-AUC:** Summarises precision-recall performance, particularly useful for imbalanced datasets.
* **Brier score:** Measures the mean squared error of predicted probabilities; lower is better.

### Baseline Results

The following results were obtained from the current validation run.

| Metric      | Logistic Regression | Random Forest | LightGBM |
| ----------- | ------------------: | ------------: | -------: |
| Accuracy    |              0.6902 |        0.7225 |   0.7069 |
| Precision   |              0.1619 |        0.1681 |   0.1704 |
| Recall      |              0.6794 |        0.6171 |   0.6802 |
| F1-score    |              0.2615 |        0.2642 |   0.2725 |
| ROC-AUC     |              0.7487 |        0.7362 |   0.7598 |
| PR-AUC      |              0.2280 |        0.2127 |   0.2481 |
| Brier score |              0.2026 |        0.1957 |   0.1912 |

### Results Interpretation

LightGBM achieved the highest ROC-AUC, PR-AUC, recall, and F1-score among the three baseline models in the current experiment.

Random Forest achieved the highest accuracy. However, because only 8.07% of applicants belong to the positive class, accuracy alone does not adequately describe model performance.

LightGBM is therefore the strongest baseline in this experiment based on ROC-AUC and PR-AUC.

## Probability Calibration

A model may rank applicants effectively without producing probabilities that accurately represent observed event frequencies.

To investigate probability reliability, sigmoid calibration (Platt scaling) was applied to LightGBM predictions.

### Calibration Results

| Metric                 | LightGBM |
| ---------------------- | -------: |
| Raw Brier score        |   0.1912 |
| Calibrated Brier score |   0.0676 |
| Absolute reduction     |   0.1236 |

The calibration curve compares predicted probabilities with observed positive-class frequencies. A curve closer to the diagonal indicates better agreement between predicted and observed probabilities.

The lower reported Brier score suggests improved probability quality in the current experiment. This result should be verified using a calibration procedure that is separate from the final evaluation data before drawing strong conclusions.

Calibration improves probability estimates; it does not automatically improve classification accuracy or ranking performance.

## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to explain LightGBM predictions at both global and individual levels.

### Global Feature Importance

The global SHAP summary plot ranks features according to their mean absolute SHAP values, highlighting features with the largest average influence on model output.

Prominent features in the current analysis include:

* `EXT_SOURCE_3`
* `EXT_SOURCE_2`
* `EXT_SOURCE_1`
* `AMT_GOODS_PRICE`
* `AMT_CREDIT`
* `DAYS_EMPLOYED`
* `DAYS_BIRTH`

### Individual Prediction Explanation

The local SHAP plot illustrates how individual features contribute to a selected applicant's prediction.

Positive SHAP values push the model output toward the positive class, while negative SHAP values push it toward the negative class.

SHAP explains the behaviour of the fitted model; it does not establish causal relationships between applicant characteristics and repayment outcomes.

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

### 3. Add the Dataset

Download `application_train.csv` from the Kaggle competition linked above and place it inside the `data/` directory.

### 4. Run Exploratory Data Analysis

```bash
python3 src/eda.py
```

### 5. Run Preprocessing

```bash
python3 src/preprocessing.py
```

### 6. Train and Evaluate Models

```bash
python3 src/models.py
```

The current modelling pipeline includes baseline model evaluation, LightGBM probability calibration, and SHAP explainability.

Generated plots may be saved in the `results/` directory. Generated outputs are excluded from version control where configured in `.gitignore`.

## Limitations and Future Work

* Results are based on the current training-validation split.
* Calibration performance should be verified on independent evaluation data.
* Classification thresholds should account for the costs of false positives and false negatives.
* Fairness and subgroup performance require further investigation.
* Additional hyperparameter tuning and independent testing could improve robustness.
* Real-world use would require further validation, governance, and monitoring.

## Conclusion

This project develops a loan repayment risk prediction pipeline using Logistic Regression, Random Forest, and LightGBM.

LightGBM achieved the strongest baseline ROC-AUC and PR-AUC in the current experiment. Probability calibration investigates the reliability of its predicted probabilities, while SHAP provides global feature importance and individual prediction explanations.

Together, these components demonstrate a workflow for evaluating predictive performance, probability reliability, and model interpretability in an imbalanced credit risk classification problem.

**Disclaimer:** This is an educational machine learning project. Its predictions are not intended to be used as the sole basis for real-world lending decisions.
