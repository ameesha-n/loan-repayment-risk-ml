# Model Interpretation and Results

## 1. Overview

This document discusses the results of three machine learning models developed for the Loan Repayment Risk Prediction project: Logistic Regression, Random Forest, and LightGBM.

The models are evaluated on a validation dataset to compare predictive performance. Probability calibration and SHAP explainability are also explored.

## 2. Model Comparison

The current validation results are:

| Metric      | Logistic Regression | Random Forest | LightGBM |
| ----------- | ------------------: | ------------: | -------: |
| Accuracy    |              0.6902 |        0.7225 |   0.7069 |
| Precision   |              0.1619 |        0.1681 |   0.1704 |
| Recall      |              0.6794 |        0.6171 |   0.6802 |
| F1-score    |              0.2615 |        0.2642 |   0.2725 |
| ROC-AUC     |              0.7487 |        0.7362 |   0.7598 |
| PR-AUC      |              0.2280 |        0.2127 |   0.2481 |
| Brier score |              0.2026 |        0.1957 |   0.1912 |

LightGBM achieved the highest ROC-AUC, PR-AUC, recall, and F1-score among the three models in this experiment. Random Forest achieved the highest accuracy.

Because only approximately 8.07% of applicants belong to the positive class, accuracy alone is not sufficient to compare model performance.

## 3. Probability Calibration

Sigmoid calibration (Platt scaling) was applied to LightGBM predictions to investigate the reliability of predicted probabilities.

| Metric              | Brier score |
| ------------------- | ----------: |
| Raw LightGBM        |      0.1912 |
| Calibrated LightGBM |      0.0676 |

The reported Brier score decreased by 0.1236 after calibration. A lower Brier score indicates better probability predictions on the evaluated data.

However, this improvement must be interpreted cautiously until it is confirmed that the calibration mapping was fitted independently of the data used for the final evaluation.

Calibration changes predicted probabilities and does not necessarily improve classification accuracy or ranking performance.

## 4. SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to examine model predictions.

### Global Explanation

The global summary plot ranks features according to their mean absolute SHAP values.

Prominent features in the current analysis include:

* EXT_SOURCE_3
* EXT_SOURCE_2
* EXT_SOURCE_1
* AMT_GOODS_PRICE
* AMT_CREDIT
* DAYS_EMPLOYED
* DAYS_BIRTH

These features have relatively large average contributions to the model output in the current analysis.

### Local Explanation

The local explanation plot shows how individual features contribute to a selected applicant's prediction.

Positive SHAP values push the model output toward the positive class, while negative SHAP values push it toward the negative class.

These explanations describe the fitted model's behaviour and should not be interpreted as evidence of causation.

## 5. Limitations

* The reported results come from the current validation split.
* Calibration results require verification using an independent evaluation procedure.
* Precision and recall reflect a trade-off in identifying applicants with payment difficulties.
* Feature importance does not establish causal relationships.
* Further testing is needed before applying these predictions to real-world lending decisions.

## 6. Conclusion

LightGBM produced the strongest baseline ranking performance in the current experiment. Calibration provides a way to investigate probability reliability, while SHAP helps explain global feature importance and individual predictions.

Together, these methods provide a foundation for evaluating performance, probability reliability, and interpretability in an imbalanced credit risk prediction task.
