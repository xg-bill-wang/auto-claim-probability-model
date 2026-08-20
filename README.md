# Auto Insurance Claim Probability Modeling

This project models whether an auto insurance policy is likely to generate a claim. It starts with a simple linear regression baseline, expands to a multivariate linear probability model, and then moves to logistic regression, which is a more appropriate probability model for binary claim occurrence. The project includes both Python and R implementations; the R version uses `lm()` and `glm(..., family = binomial)` for a more actuarial-style GLM workflow.

## Business Question

Can policy, driver, region, and vehicle attributes help identify policies with above-average claim probability?

The outcome is `claim_status`, where `1` indicates that a claim occurred and `0` indicates no claim. The portfolio base claim rate in the provided dataset is approximately 6.4%, so model evaluation focuses on probability ranking, calibration, and lift instead of accuracy alone.

## Why This Is Actuarial

Claim occurrence modeling is a core part of P&C insurance pricing and portfolio monitoring. This project frames the task as a frequency model:

- estimate claim probability at the policy level;
- compare a simple regression baseline with a more suitable GLM-style model;
- evaluate whether the model separates high-risk and low-risk policies;
- interpret results using metrics that matter for insurance, including ROC-AUC, Brier score, calibration, and top-decile lift.

## Model Progression

1. **Simple linear regression baseline**
   Uses `subscription_length` only to predict the claim indicator. This is intentionally simple and easy to explain, but it can predict values outside the `[0, 1]` probability range.

2. **Multiple linear probability model**
   Adds driver, region, and vehicle features using one-hot encoding for categorical variables. This tests whether broader underwriting variables improve segmentation.

3. **Logistic regression**
   Uses the same feature set as the multivariate linear model, but constrains predicted probabilities to `[0, 1]`. This is closer to the binomial GLM framework commonly used in insurance modeling.

## Data

The raw Kaggle CSV is not committed to this repository. Place it at:

```text
data/raw/Insurance claims data.csv
```

Or provide the file path when running:

```bash
INSURANCE_CLAIMS_CSV="/path/to/Insurance claims data.csv" python src/claim_probability_models.py
INSURANCE_CLAIMS_CSV="/path/to/Insurance claims data.csv" Rscript R/claim_probability_models.R
```

The dataset used locally contains 58,592 policies, 41 columns, and no missing values in the supplied file.

## Results

Public report:

- `docs/Auto_Insurance_Claim_Probability_Model_Report.pdf`
- `docs/Auto_Insurance_Claim_Probability_Model_Report.tex`

Generated outputs are saved in:

- `outputs/model_metrics.csv`
- `outputs/calibration_by_decile.csv`
- `outputs/top_logistic_coefficients.csv`
- `outputs/data_summary.csv`
- `outputs/r_model_metrics.csv`
- `outputs/r_calibration_by_decile.csv`
- `outputs/r_top_logistic_coefficients.csv`
- `outputs/r_data_summary.csv`

Figures:

![Model performance](figures/model_performance.svg)

![Logistic calibration](figures/logistic_calibration.svg)

The R implementation also produces:

- `figures/r_model_performance.svg`
- `figures/r_logistic_calibration.svg`

Current holdout-set summary:

| Model | ROC-AUC | Precision | Recall | Brier Score | Top-Decile Lift |
|---|---:|---:|---:|---:|---:|
| Simple linear baseline | 0.592 | 0.083 | 0.622 | 0.0595 | 1.27x |
| Multiple linear probability | 0.602 | 0.082 | 0.634 | 0.0595 | 1.54x |
| Logistic regression | 0.603 | 0.086 | 0.589 | 0.0595 | 1.55x |

The logistic model produces valid probabilities between 0 and 1, while the multiple linear probability model produces a small share of impossible values outside that range. The main improvement is not raw accuracy; it is better risk ranking in the highest predicted-risk group.

R implementation holdout summary:

| Model | ROC-AUC | Precision | Recall | Brier Score | Top-Decile Lift |
|---|---:|---:|---:|---:|---:|
| Simple linear baseline | 0.592 | 0.082 | 0.601 | 0.0595 | 1.42x |
| Multiple linear probability | 0.606 | 0.082 | 0.639 | 0.0594 | 1.63x |
| Logistic regression | 0.606 | 0.085 | 0.587 | 0.0594 | 1.68x |

## Interpretation

Accuracy can be misleading because most policies do not have claims. A model that predicts "no claim" for almost everyone can appear accurate while failing to identify claim-prone policies. This project therefore compares models using:

- **ROC-AUC**: how well the model ranks claim vs. non-claim policies;
- **Brier score**: probability forecast error;
- **Recall and precision**: claim identification tradeoff at the portfolio base-rate threshold;
- **Top-decile lift**: whether the highest predicted-risk policies have a higher observed claim rate.

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/claim_probability_models.py
Rscript R/claim_probability_models.R
```

The script is deterministic and uses a stratified train/test split with a fixed random seed.

## Interview Framing

This project can be described as:

> I built a policy-level auto insurance claim probability model using a Kaggle dataset. I started with a simple linear regression baseline, then moved to a logistic regression framework to better model binary claim occurrence. I evaluated the models using ROC-AUC, Brier score, calibration by risk decile, and top-decile lift because claim data is imbalanced and accuracy alone can be misleading.

For an R/actuarial version:

> I implemented the same workflow in R using `lm()` for the linear probability baselines and `glm(..., family = binomial)` for the claim occurrence model, connecting the project to the GLM framework commonly used in actuarial pricing.

## Next Steps

- Add regularized logistic regression with cross-validation.
- Compare against a Poisson or binomial GLM workflow if using actuarial software.
- Add credibility-style segmentation by region, vehicle age, and subscription length.
- Extend the project from claim occurrence to claim frequency and severity if claim counts or losses are available.
