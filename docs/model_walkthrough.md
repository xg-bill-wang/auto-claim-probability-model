# Model Walkthrough

## Objective

Estimate the probability that an auto insurance policy generates a claim.

This is an actuarial frequency-style problem. The target variable is binary:

```text
claim_status = 1 if a claim occurred
claim_status = 0 otherwise
```

The dataset is imbalanced: only about 6.4% of policies have claims. Because of that, accuracy alone is not a strong evaluation metric.

## Step 1: Simple Linear Regression Baseline

The first model uses one explanatory variable:

```text
claim_status = beta_0 + beta_1 * subscription_length + error
```

This is useful as a teaching baseline because it is transparent and easy to explain. It shows whether one intuitive policy variable has any relationship with claim occurrence.

Limitations:

- the target is binary, not continuous;
- predictions are interpreted as probabilities but may fall below 0 or above 1;
- one variable cannot capture underwriting and vehicle differences.

## Step 2: Multiple Linear Probability Model

The second model keeps the linear regression structure but adds more policy, customer, region, and vehicle variables:

```text
claim_status = beta_0 + beta_1*x_1 + ... + beta_k*x_k + error
```

Categorical fields such as region, segment, model, fuel type, engine type, and transmission type are one-hot encoded. This model tests whether a broader feature set improves risk segmentation.

This model improves ranking modestly, but it still has the same probability limitation: linear predictions are not naturally constrained to the 0 to 1 range.

## Step 3: Logistic Regression

The third model uses the same feature set but changes the link function:

```text
logit(p) = log(p / (1 - p)) = beta_0 + beta_1*x_1 + ... + beta_k*x_k
```

Where:

```text
p = probability that a policy has a claim
```

This is more appropriate for claim occurrence because predicted values are valid probabilities. It also connects naturally to the GLM framework used in insurance pricing.

## Evaluation Metrics

The threshold is set near the training portfolio base claim rate, not 50%, because the true claim rate is much lower than 50%.

- **ROC-AUC** measures ranking quality.
- **Brier score** measures probability forecast error.
- **Precision** measures how many flagged policies actually had claims.
- **Recall** measures how many claim policies were captured.
- **Top-decile lift** compares the claim rate in the highest predicted-risk decile with the overall claim rate.

## Main Finding

The logistic regression model has the strongest ranking result among the tested models, with a holdout ROC-AUC of about 0.603 and a top-decile lift of about 1.55x. This means the highest predicted-risk policies have roughly 55% higher observed claim frequency than the overall test portfolio.

The result is realistic rather than dramatic: the available variables provide some segmentation power, but not enough to create a highly predictive claim model. In an actuarial setting, this would motivate adding stronger risk variables, exposure details, historical claim records, coverage information, and claim severity data.

## Resume Bullet Draft

- Built an auto insurance claim probability model in Python using 58K policy records, comparing simple linear regression, multivariate linear probability modeling, and logistic regression for binary claim occurrence.
- Evaluated model performance with ROC-AUC, Brier score, calibration by risk decile, and top-decile lift, identifying a high-risk segment with approximately 1.55x portfolio claim frequency.
