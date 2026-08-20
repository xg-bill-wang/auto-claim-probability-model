# Data

This project uses a Kaggle auto insurance claims dataset with policy-level vehicle and customer attributes.

The raw CSV is not committed to the repository. To reproduce the analysis, place the file here:

```text
data/raw/Insurance claims data.csv
```

Or run the model with:

```bash
INSURANCE_CLAIMS_CSV="/path/to/Insurance claims data.csv" python src/claim_probability_models.py
```
