from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "claim_status"
SEED = 20260820


def resolve_data_path() -> Path:
    candidates = []
    if os.environ.get("INSURANCE_CLAIMS_CSV"):
        candidates.append(Path(os.environ["INSURANCE_CLAIMS_CSV"]))
    candidates.extend([
        ROOT / "data" / "raw" / "Insurance claims data.csv",
        Path("/Users/personal/Downloads/Insurance claims data.csv"),
    ])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Place the Kaggle CSV at data/raw/Insurance claims data.csv or set INSURANCE_CLAIMS_CSV.")


def stratified_split(y: pd.Series, test_size: float = 0.25):
    rng = np.random.default_rng(SEED)
    train, test = [], []
    for value in sorted(y.unique()):
        idx = np.flatnonzero(y.to_numpy() == value)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return np.array(train), np.array(test)


def prepare_features(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    features = df.drop(columns=[TARGET, "policy_id"]).copy()
    yes_no_cols = [col for col in features.columns if set(features[col].dropna().unique()).issubset({"Yes", "No"})]
    for col in yes_no_cols:
        features[col] = features[col].map({"Yes": 1, "No": 0}).astype(float)

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    train_raw = features.iloc[train_idx]
    test_raw = features.iloc[test_idx]
    x_train = pd.get_dummies(train_raw, columns=categorical_cols, drop_first=True, dtype=float)
    x_test = pd.get_dummies(test_raw, columns=categorical_cols, drop_first=True, dtype=float)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)

    means = x_train[numeric_cols].mean()
    stds = x_train[numeric_cols].std(ddof=0).replace(0, 1)
    x_train[numeric_cols] = (x_train[numeric_cols] - means) / stds
    x_test[numeric_cols] = (x_test[numeric_cols] - means) / stds
    return x_train.astype(float), x_test.astype(float)


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_linear_probability(x: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0) -> np.ndarray:
    x_i = add_intercept(x)
    penalty = np.eye(x_i.shape[1]) * ridge_lambda
    penalty[0, 0] = 0
    return np.linalg.solve(x_i.T @ x_i + penalty, x_i.T @ y)


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge_lambda: float = 1.0, max_iter: int = 80) -> np.ndarray:
    x_i = add_intercept(x)
    beta = np.zeros(x_i.shape[1])
    penalty = np.eye(x_i.shape[1]) * ridge_lambda
    penalty[0, 0] = 0
    for _ in range(max_iter):
        eta = np.clip(x_i @ beta, -35, 35)
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        grad = x_i.T @ (p - y) + penalty @ beta
        hess = x_i.T @ (x_i * w[:, None]) + penalty
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(name: str, y_true: np.ndarray, raw_scores: np.ndarray, threshold: float) -> dict[str, float | str]:
    prob = np.clip(raw_scores, 0, 1)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    cutoff = np.quantile(prob, 0.90)
    top_decile_lift = y_true[prob >= cutoff].mean() / y_true.mean()
    return {
        "model": name,
        "roc_auc": auc_score(y_true, raw_scores),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": float(np.mean((prob - y_true) ** 2)),
        "top_decile_lift": float(top_decile_lift),
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_path = resolve_data_path()
    df = pd.read_csv(data_path)
    train_idx, test_idx = stratified_split(df[TARGET])
    y_train = df.iloc[train_idx][TARGET].to_numpy(dtype=float)
    y_test = df.iloc[test_idx][TARGET].to_numpy(dtype=float)
    threshold = float(y_train.mean())
    x_train, x_test = prepare_features(df, train_idx, test_idx)

    simple_beta = fit_linear_probability(x_train[["subscription_length"]].to_numpy(), y_train, ridge_lambda=0)
    multi_beta = fit_linear_probability(x_train.to_numpy(), y_train, ridge_lambda=2)
    logit_beta = fit_logistic(x_train.to_numpy(), y_train, ridge_lambda=2)

    simple_scores = add_intercept(x_test[["subscription_length"]].to_numpy()) @ simple_beta
    multi_scores = add_intercept(x_test.to_numpy()) @ multi_beta
    logit_scores = 1 / (1 + np.exp(-np.clip(add_intercept(x_test.to_numpy()) @ logit_beta, -35, 35)))

    metrics = [
        evaluate("Simple linear baseline", y_test, simple_scores, threshold),
        evaluate("Multiple linear probability", y_test, multi_scores, threshold),
        evaluate("Logistic regression", y_test, logit_scores, threshold),
    ]
    write_rows(ROOT / "outputs" / "model_metrics.csv", metrics)
    print("Data:", data_path)
    print("Rows:", len(df), "Claim rate:", f"{df[TARGET].mean():.2%}")
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
