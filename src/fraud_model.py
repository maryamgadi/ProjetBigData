"""Shared code for the LightGBM baseline (15) and the robustness test (16):
data loading, features, training and metrics."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

DTYPES = {
    "step": "int16",
    "type": "category",
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
}
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Feature sets:
#  - pre_tx    : only what is known BEFORE the bank validates the transaction
#                (the spec scores transactions before validation) -> official baseline
#  - pre_tx_v1 : first version, with the coarse flag amount >= balance (0.17% false
#                positives) -> kept as an ablation, closer to a model without the
#                PaySim "exact draining" pattern
#  - post_tx   : also uses the balances AFTER the transaction. In PaySim they give
#                the answer away (leakage), kept only for comparison in the report.
FEATURE_SETS = {
    "pre_tx": [
        "type", "hour", "amount", "oldbalanceOrg", "oldbalanceDest",
        "amountToBalanceOrig", "destEmptyBefore",
        # Exact equality vs. overdraft: a fraudster drains the account EXACTLY
        # (97.6% of frauds), while legitimate customers sometimes ask for more
        # than their balance. Mixing both (amount >= balance) caused the false positives.
        "amountEqualsBalance", "amountExceedsBalance", "amountIsCap", "remainingOrig",
    ],
    "pre_tx_v1": [
        "type", "hour", "amount", "oldbalanceOrg", "oldbalanceDest",
        "amountToBalanceOrig", "emptiesOrig", "destEmptyBefore",
    ],
    "post_tx": [
        "type", "hour", "amount",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "errorBalanceOrig", "errorBalanceDest", "origEmptied", "destZeroBefore",
    ],
}

# Decision thresholds from the spec
THRESHOLD_2FA = 0.60
THRESHOLD_REFUSE = 0.85


def load_raw(filename):
    """Reads a CSV and keeps TRANSFER/CASH_OUT. Returns (df, number of other rows)."""
    df = pd.read_csv(DATA_DIR / filename, usecols=list(DTYPES), dtype=DTYPES)
    n_other = int((~df["type"].isin(FRAUD_TYPES)).sum())   # approved automatically
    df = df[df["type"].isin(FRAUD_TYPES)].copy()
    # Same categories in train and test, otherwise LightGBM encodes them differently
    df["type"] = pd.Categorical(df["type"], categories=FRAUD_TYPES)
    return df, n_other


def add_features(df):
    """Computes every feature from the raw columns (call again after changing amount)."""
    df["hour"] = (df["step"] % 24).astype("int8")
    # --- pre_tx features (known before validation) ---
    df["amountToBalanceOrig"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["destEmptyBefore"] = (df["oldbalanceDest"] == 0).astype("int8")
    df["amountEqualsBalance"] = ((df["amount"] - df["oldbalanceOrg"]).abs() < 0.01).astype("int8")
    df["amountExceedsBalance"] = (df["amount"] > df["oldbalanceOrg"] + 0.01).astype("int8")
    df["amountIsCap"] = (df["amount"] == 1e7).astype("int8")   # PaySim caps a transfer at 10M
    df["remainingOrig"] = df["oldbalanceOrg"] - df["amount"]
    df["emptiesOrig"] = (df["amount"] >= df["oldbalanceOrg"]).astype("int8")   # v1 only
    # --- post_tx features (use the balances after the transaction) ---
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["origEmptied"] = (df["newbalanceOrig"] == 0).astype("int8")
    df["destZeroBefore"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype("int8")
    return df


def load(filename):
    df, n_other = load_raw(filename)
    return add_features(df), n_other


def chrono_split(history, quantile=0.85):
    """Train = first 85% of the steps, validation = the last 15% (no future leakage)."""
    cut = history["step"].quantile(quantile)
    return history[history["step"] <= cut], history[history["step"] > cut]


def train(train_df, valid_df, features, verbose=True):
    pos = int(train_df["isFraud"].sum())
    neg = len(train_df) - pos
    params = {
        "objective": "binary",
        # Early stopping on logloss: AUC saturates near 1.0 and stops too early
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        # sqrt of the ratio: the full ratio (~700) would push every probability up
        # and make the 0.60 / 0.85 thresholds from the spec meaningless
        "scale_pos_weight": float(np.sqrt(neg / pos)),
        "num_threads": 4,       # leave CPU free for the rest of the PC
        "seed": 42,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(train_df[features], train_df["isFraud"])
    dvalid = lgb.Dataset(valid_df[features], valid_df["isFraud"], reference=dtrain)
    callbacks = [lgb.early_stopping(50, first_metric_only=True, verbose=verbose)]
    if verbose:
        callbacks.append(lgb.log_evaluation(50))
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dvalid], callbacks=callbacks)
    return model, params


def predict(model, df, features):
    return model.predict(df[features], num_iteration=model.best_iteration)


def metrics_at(y, proba, threshold, n_other):
    """Confusion matrix at a threshold. Other types count as true negatives."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "fpr_transfer_cashout": float(fp / (fp + tn)) if fp + tn else 0.0,
        "fpr_all_transactions": float(fp / (fp + tn + n_other)),
    }


def evaluate(y, proba, n_other):
    return {
        "auc_roc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "at_2fa_threshold": metrics_at(y, proba, THRESHOLD_2FA, n_other),
        "at_refuse_threshold": metrics_at(y, proba, THRESHOLD_REFUSE, n_other),
    }
