"""Robustness test: what if fraudsters stop draining the account EXACTLY?

In PaySim, 97.6% of frauds move exactly the whole balance (amount == oldbalanceOrg).
The official baseline relies heavily on that simulator rule. Here every fraudster
leaves a random share u ~ U(0, level) of the balance on the account, which breaks
the exact equality, and we measure how each model holds up.

Two scenarios per noise level:
  A  attack_test_only   : model trained on clean data, frauds change behaviour in
                          production (test set only) -> how brittle is the model?
  B  retrained_on_noisy : the bank retrains on data where frauds already behave
                          this way -> what can still be learned without the rule?

Legitimate transactions are never modified, so the false-positive rate only
moves through the model, not the data.

Models compared: pre_tx (official) and pre_tx_v1 (ablation, no exact-equality flag).
Outputs: reports/robustness_noisy_amounts.csv and .json

Run (from the repo root, Docker can stay stopped, ~5 min):
  .venv\\Scripts\\python src\\16_robustness_noisy_amounts.py
"""

import json

import numpy as np
import pandas as pd

from fraud_model import (
    FEATURE_SETS, REPORT_DIR, add_features, chrono_split, evaluate, load, predict, train,
)

NOISE_LEVELS = [0.0, 0.001, 0.01, 0.05, 0.10]   # max share of the balance left behind
MODELS = ["pre_tx", "pre_tx_v1"]
SEED = 42


def perturb_frauds(df, level, seed):
    """Each fraudster leaves u ~ U(0, level) of the balance: amount = amount * (1 - u)."""
    df = df.copy()
    if level > 0:
        rng = np.random.default_rng(seed)
        fraud = df["isFraud"] == 1
        u = rng.uniform(0.0, level, int(fraud.sum()))
        df.loc[fraud, "amount"] = (df.loc[fraud, "amount"] * (1.0 - u)).round(2)
    return add_features(df)


def summarize(model_name, level, scenario, metrics):
    m = metrics["at_2fa_threshold"]
    return {
        "model": model_name,
        "noise_level": level,
        "scenario": scenario,
        "auc_roc": round(metrics["auc_roc"], 4),
        "pr_auc": round(metrics["pr_auc"], 4),
        "recall_at_0.60": round(m["recall"], 4),
        "fpr_all_tx_at_0.60": round(m["fpr_all_transactions"], 6),
        "fp_at_0.60": m["fp"],
        "missed_frauds_at_0.60": m["fn"],
    }


def main():
    print("Loading data...")
    history, _ = load("paysim_historical.csv")
    test, n_other = load("paysim_streaming.csv")
    train_clean, valid_clean = chrono_split(history)
    y = test["isFraud"].to_numpy()

    # Same noisy test set for every model (seed depends only on the level)
    noisy_tests = {lvl: perturb_frauds(test, lvl, SEED + i) for i, lvl in enumerate(NOISE_LEVELS)}
    exact = {lvl: float(t.loc[t["isFraud"] == 1, "amountEqualsBalance"].mean())
             for lvl, t in noisy_tests.items()}
    print("Share of frauds still draining the account exactly:",
          {lvl: f"{v:.1%}" for lvl, v in exact.items()})

    rows = []
    for name in MODELS:
        features = FEATURE_SETS[name]
        print(f"\n=== Model {name} ===")
        clean_model, _ = train(train_clean, valid_clean, features, verbose=False)

        for i, level in enumerate(NOISE_LEVELS):
            noisy_test = noisy_tests[level]

            # A: trained on clean data, attacked in production
            metrics = evaluate(y, predict(clean_model, noisy_test, features), n_other)
            rows.append(summarize(name, level, "A_attack_test_only", metrics))

            # B: retrained on data where frauds already leave money behind
            if level > 0:
                train_noisy = perturb_frauds(train_clean, level, SEED + 100 + i)
                valid_noisy = perturb_frauds(valid_clean, level, SEED + 200 + i)
                model_b, _ = train(train_noisy, valid_noisy, features, verbose=False)
                metrics = evaluate(y, predict(model_b, noisy_test, features), n_other)
                rows.append(summarize(name, level, "B_retrained_on_noisy", metrics))
                del train_noisy, valid_noisy

            print(f"  noise {level:>5.1%} done")

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)

    print("\n=== Recall @0.60 (spec target > 94%) ===")
    print(table.pivot_table(index="noise_level", columns=["scenario", "model"],
                            values="recall_at_0.60").map("{:.2%}".format))
    print("\n=== False positives, all transactions @0.60 (spec target < 0.1%) ===")
    print(table.pivot_table(index="noise_level", columns=["scenario", "model"],
                            values="fpr_all_tx_at_0.60").map("{:.4%}".format))
    print("\n=== AUC-ROC (spec target > 0.96) ===")
    print(table.pivot_table(index="noise_level", columns=["scenario", "model"],
                            values="auc_roc").map("{:.4f}".format))

    REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORT_DIR / "robustness_noisy_amounts.csv", index=False)
    (REPORT_DIR / "robustness_noisy_amounts.json").write_text(json.dumps(
        {"noise_levels": NOISE_LEVELS, "frauds_still_exact": exact, "results": rows}, indent=2))
    print(f"\nSaved: {REPORT_DIR / 'robustness_noisy_amounts.csv'}")


if __name__ == "__main__":
    main()
