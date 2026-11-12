"""LightGBM baseline on PaySim balances (week 2).

Train: paysim_historical.csv (chronological), test: paysim_streaming.csv.
Outputs: models/baseline_lgbm_<set>.txt and reports/baseline_metrics_<set>.json
Shared code (features, training, metrics) lives in src/fraud_model.py.

Run (from the repo root, Docker can stay stopped):
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py                        (official)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features pre_tx_v1   (ablation)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features post_tx     (leakage, comparison)
"""

import argparse
import json

import numpy as np
import pandas as pd

from fraud_model import (
    FEATURE_SETS, MODEL_DIR, REPORT_DIR, THRESHOLD_2FA, THRESHOLD_REFUSE,
    chrono_split, evaluate, load, predict, train,
)


def main():
    parser = argparse.ArgumentParser(description="LightGBM baseline")
    parser.add_argument("--features", choices=FEATURE_SETS, default="pre_tx",
                        help="pre_tx = official, pre_tx_v1 = ablation, post_tx = leakage")
    args = parser.parse_args()
    features = FEATURE_SETS[args.features]
    print(f"Feature set: {args.features} -> {features}")

    print("Loading data...")
    history, _ = load("paysim_historical.csv")
    test, n_other_test = load("paysim_streaming.csv")
    train_df, valid_df = chrono_split(history)
    print(f"Train {len(train_df)} | Valid {len(valid_df)} | Test {len(test)}")

    model, params = train(train_df, valid_df, features)

    # --- Evaluation on the test set (future, never seen) ---
    y = test["isFraud"].to_numpy()
    proba = predict(model, test, features)

    zones = pd.cut(proba, bins=[-np.inf, THRESHOLD_2FA, THRESHOLD_REFUSE, np.inf],
                   right=False, labels=["APPROVE", "2FA", "REFUSE"])
    zone_table = pd.crosstab(zones, y).rename(columns={0: "legit", 1: "fraud"})

    results = {
        "feature_set": args.features,
        "features": features,
        **evaluate(y, proba, n_other_test),
        "best_iteration": model.best_iteration,
        "scale_pos_weight": params["scale_pos_weight"],
        "decision_zones": {k: {str(z): int(v) for z, v in col.items()}
                           for k, col in zone_table.to_dict().items()},
    }

    print("\n=== TEST RESULTS ===")
    print(f"AUC-ROC : {results['auc_roc']:.4f}   (spec target > 0.96)")
    print(f"PR-AUC  : {results['pr_auc']:.4f}")
    for key in ("at_2fa_threshold", "at_refuse_threshold"):
        m = results[key]
        print(f"Threshold {m['threshold']}: recall {m['recall']:.2%} (target > 94%) | "
              f"precision {m['precision']:.2%} | FPR all tx {m['fpr_all_transactions']:.4%} (target < 0.1%)")
    print("\nDecision zones (rows) x truth (columns):")
    print(zone_table)

    importance = pd.Series(model.feature_importance("gain"), index=features).sort_values(ascending=False)
    print("\nFeature importance (gain):")
    print(importance.round(0))
    results["feature_importance_gain"] = {k: float(v) for k, v in importance.items()}

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"baseline_lgbm_{args.features}.txt"
    report_path = REPORT_DIR / f"baseline_metrics_{args.features}.json"
    model.save_model(str(model_path))
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nModel: {model_path}")
    print(f"Metrics: {report_path}")


if __name__ == "__main__":
    main()
