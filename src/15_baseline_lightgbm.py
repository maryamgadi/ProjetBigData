"""LightGBM baseline on PaySim balances (week 2).

Train: paysim_historical.csv (chronological), test: paysim_streaming.csv.
Outputs: models/baseline_lgbm_<set>.txt and reports/baseline_metrics_<set>.json
Shared code (features, training, metrics) lives in src/fraud_model.py.

Run (from the repo root, Docker can stay stopped):
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py                        (official)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features pre_tx_v1   (ablation)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features post_tx     (leakage, comparison)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features pre_tx_graph            (fusion graphe)
  .venv\\Scripts\\python src\\15_baseline_lightgbm.py --features pre_tx_graph --embeddings data/embeddings.parquet
"""

import argparse
import json

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from fraud_model import (
    FEATURE_SETS, MODEL_DIR, REPORT_DIR, THRESHOLD_2FA, THRESHOLD_REFUSE,
    chrono_split, evaluate, load, log_dataset_tags, log_evaluation, log_lgbm_params,
    predict, setup_mlflow, train, uses_graph,
)
from graph_features import embedding_feature_names


def main():
    parser = argparse.ArgumentParser(description="LightGBM baseline")
    parser.add_argument("--features", choices=FEATURE_SETS, default="pre_tx",
                        help="pre_tx = official, pre_tx_v1 = ablation, post_tx = leakage, "
                             "*_graph = + point-in-time graph features")
    parser.add_argument("--embeddings", default=None,
                        help="parquet of GNN embeddings (account, emb_0..emb_63) to add as features")
    args = parser.parse_args()
    features = list(FEATURE_SETS[args.features])
    args.set_name = args.features
    if args.embeddings:
        features += embedding_feature_names(args.embeddings)
        args.set_name = f"{args.features}_gnn"
    print(f"Feature set: {args.set_name} -> {len(features)} features")

    print("Loading data...")
    graph = uses_graph(features)
    history, _ = load("paysim_historical.csv", graph=graph, embeddings=args.embeddings)
    test, n_other_test = load("paysim_streaming.csv", graph=graph, embeddings=args.embeddings)
    train_df, valid_df = chrono_split(history)
    print(f"Train {len(train_df)} | Valid {len(valid_df)} | Test {len(test)}")

    setup_mlflow()
    fusion = graph or bool(args.embeddings)
    prefix = "fusion" if fusion else "baseline"
    with mlflow.start_run(run_name=f"{prefix}_lgbm_{args.set_name}") as run:
        mlflow.set_tags({"model_family": "lightgbm",
                         "stage": "fusion_week3" if fusion else "baseline_week2"})
        if args.embeddings:
            mlflow.set_tag("data.embeddings", args.embeddings)
        log_dataset_tags(train_history=("paysim_historical.csv", len(history)),
                         test_stream=("paysim_streaming.csv", len(test)))
        run_and_report(args, features, train_df, valid_df, test, n_other_test)
        print(f"MLflow run: {run.info.run_id}")


def run_and_report(args, features, train_df, valid_df, test, n_other_test):
    model, params = train(train_df, valid_df, features)
    log_lgbm_params(params, features, args.set_name)
    mlflow.log_metric("best_iteration", model.best_iteration)

    # --- Evaluation on the test set (future, never seen) ---
    y = test["isFraud"].to_numpy()
    proba = predict(model, test, features)

    zones = pd.cut(proba, bins=[-np.inf, THRESHOLD_2FA, THRESHOLD_REFUSE, np.inf],
                   right=False, labels=["APPROVE", "2FA", "REFUSE"])
    zone_table = pd.crosstab(zones, y).rename(columns={0: "legit", 1: "fraud"})

    results = {
        "feature_set": args.set_name,
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
    model_path = MODEL_DIR / f"baseline_lgbm_{args.set_name}.txt"
    report_path = REPORT_DIR / f"baseline_metrics_{args.set_name}.json"
    model.save_model(str(model_path))
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\nModel: {model_path}")
    print(f"Metrics: {report_path}")

    # --- MLflow : métriques, rapport JSON et modèle versionné ---
    log_evaluation(results)
    mlflow.log_artifact(str(report_path), artifact_path="reports")
    mlflow.log_text(zone_table.to_string(), "reports/decision_zones.txt")
    mlflow.lightgbm.log_model(model, name="model")


if __name__ == "__main__":
    main()
