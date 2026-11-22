"""Benchmark de l'inférence temps réel (objectif CDC : < 20 ms par transaction).

1. Cohérence : le scorer transaction-par-transaction (src/scoring.py) donne les mêmes
   scores et décisions que le modèle évalué en batch (script 15), sur tout le jeu de test.
2. Latence : temps de scoring d'UNE transaction (features + LightGBM + décision),
   mesuré sur N transactions du jeu de test -> p50 / p95 / p99 / max.
3. Débit : transactions scorées par seconde sur un seul cœur.

Ne mesure pas le HTTP (FastAPI) ni la lecture Redis : ce sont les briques de Maryam,
à ajouter dans le test de charge Locust (semaine 4).

Run (from the repo root, Docker can stay stopped):
  .venv\\Scripts\\python src\\17_inference_benchmark.py
"""

import argparse
import json
import platform
import time

import numpy as np

from fraud_model import REPORT_DIR, load, predict
from scoring import FEATURES, FraudScorer, decide

RAW_FIELDS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]
TARGET_MS = 20.0
WARMUP = 200


def check_consistency(scorer, test):
    """Scores du scorer (1 transaction à la fois) vs prédiction batch du script 15."""
    batch = predict(scorer.model, test, FEATURES)
    raw = test[RAW_FIELDS].assign(type=test["type"].astype(str)).to_dict("records")
    single = np.array([scorer.predict_proba(tx) for tx in raw])
    max_diff = float(np.abs(single - batch).max())
    same_decisions = int(sum(decide(a) == decide(b) for a, b in zip(single, batch)))
    return {"n": len(raw), "max_abs_score_diff": max_diff,
            "same_decision": same_decisions, "all_decisions_equal": same_decisions == len(raw)}, raw


def measure_latency(scorer, raw, n):
    rng = np.random.default_rng(42)
    sample = [raw[i] for i in rng.choice(len(raw), size=n + WARMUP, replace=len(raw) < n + WARMUP)]
    for tx in sample[:WARMUP]:          # chauffe : premiers appels plus lents (caches, allocations)
        scorer.score(tx)

    timings = np.empty(n)
    for i, tx in enumerate(sample[WARMUP:]):
        start = time.perf_counter()
        scorer.score(tx)
        timings[i] = (time.perf_counter() - start) * 1000.0

    wall = timings.sum() / 1000.0
    return {
        "n": n,
        "mean_ms": float(timings.mean()),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "max_ms": float(timings.max()),
        "throughput_tx_per_s_one_core": float(n / wall),
        "target_ms": TARGET_MS,
        "p99_under_target": bool(np.percentile(timings, 99) < TARGET_MS),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark de l'inférence")
    parser.add_argument("-n", type=int, default=10_000, help="Nombre de transactions chronométrées")
    args = parser.parse_args()

    print("Loading test set...")
    test, _ = load("paysim_streaming.csv")
    scorer = FraudScorer()
    print(f"Model: {scorer.model_path.name} | {len(FEATURES)} features | {len(test)} test transactions")

    print("\n[1/2] Consistency with batch evaluation...")
    consistency, raw = check_consistency(scorer, test)
    print(f"      max |score diff| = {consistency['max_abs_score_diff']:.2e} | "
          f"same decision: {consistency['same_decision']}/{consistency['n']}")

    print(f"\n[2/2] Latency on {args.n} single-transaction calls...")
    latency = measure_latency(scorer, raw, args.n)
    print(f"      mean {latency['mean_ms']:.3f} ms | p50 {latency['p50_ms']:.3f} | "
          f"p95 {latency['p95_ms']:.3f} | p99 {latency['p99_ms']:.3f} | max {latency['max_ms']:.3f} ms")
    print(f"      throughput (1 core): {latency['throughput_tx_per_s_one_core']:,.0f} tx/s")
    verdict = "OK" if latency["p99_under_target"] else "ECHEC"
    print(f"\nRESULTAT : {verdict} - p99 {latency['p99_ms']:.3f} ms (objectif < {TARGET_MS:.0f} ms)")

    report = {
        "model": scorer.model_path.name,
        "consistency": consistency,
        "latency": latency,
        "machine": {"python": platform.python_version(), "processor": platform.processor(),
                    "system": f"{platform.system()} {platform.release()}"},
        "scope": "features + LightGBM + decision, one transaction per call; excludes HTTP and Redis",
    }
    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "inference_benchmark.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
