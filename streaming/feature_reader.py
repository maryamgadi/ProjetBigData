"""
Lecture et benchmark du Feature Store Redis.

    python streaming/feature_reader.py --top 10          # comptes les plus actifs (5 min)
    python streaming/feature_reader.py C1231006815       # features d'un compte
    python streaming/feature_reader.py C1231006815 --bench 1000   # latence (objectif < 2 ms)
    python streaming/feature_reader.py --bench 1000               # latence sur le compte le plus actif
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redis_sink as rs  # noqa: E402


def top_accounts(client, n: int, label: str = "5m", max_keys: int = 20000) -> list:
    prefix = rs.window_prefix(label)
    scores = []
    for i, key in enumerate(client.scan_iter(match=f"{prefix}*", count=1000)):
        if i >= max_keys:
            break
        count = client.hget(key, "tx_count")
        if count is not None:
            scores.append((float(count), key[len(prefix):]))
    return sorted(scores, reverse=True)[:n]


def benchmark(client, account: str, iterations: int) -> dict:
    for _ in range(min(50, iterations)):  # échauffement
        rs.get_features(client, account)
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        rs.get_features(client, account)
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    return {
        "iterations": iterations,
        "p50_ms": round(statistics.median(latencies), 3),
        "p95_ms": round(latencies[int(0.95 * (iterations - 1))], 3),
        "p99_ms": round(latencies[int(0.99 * (iterations - 1))], 3),
        "max_ms": round(latencies[-1], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lecture du Feature Store Redis")
    parser.add_argument("account", nargs="?", help="Identifiant nameOrig (ex: C1231006815)")
    parser.add_argument("--top", type=int, help="Affiche les N comptes les plus actifs")
    parser.add_argument("--window", default="5m", choices=list(rs.WINDOWS))
    parser.add_argument("--bench", type=int, help="Nombre de lectures pour mesurer la latence")
    args = parser.parse_args()

    client = rs.get_client()
    client.ping()

    if args.top:
        print(f"Top {args.top} comptes (tx_count fenêtre {args.window}) :")
        for count, account in top_accounts(client, args.top, args.window):
            print(f"  {account:<15} {int(count)} tx")

    if args.bench and not args.account:
        best = top_accounts(client, 1, args.window)
        if not best:
            print("Aucun compte dans Redis : lancez le job Spark puis le producteur.")
            return
        args.account = best[0][1]
        print(f"Compte choisi automatiquement : {args.account}")

    if args.account:
        print(json.dumps(rs.get_features(client, args.account), indent=2, ensure_ascii=False))
        if args.bench:
            result = benchmark(client, args.account, args.bench)
            status = "OK" if result["p99_ms"] < 2 else "AU-DESSUS DE 2 ms"
            print(f"Latence get_features : {result}  -> {status}")

    if not args.top and not args.account:
        parser.print_help()


if __name__ == "__main__":
    main()

