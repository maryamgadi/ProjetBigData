# Mission 1 — Spark Streaming → Redis Feature Store

```
producer.py ──► Kafka (financial-transactions) ──► Spark Structured Streaming ──► Redis
                                                    • fenêtres 5 min / 1 h / 24 h
                                                    • dernière transaction
```

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `spark_streaming_redis.py` | Job Spark : lecture Kafka, agrégations fenêtrées, écriture Redis |
| `redis_sink.py` | Schéma des clés, écriture par pipeline, `get_features()` (réutilisable par l'API) |
| `feature_reader.py` | Lecture d'un compte, top comptes actifs, benchmark de latence |
| `../tests/test_redis_sink.py` | Tests unitaires (sans Redis ni Spark) |

## Clés Redis

| Clé | Contenu | TTL |
| --- | --- | --- |
| `account:{nameOrig}` | `last_type, last_amount, last_step, last_dest, last_balance_error_orig, last_event_time` | 24 h |
| `features:5m:{nameOrig}` | `tx_count, amount_sum, amount_max, amount_avg, risky_tx_count, drained_count, distinct_dest_count, window_start, window_end, last_event_time` | 5 min |
| `features:1h:{nameOrig}` | idem | 1 h |
| `features:24h:{nameOrig}` | idem | 24 h |

Le TTL = taille de la fenêtre : sans activité pendant toute la fenêtre, la clé expire et
`get_features()` renvoie 0.

## Lancement

```powershell
.venv\Scripts\activate
pip install -r requirements.txt

docker compose -f infra/docker-compose.yml up -d kafka kafka-init redis

# Terminal 1 : job Spark
python streaming/spark_streaming_redis.py

# Terminal 2 : flux de transactions
python producer/producer.py --delay 0.01

# Terminal 3 : vérification
python streaming/feature_reader.py --top 5
python streaming/feature_reader.py C1231006815 --bench 1000   # remplacer par un id affiché par --top
python streaming/feature_reader.py --bench 1000               # ou : compte le plus actif choisi automatiquement
```

Tests : `python -m unittest discover tests -v`

## Notes

- **Windows** : Spark exige `winutils.exe` + `hadoop.dll`. Une seule fois :
  `powershell -ExecutionPolicy Bypass -File scripts/setup_winutils.ps1`
  (installe dans `tools/hadoop/bin`, détecté automatiquement par le job).
- **Latence** : depuis Windows, la redirection de port de Docker Desktop ajoute ~1,5 ms
  par aller-retour (un simple `PING` coûte déjà 1,5 ms). `get_features()` fait un seul
  aller-retour ; dans le réseau Docker (API en conteneur), on est bien sous 1 ms.
- Pour rejouer tout le topic : `STARTING_OFFSETS=earliest` et supprimer `checkpoints/`.
- `isFraud` n'est jamais utilisé comme feature (fuite de label).

