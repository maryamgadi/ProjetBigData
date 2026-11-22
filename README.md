# Plateforme Big Data & ML de détection de fraude en temps réel

Projet Big Data (cahier des charges CDC-FRAUD-2026-V1), dataset **PaySim**.

```
producer.py ──► Kafka (financial-transactions, 3 partitions, clé = nameOrig)
                   │
                   ├──► Spark Streaming ──► Delta Lake bronze sur MinIO   (src/11)
                   └──► Spark Streaming ──► Redis Feature Store            (semaine 2)
```

## Prérequis

- Docker Desktop
- Python 3.10+ (`pip install -r requirements.txt`) pour le producteur et les scripts 01–07
- Le CSV PaySim (Kaggle) placé dans `data/archive/PS_20174392719_1491204439457_log.csv`

## Démarrer la stack

```bash
cp infra/.env.example infra/.env      # puis changer le mot de passe MinIO
cd infra
docker compose up -d
```

| Service | Conteneur | Accès |
|---|---|---|
| Kafka (KRaft) | `kafka-broker` | `localhost:9092` (hôte), `kafka:29092` (réseau docker) |
| Redis | `redis-feature-store` | `localhost:6379` |
| MinIO | `fraud-minio` | API `localhost:9000`, console http://localhost:9001 |
| Spark | `fraud-spark` | UI http://localhost:4040 pendant un job |

`kafka-init` crée le topic, `minio-init` crée les buckets
(`paysim-raw`, `paysim-historical`, `paysim-streaming`, `fraud-lakehouse`).

## Pipeline

### 1. Préparation batch (hôte, Python)

| Script | Rôle |
|---|---|
| `01_exploration_paysim.py` | Exploration du dataset |
| `02_upload_paysim_minio.py` | Upload du CSV brut dans `paysim-raw` |
| `03_split_paysim.py` | Découpage chronologique 70 % historique / 30 % streaming |
| `04_upload_split_minio.py` | Upload des deux parties dans MinIO |
| `05_validation_split.py` | Contrôle du découpage |
| `06_verify_minio.py` | Contrôle des objets MinIO |
| `07_prepare_spark_sample.py` | Échantillon de 1000 lignes pour tester Spark |

### 2. Traitements Spark (conteneur `fraud-spark`)

Tous les jobs se lancent avec le même wrapper, qui ajoute les connecteurs Kafka, S3A et Delta :

```bash
docker compose exec spark /workspace/scripts/run_spark.sh src/<job>.py
```

| Job | Rôle |
|---|---|
| `src/08_spark_read_minio.py` | Lecture S3A depuis MinIO |
| `src/09_csv_to_parquet.py` / `src/10_full_csv_to_parquet.py` | Historique CSV → Parquet |
| `src/11_stream_kafka_to_delta.py` | **Ingestion continue Kafka → Delta bronze** (`s3a://fraud-lakehouse/bronze/transactions`) |
| `src/12_verify_no_loss.py` | Preuve « zéro perte » : offsets Kafka vs lignes Delta |
| `src/13_historical_to_delta.py` | Historique PaySim (CSV MinIO) → Delta `bronze/historical`, partitionné par `type` |
| `src/14_delta_demo.py` | Démo Delta : transaction ACID, Time Travel, RESTORE, schema enforcement, OPTIMIZE |

### 3. Baseline ML (hôte, Python — Docker peut rester arrêté)

Prérequis : `data/processed/paysim_historical.csv` et `paysim_streaming.csv` (script `03_split_paysim.py`).

```bash
.venv\Scripts\python src\15_baseline_lightgbm.py                     # features pré-transaction (réaliste)
.venv\Scripts\python src\15_baseline_lightgbm.py --features post_tx  # avec soldes après transaction (fuite)
```

Entraînement chronologique (85 % historique / 15 % validation), test sur la partie streaming.
Métriques dans `reports/baseline_metrics_<set>.json`, modèles dans `models/` (non versionné).

| Jeu de features | AUC-ROC | PR-AUC | Rappel @0,60 | Faux positifs (toutes tx) |
|---|---|---|---|---|
| `pre_tx` (réaliste, retenu) | 0,9975 | 0,998 | 99,75 % | 0,0004 % (1 FP) |
| `pre_tx` v1 (`amount >= balance`) | 0,984 | 0,920 | 95,1 % | 0,17 % |
| `post_tx` (fuite) | 0,9996 | 0,999 | 99,0 % | 0,004 % |
| Cible CDC | > 0,96 | | > 94 % | < 0,1 % |

Les faux positifs de la v1 venaient d'une feature trop grossière : `amount >= oldbalanceOrg`
mélangeait les fraudes (le fraudeur vide le compte **exactement** : 97,6 % des fraudes) et des
clients légitimes qui demandent plus que leur solde (0 % d'égalité exacte). La v2 sépare
`amountEqualsBalance` et `amountExceedsBalance`.

Limite : ce motif « vidage exact du compte » est une règle du simulateur PaySim ; sur des données
réelles il serait moins net. C'est pour cela que la v1 est gardée comme ablation dans le rapport
(`--features pre_tx_v1`).

### 4. Test de robustesse : montants bruités

```bash
.venv\Scripts\python src\16_robustness_noisy_amounts.py      # ~5 min
```

Chaque fraudeur laisse une part aléatoire `u ~ U(0, niveau)` du solde sur le compte : le vidage
exact disparaît (97,6 % → 0,4 % des fraudes). Les transactions légitimes ne sont pas modifiées.
Deux scénarios : **A** modèle entraîné sur données propres, fraudeurs qui changent de comportement
en production ; **B** modèle réentraîné sur des fraudes déjà bruitées.

| Bruit max | A — `pre_tx` rappel / AUC | A — `pre_tx_v1` rappel / AUC | B — `pre_tx` rappel / FP | B — `pre_tx_v1` rappel / FP |
|---|---|---|---|---|
| 0 % | 99,75 % / 0,998 | 95,1 % / 0,984 | — | — |
| 0,1 % | 19,1 % / 0,335 | 48,9 % / 0,769 | 99,6 % / 0,02 % | 98,4 % / 0,05 % |
| 1 % | 19,1 % / 0,334 | 48,9 % / 0,770 | 98,3 % / 0,12 % | 89,2 % / 0,15 % |
| 5 % | 19,1 % / 0,333 | 49,0 % / 0,773 | 98,4 % / 0,20 % | 91,9 % / 0,13 % |
| 10 % | 19,7 % / 0,341 | 49,0 % / 0,777 | 97,5 % / 0,33 % | 88,9 % / 0,35 % |

Lecture :
- **A** : la baseline officielle s'effondre dès qu'un fraudeur laisse 0,1 % sur le compte
  (rappel 19 %, AUC 0,33 : pire que le hasard, une fraude « non exacte » ressemble à une
  transaction légitime). Elle a appris la règle du simulateur, pas la fraude.
- **B** : réentraînée, elle retrouve le rappel (> 97 %) mais les faux positifs dépassent la
  cible de 0,1 % dès 1 % de bruit. Les seules variables de montant/solde ne suffisent plus.
- C'est l'écart que le modèle **GNN + LightGBM** (semaine 3) doit combler avec la structure du
  réseau de comptes (comptes mules, chaînes TRANSFER → CASH_OUT).

Résultats complets : `reports/robustness_noisy_amounts.csv` / `.json`.

### 5. Suivi des expériences : MLflow

Chaque exécution des scripts 15 et 16 crée un run dans l'expérience `paysim-fraud-detection`
(base locale `mlflow.db`, artefacts dans `mlruns/`, non versionnés) :

| Enregistré | Contenu |
|---|---|
| Paramètres | jeu de features, liste des features, hyperparamètres LightGBM (`lgbm.*`), seuils 0,60 / 0,85 |
| Métriques | AUC-ROC, PR-AUC, rappel / précision / FPR / TP / FP / FN à chaque seuil, `best_iteration` |
| Tags données | chemin, lignes utilisées, taille et **empreinte SHA-256** des CSV d'entraînement et de test, table Delta correspondante |
| Artefacts | rapport JSON, zones de décision, **modèle LightGBM** rechargeable (`runs:/<id>/model`) |

Le test de robustesse crée un run parent et un run enfant par (modèle, scénario, niveau de bruit),
comparables côte à côte dans l'interface.

```bash
.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db
```
puis http://localhost:5000. Pour viser un serveur MLflow partagé : variable `MLFLOW_TRACKING_URI`.

Auditabilité (exigence du CDC) : l'empreinte SHA-256 identifie la version exacte des données de
chaque modèle ; la même donnée est versionnée dans Delta (`bronze/historical`, Time Travel).

`post_tx` n'est pas utilisable : le CDC note la transaction **avant validation**, les soldes
après transaction ne sont pas encore connus et, dans PaySim, ils trahissent la fraude.

### 6. Démo de bout en bout (livrable semaine 1)

```bash
# Terminal 1 : job streaming (tourne en continu)
docker compose exec spark /workspace/scripts/run_spark.sh src/11_stream_kafka_to_delta.py

# Terminal 2 : envoi des transactions (depuis la racine du repo)
python producer/producer.py --file data/paysim_sample.csv --delay 0

# Quand le producteur a fini et que le job a rattrapé (Ctrl+C sur le terminal 1) :
docker compose exec spark /workspace/scripts/run_spark.sh src/12_verify_no_loss.py
```

Test de reprise : arrêter le job 11 au milieu de l'envoi, le relancer ; le
checkpoint le fait repartir au bon offset et le script 12 doit toujours
afficher `OK - aucune perte, aucun doublon`.

## Réinitialiser

```bash
docker compose down -v   # supprime aussi les volumes (Kafka, MinIO, Redis)
```

## Inférence temps réel

`src/scoring.py` score **une** transaction (features → LightGBM → décision). Il est prévu pour être
importé tel quel par l'API FastAPI :

```python
from scoring import FraudScorer
scorer = FraudScorer()   # charge models/baseline_lgbm_pre_tx.txt une seule fois
scorer.score({"step": 521, "type": "TRANSFER", "amount": 181.0,
              "oldbalanceOrg": 181.0, "oldbalanceDest": 0.0})
# {"score": 0.99..., "decision": "REFUSE", "latency_ms": 0.1}
```

- Entrée : `step, type, amount, oldbalanceOrg, oldbalanceDest`, c'est-à-dire uniquement ce
  qui est connu **avant** validation.
- Les types sans fraude dans PaySim (`PAYMENT`, `CASH_IN`, `DEBIT`) reçoivent le score 0, donc
  `APPROVE`.
- Décision : `< 0,60` → `APPROVE`, `0,60–0,85` → `2FA`, `≥ 0,85` → `REFUSE`.

Benchmark (`reports/inference_benchmark.json`) :

```bash
.venv\Scripts\python src\17_inference_benchmark.py
```

| Mesure | Résultat | Objectif CDC |
|---|---|---|
| Cohérence avec l'évaluation batch | écart max 0, 116 680 / 116 680 décisions identiques | — |
| Latence p50 / p95 / p99 | 0,05 / 0,17 / 0,38 ms | < 20 ms |
| Débit sur 1 cœur | ~13 000 tx/s | 10k tx/s (en pic, tout le système) |

Mesure sur un PC portable, 1 thread. Elle ne comprend ni le HTTP (FastAPI) ni la lecture Redis
(~1,5 ms depuis Windows) : le budget de 20 ms laisse une large marge pour ces deux étapes, qui
seront mesurées par le test de charge Locust (P95 < 80 ms de bout en bout).
