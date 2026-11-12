"""
Feature Store Redis : schéma des clés, écriture (côté Spark) et lecture (côté API / ML).

Ce module n'importe PAS pyspark : il est envoyé aux workers Spark (addPyFile)
et pourra être réutilisé tel quel par l'API FastAPI en Semaine 3.

Organisation des clés Redis
---------------------------
    account:{nameOrig}          HASH  dernière transaction du compte      TTL 24h
    features:5m:{nameOrig}      HASH  agrégats fenêtre glissante 5 min    TTL 5 min
    features:1h:{nameOrig}      HASH  agrégats fenêtre glissante 1 h      TTL 1 h
    features:24h:{nameOrig}     HASH  agrégats fenêtre glissante 24 h     TTL 24 h

Le TTL de chaque clé de fenêtre est égal à la taille de la fenêtre : si un compte
n'a plus d'activité pendant toute la durée de la fenêtre, la clé disparaît
d'elle-même et la lecture renvoie 0 (aucune activité récente).
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional

import redis

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

ID_FIELD = "nameOrig"
ACCOUNT_PREFIX = "account:"
ACCOUNT_TTL_SECONDS = 24 * 3600

# label -> (taille fenêtre Spark, pas de glissement Spark, TTL Redis en secondes)
# Chaque transaction est copiée dans taille/pas fenêtres : 5 + 4 + 12 = 21 lignes d'état
# (au lieu de 41 avec des pas de 5 min / 1 h), ce qui divise la mémoire d'état par 2.
WINDOWS: Dict[str, tuple] = {
    "5m": ("5 minutes", "1 minute", 5 * 60),
    "1h": ("1 hour", "15 minutes", 3600),
    "24h": ("24 hours", "2 hours", 24 * 3600),
}

# Champs stockés dans chaque hash features:{label}:{account}
WINDOW_FEATURES = (
    "tx_count",             # nombre de transactions
    "amount_sum",           # montant total
    "amount_max",           # plus gros montant
    "amount_avg",           # montant moyen
    "risky_tx_count",       # nb de TRANSFER / CASH_OUT
    "drained_count",        # nb de transactions qui vident le compte (solde -> 0)
    "distinct_dest_count",  # nb (approx.) de bénéficiaires distincts
)

# Champs stockés dans le hash account:{account}
ACCOUNT_FEATURES = (
    "last_type",
    "last_amount",
    "last_step",
    "last_dest",
    "last_balance_error_orig",
    "last_event_time",
)
_ACCOUNT_STRING_FIELDS = {"last_type", "last_dest", "last_event_time"}


def window_prefix(label: str) -> str:
    return f"features:{label}:"


def feature_names() -> list:
    """Liste ordonnée des features renvoyées par get_features() (pour le ML)."""
    names = list(ACCOUNT_FEATURES)
    for label in WINDOWS:
        names += [f"{f}_{label}" for f in WINDOW_FEATURES]
    return names


def get_client(host: Optional[str] = None, port: Optional[int] = None,
               db: Optional[int] = None) -> redis.Redis:
    return redis.Redis(
        host=host or REDIS_HOST,
        port=port or REDIS_PORT,
        db=REDIS_DB if db is None else db,
        decode_responses=True,
        socket_connect_timeout=5,
    )


# ---------------------------------------------------------------------------
# Écriture (exécutée sur les workers Spark)
# ---------------------------------------------------------------------------
def to_redis_value(value: Any) -> Optional[str]:
    """Convertit une valeur Spark en chaîne Redis (None = champ ignoré)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return repr(round(value, 2))
    return str(value)


def to_mapping(record: Mapping[str, Any], exclude: Iterable[str] = (ID_FIELD,)) -> Dict[str, str]:
    excluded = set(exclude)
    mapping = {}
    for field, value in record.items():
        if field in excluded:
            continue
        converted = to_redis_value(value)
        if converted is not None:
            mapping[field] = converted
    return mapping


def write_records(client, records: Iterable[Mapping[str, Any]], key_prefix: str,
                  ttl_seconds: int, chunk_size: int = 500) -> int:
    """Écrit des enregistrements {nameOrig, champs...} en HASH avec TTL, par pipeline."""
    pipe = client.pipeline(transaction=False)
    written = pending = 0
    for record in records:
        account = record.get(ID_FIELD)
        if not account:
            continue
        mapping = to_mapping(record)
        if not mapping:
            continue
        key = f"{key_prefix}{account}"
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl_seconds)
        written += 1
        pending += 1
        if pending >= chunk_size:
            pipe.execute()
            pending = 0
    if pending:
        pipe.execute()
    return written


def make_partition_writer(key_prefix: str, ttl_seconds: int, host: Optional[str] = None,
                          port: Optional[int] = None) -> Callable[[Iterator], None]:
    """Fonction à passer à DataFrame.foreachPartition : une connexion Redis par partition."""
    host = host or REDIS_HOST
    port = port or REDIS_PORT

    def write_partition(rows: Iterator) -> None:
        client = get_client(host, port)
        try:
            write_records(client, (row.asDict() for row in rows), key_prefix, ttl_seconds)
        finally:
            client.close()

    return write_partition


# ---------------------------------------------------------------------------
# Lecture (API / ML) : 1 seul aller-retour réseau grâce au pipeline
# ---------------------------------------------------------------------------
def _to_number(value: Optional[str], default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_features(client, account: str) -> Dict[str, Any]:
    """Renvoie toutes les features temps réel d'un compte (0 si pas d'activité récente)."""
    pipe = client.pipeline(transaction=False)
    pipe.hgetall(f"{ACCOUNT_PREFIX}{account}")
    for label in WINDOWS:
        pipe.hgetall(f"{window_prefix(label)}{account}")
    results = pipe.execute()

    last_tx, window_hashes = results[0] or {}, results[1:]
    features: Dict[str, Any] = {}
    for field in ACCOUNT_FEATURES:
        raw = last_tx.get(field)
        features[field] = raw if field in _ACCOUNT_STRING_FIELDS else _to_number(raw, default=None)
    for label, data in zip(WINDOWS, window_hashes):
        data = data or {}
        for field in WINDOW_FEATURES:
            features[f"{field}_{label}"] = _to_number(data.get(field))
    features["known_account"] = bool(last_tx) or any(window_hashes)
    return features

