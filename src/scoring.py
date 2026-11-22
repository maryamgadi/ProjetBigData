"""Scoring temps réel d'UNE transaction : features -> LightGBM -> décision.

Module pensé pour être importé tel quel par l'API FastAPI (semaine 3) :

    from scoring import FraudScorer
    scorer = FraudScorer()                      # charge le modèle une seule fois au démarrage
    scorer.score({"step": 521, "type": "TRANSFER", "amount": 181.0,
                  "oldbalanceOrg": 181.0, "oldbalanceDest": 0.0})
    # -> {"score": 0.99, "decision": "REFUSE", "latency_ms": 0.3}

Uniquement des données connues AVANT validation (pas de newbalance*) : c'est ce que
reçoit l'API au moment où la banque doit décider.
Objectif CDC : inférence < 20 ms (service d'inférence, section 4).
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fraud_model import FEATURE_SETS, FRAUD_TYPES, MODEL_DIR, THRESHOLD_2FA, THRESHOLD_REFUSE

FEATURE_SET = "pre_tx"
FEATURES = FEATURE_SETS[FEATURE_SET]
DEFAULT_MODEL = MODEL_DIR / f"baseline_lgbm_{FEATURE_SET}.txt"
REQUIRED_FIELDS = ("step", "type", "amount", "oldbalanceOrg", "oldbalanceDest")

# Même ordre que pd.Categorical(categories=FRAUD_TYPES) à l'entraînement :
# LightGBM reçoit le code de la catégorie (TRANSFER = 0, CASH_OUT = 1).
TYPE_CODES = {t: i for i, t in enumerate(FRAUD_TYPES)}


def decide(score):
    """Règle métier du CDC : < 0,60 approuvé, 0,60–0,85 authentification forte, > 0,85 refus."""
    if score >= THRESHOLD_REFUSE:
        return "REFUSE"
    if score >= THRESHOLD_2FA:
        return "2FA"
    return "APPROVE"


def build_features(tx):
    """Features pre_tx d'une transaction brute, dans l'ordre attendu par le modèle.

    Doit rester identique à fraud_model.add_features (vérifié par tests/test_scoring.py
    et par le benchmark 17 sur tout le jeu de test).
    """
    amount = float(tx["amount"])
    old_orig = float(tx["oldbalanceOrg"])
    old_dest = float(tx["oldbalanceDest"])
    values = {
        "type": TYPE_CODES[tx["type"]],
        "hour": int(tx["step"]) % 24,
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "oldbalanceDest": old_dest,
        "amountToBalanceOrig": amount / (old_orig + 1.0),
        "destEmptyBefore": float(old_dest == 0),
        "amountEqualsBalance": float(abs(amount - old_orig) < 0.01),
        "amountExceedsBalance": float(amount > old_orig + 0.01),
        "amountIsCap": float(amount == 1e7),
        "remainingOrig": old_orig - amount,
    }
    return [values[name] for name in FEATURES]


class FraudScorer:
    """Charge le modèle une fois, puis score une transaction en une fraction de milliseconde."""

    def __init__(self, model_path=DEFAULT_MODEL):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modèle introuvable : {model_path}. Lancer d'abord src/15_baseline_lightgbm.py")
        self.model = lgb.Booster(model_file=str(model_path))
        if self.model.feature_name() != FEATURES:
            raise ValueError(f"Le modèle attend {self.model.feature_name()}, "
                             f"le scorer produit {FEATURES}")
        self.model_path = model_path

    def predict_proba(self, tx):
        """Probabilité de fraude. Types sans fraude dans PaySim (PAYMENT, CASH_IN, DEBIT) -> 0."""
        missing = [f for f in REQUIRED_FIELDS if f not in tx]
        if missing:
            raise KeyError(f"Champs manquants : {missing}")
        if tx["type"] not in TYPE_CODES:
            return 0.0
        x = np.array([build_features(tx)], dtype=np.float64)
        # 1 thread : pour une seule ligne, répartir sur plusieurs cœurs coûte plus que ça ne rapporte
        return float(self.model.predict(x, num_threads=1)[0])

    def score(self, tx):
        start = time.perf_counter()
        proba = self.predict_proba(tx)
        return {
            "score": proba,
            "decision": decide(proba),
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }
