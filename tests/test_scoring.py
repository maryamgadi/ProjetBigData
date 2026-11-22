"""
Tests du scoring temps réel (sans modèle entraîné : features et règle de décision).
Exécution : python -m unittest discover tests -v
"""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import fraud_model as fm  # noqa: E402
import scoring  # noqa: E402


class TestBuildFeatures(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "step": [1, 30, 521, 700],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT"],
            "amount": [181.0, 5000.0, 1e7, 120.5],
            "oldbalanceOrg": [181.0, 900.0, 2e7, 0.0],
            "newbalanceOrig": [0.0, 0.0, 1e7, 0.0],
            "oldbalanceDest": [0.0, 1000.0, 0.0, 55.0],
            "newbalanceDest": [0.0, 6000.0, 1e7, 175.5],
            "isFraud": [1, 0, 1, 0],
        })

    def test_same_features_as_training(self):
        """build_features (1 transaction) == fraud_model.add_features (batch d'entraînement)."""
        batch = self.raw.copy()
        batch["type"] = pd.Categorical(batch["type"], categories=fm.FRAUD_TYPES)
        batch = fm.add_features(batch)
        expected = batch[scoring.FEATURES].assign(type=batch["type"].cat.codes).astype(float)

        for i, tx in enumerate(self.raw.to_dict("records")):
            self.assertEqual(scoring.build_features(tx), expected.iloc[i].tolist(), f"ligne {i}")

    def test_ignores_post_transaction_balances(self):
        """Anti-fuite : les soldes après transaction ne changent pas les features."""
        tx = self.raw.iloc[0].to_dict()
        altered = dict(tx, newbalanceOrig=999.0, newbalanceDest=123.0)
        self.assertEqual(scoring.build_features(tx), scoring.build_features(altered))

    def test_exact_drain_vs_overdraft(self):
        names = scoring.FEATURES
        drain = scoring.build_features(self.raw.iloc[0].to_dict())      # 181 == 181
        overdraft = scoring.build_features(self.raw.iloc[1].to_dict())  # 5000 > 900
        self.assertEqual(drain[names.index("amountEqualsBalance")], 1.0)
        self.assertEqual(overdraft[names.index("amountEqualsBalance")], 0.0)
        self.assertEqual(overdraft[names.index("amountExceedsBalance")], 1.0)


class TestDecision(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(scoring.decide(0.10), "APPROVE")
        self.assertEqual(scoring.decide(0.5999), "APPROVE")
        self.assertEqual(scoring.decide(0.60), "2FA")
        self.assertEqual(scoring.decide(0.8499), "2FA")
        self.assertEqual(scoring.decide(0.85), "REFUSE")
        self.assertEqual(scoring.decide(1.0), "REFUSE")


if __name__ == "__main__":
    unittest.main()
