"""
Tests des features de graphe point-in-time (anti-fuite) et du branchement des embeddings.
Exécution : python -m unittest discover tests -v
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import graph_features as gf  # noqa: E402


class TestPriorCumulative(unittest.TestCase):
    def test_counts_only_strictly_earlier_steps(self):
        # compte 7 : événements aux steps 1, 1, 3, 5
        acc = np.array([7, 7, 7, 7])
        step = np.array([1, 1, 3, 5])
        val = np.ones(4)
        # requêtes au step 1 (rien avant), 2 (2 avant), 3 (même step exclu -> 2), 6 (4)
        q_acc = np.array([7, 7, 7, 7, 8])
        q_step = np.array([1, 2, 3, 6, 6])
        got = gf.prior_cumulative(acc, step, val, q_acc, q_step)
        self.assertEqual(got.tolist(), [0.0, 2.0, 2.0, 4.0, 0.0])   # compte 8 inconnu -> 0

    def test_sums_values(self):
        got = gf.prior_cumulative(np.array([1, 1]), np.array([1, 2]), np.array([10.0, 5.0]),
                                  np.array([1]), np.array([3]))
        self.assertEqual(got.tolist(), [15.0])


class TestComputeGraphFeatures(unittest.TestCase):
    def setUp(self):
        # Chaîne de type mule : A -> M (step 1), B -> M (step 2), M -> C (step 3), D -> M (step 3)
        self.history = pd.DataFrame({
            "step": [1, 2], "amount": [100.0, 50.0],
            "nameOrig": ["CA", "CB"], "nameDest": ["CM", "CM"],
        })
        self.stream = pd.DataFrame({
            "step": [3, 3], "amount": [140.0, 30.0],
            "nameOrig": ["CM", "CD"], "nameDest": ["CC", "CM"],
        }, index=[10, 11])

    def test_point_in_time_features(self):
        out = gf.compute_graph_features({"h": self.history, "s": self.stream})
        h, s = out["h"], out["s"]

        # 1re transaction : rien avant -> tout à 0 (la transaction ne se compte pas elle-même)
        self.assertEqual(h.iloc[0].tolist(), [0.0] * len(gf.GRAPH_FEATURES))
        # 2e : M a déjà reçu 1 transaction de 100
        self.assertEqual(h.iloc[1]["g_dest_in_count"], 1.0)
        self.assertAlmostEqual(h.iloc[1]["g_dest_in_amount_log"], np.log1p(100.0))

        # Flux : M émet au step 3 -> M (émetteur) a reçu 2 transactions avant
        self.assertEqual(s.loc[10, "g_orig_in_count"], 2.0)
        # D -> M au step 3 : M a reçu 2 (steps 1-2) ; son émission du même step 3 est exclue
        self.assertEqual(s.loc[11, "g_dest_in_count"], 2.0)
        self.assertEqual(s.loc[11, "g_dest_out_count"], 0.0)
        # L'index d'origine est conservé (jointure avec le CSV)
        self.assertEqual(list(s.index), [10, 11])

    def test_future_does_not_change_past(self):
        """Ajouter des transactions futures ne modifie pas les features des transactions passées."""
        base = gf.compute_graph_features({"h": self.history})["h"]
        future = pd.DataFrame({"step": [9], "amount": [1e6], "nameOrig": ["CZ"], "nameDest": ["CM"]})
        with_future = gf.compute_graph_features({"h": self.history, "f": future})["h"]
        pd.testing.assert_frame_equal(base, with_future)


class TestEmbeddings(unittest.TestCase):
    def test_attach_known_and_cold_start(self):
        emb = pd.DataFrame({"account": ["CA", "CM"], "emb_0": [0.5, 1.0], "emb_1": [-1.0, 2.0]})
        df = pd.DataFrame({"amount": [1.0, 2.0]})
        out = gf.attach_embeddings_frame(df, np.array(["CA", "CX"]), np.array(["CM", "CM"]), emb)

        self.assertEqual(out["known_eo"].tolist(), [1, 0])          # CX inconnu
        self.assertEqual(out["eo_emb_0"].tolist(), [0.5, 0.0])      # vecteur nul si inconnu
        self.assertEqual(out["ed_emb_1"].tolist(), [2.0, 2.0])
        self.assertEqual(out["known_ed"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
