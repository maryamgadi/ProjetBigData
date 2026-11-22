"""
Tests unitaires pour l'encodeur GraphSAGE et l'extraction d'embeddings 64 dimensions.
Exécution : python -m unittest tests/test_graphsage.py -v
"""

import os
import sys
import tempfile
import unittest

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
import graph_builder as gb  # noqa: E402
import graphsage as gs  # noqa: E402


class TestGraphSAGE(unittest.TestCase):
    def setUp(self):
        # Mini-graphe synthétique PaySim avec suffisamment d'arêtes pour le split train/val/test
        rows = []
        for i in range(20):
            is_fraud = 1 if i % 4 == 0 else 0
            rows.append({
                "step": i + 1,
                "type": "TRANSFER" if is_fraud else "PAYMENT",
                "amount": 50000.0 if is_fraud else 100.0 + i,
                "nameOrig": f"C100{i % 6}",
                "oldbalanceOrg": 50000.0 if is_fraud else 1000.0,
                "newbalanceOrig": 0.0 if is_fraud else 900.0,
                "nameDest": f"C200{i % 5}" if is_fraud else f"M300{i % 4}",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 50000.0 if is_fraud else 0.0,
                "isFraud": is_fraud,
            })
        self.df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "mini.csv")
            self.df.to_csv(csv_path, index=False)
            self.graph, _ = gb.build_paysim_graph(file_path=csv_path)

    def test_encoder_output_shape_64d(self):
        encoder = gs.GraphSAGEEncoder(
            in_channels=self.graph.x.shape[1],
            hidden_channels=32,
            out_channels=64,
        )
        encoder.eval()
        with torch.no_grad():
            emb = encoder(self.graph.x, self.graph.edge_index)
        self.assertEqual(emb.shape, (self.graph.num_nodes, 64))
        self.assertFalse(torch.isnan(emb).any().item())

    def test_train_and_extract_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "graphsage_best.pt")
            emb_csv_path = os.path.join(tmpdir, "embeddings.csv")

            model, x_scaled, edge_index_cpu, metrics = gs.train_graphsage(
                graph=self.graph,
                epochs=3,
                hidden_dim=32,
                emb_dim=64,
                out_model=ckpt_path,
                seed=42,
            )

            self.assertTrue(os.path.exists(ckpt_path))
            self.assertIn("roc_auc", metrics)

            id_to_node = getattr(self.graph, "id_to_node", None)
            emb = gs.extract_embeddings(
                model.encoder.cpu(),
                x_scaled,
                edge_index_cpu,
                id_to_node,
                out_embeddings=emb_csv_path,
            )

            self.assertEqual(emb.shape, (self.graph.num_nodes, 64))
            self.assertTrue(os.path.exists(emb_csv_path))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "embeddings.pt")))

            df_emb = pd.read_csv(emb_csv_path)
            self.assertEqual(df_emb.shape, (self.graph.num_nodes, 65))
            self.assertEqual(df_emb.columns[0], "account_id")


if __name__ == "__main__":
    unittest.main()
