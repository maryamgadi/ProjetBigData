"""
Tests unitaires pour la modélisation du graphe PaySim (PyTorch Geometric).
Exécution : python -m unittest discover tests -v
"""

import os
import sys
import tempfile
import unittest
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml"))
import graph_builder as gb  # noqa: E402


class TestGraphBuilder(unittest.TestCase):
    def setUp(self):
        # Création d'un mini-dataset de test synthétique PaySim
        self.sample_data = pd.DataFrame({
            "step": [1, 2, 3, 4],
            "type": ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER"],
            "amount": [100.0, 50000.0, 50000.0, 200.0],
            "nameOrig": ["C1001", "C1002", "C1003", "C1001"],
            "oldbalanceOrg": [1000.0, 50000.0, 50000.0, 900.0],
            "newbalanceOrig": [900.0, 0.0, 0.0, 700.0],
            "nameDest": ["M2001", "C1003", "M2002", "C1002"],
            "oldbalanceDest": [0.0, 1000.0, 0.0, 0.0],
            "newbalanceDest": [0.0, 51000.0, 0.0, 200.0],
            "isFraud": [0, 1, 1, 0]
        })

    def test_build_node_mappings(self):
        node_to_id, id_to_node = gb.build_node_mappings(self.sample_data)
        # Nœuds uniques : C1001, C1002, C1003, M2001, M2002 (5 nœuds)
        self.assertEqual(len(node_to_id), 5)
        self.assertEqual(len(id_to_node), 5)
        self.assertIn("C1001", node_to_id)
        self.assertIn("M2001", node_to_id)
        self.assertEqual(id_to_node[node_to_id["C1001"]], "C1001")

    def test_extract_node_features(self):
        node_to_id, id_to_node = gb.build_node_mappings(self.sample_data)
        x, feature_names = gb.extract_node_features(self.sample_data, node_to_id, id_to_node)
        
        self.assertEqual(x.shape[0], 5)
        self.assertEqual(x.shape[1], len(feature_names))
        
        # Vérifier que M2001 est bien détecté comme marchand
        m_idx = node_to_id["M2001"]
        is_merchant_col = feature_names.index("is_merchant")
        self.assertEqual(x[m_idx, is_merchant_col].item(), 1.0)

        # C1001 n'est pas un marchand
        c_idx = node_to_id["C1001"]
        self.assertEqual(x[c_idx, is_merchant_col].item(), 0.0)

        # C1003 a reçu et envoyé (candidat mule)
        c1003_idx = node_to_id["C1003"]
        mule_col = feature_names.index("is_mule_candidate")
        self.assertEqual(x[c1003_idx, mule_col].item(), 1.0)

    def test_extract_edges_and_features(self):
        node_to_id, _ = gb.build_node_mappings(self.sample_data)
        edge_index, edge_attr, y_edge, edge_feature_names = gb.extract_edges_and_features(self.sample_data, node_to_id)

        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], 4)  # 4 transactions
        self.assertEqual(edge_attr.shape[0], 4)
        self.assertEqual(edge_attr.shape[1], len(edge_feature_names))
        self.assertEqual(y_edge.shape[0], 4)
        self.assertEqual((y_edge == 1).sum().item(), 2)  # 2 fraudes

    def test_build_and_save_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_paysim.csv")
            self.sample_data.to_csv(csv_path, index=False)

            graph_path = os.path.join(tmpdir, "graph.pt")
            meta_path = os.path.join(tmpdir, "meta.json")

            graph, meta = gb.build_paysim_graph(file_path=csv_path)
            gb.save_graph(graph, meta, output_graph_path=graph_path, output_meta_path=meta_path)

            self.assertTrue(os.path.exists(graph_path))
            self.assertTrue(os.path.exists(meta_path))

            # Rechargement et vérification
            loaded_graph = torch.load(graph_path, weights_only=False)
            self.assertEqual(loaded_graph.num_nodes, 5)
            self.assertEqual(loaded_graph.edge_index.shape[1], 4)
            self.assertTrue(loaded_graph.validate())


if __name__ == "__main__":
    unittest.main()

