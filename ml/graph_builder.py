"""
Mission 2 - Semaine 2 : Modélisation du Graphe PaySim sous PyTorch Geometric (PyG)

Ce module transforme les transactions financières en un graphe orienté hétérogène / relationnel :
    - Nœuds : Comptes bancaires et marchands (nameOrig, nameDest).
    - Arêtes (edge_index) : Flux financiers orientés (nameOrig -> nameDest).
    - Features arêtes (edge_attr) : Montants (log), types encodés, step, anomalies de solde.
    - Features nœuds (x) : Type (Client/Marchand), degrés (in/out), volumes cumulés, rôle (mule/source/cible).
    - Labels (y) : Cible de fraude (isFraud).

L'objet produit est une instance de torch_geometric.data.Data prête pour GraphSAGE en Semaine 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Types de transactions connus dans PaySim
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
RISKY_TYPES = {"TRANSFER", "CASH_OUT"}
MAX_STEP_NORMALIZATION = 744.0  # PaySim simule 744 heures (30 jours)


def load_dataset(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Charge le fichier PaySim (CSV ou Parquet) avec validation des colonnes."""
    print(f"[1/5] Chargement des transactions depuis : {file_path}")
    start = time.perf_counter()
    
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
        if nrows:
            df = df.head(nrows)
    else:
        df = pd.read_csv(file_path, nrows=nrows)

    required_cols = {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

    elapsed = time.perf_counter() - start
    print(f"      {len(df):,} transactions chargées ({elapsed:.2f} s)")
    return df


def build_node_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """Construit un mapping bi-directionnel entre identifiants textuels et entiers 0..N-1."""
    print("[2/5] Indexation des comptes uniques...")
    start = time.perf_counter()

    # Union de tous les comptes émetteurs et destinataires
    orig_nodes = df["nameOrig"].unique()
    dest_nodes = df["nameDest"].unique()
    all_unique_nodes = np.unique(np.concatenate([orig_nodes, dest_nodes]))

    node_to_id = {name: idx for idx, name in enumerate(all_unique_nodes)}
    id_to_node = all_unique_nodes.tolist()

    elapsed = time.perf_counter() - start
    print(f"      {len(all_unique_nodes):,} comptes uniques identifiés ({elapsed:.2f} s)")
    print(f"      - Émetteurs uniques    : {len(orig_nodes):,}")
    print(f"      - Destinataires uniques : {len(dest_nodes):,}")
    return node_to_id, id_to_node


def extract_node_features(df: pd.DataFrame, node_to_id: Dict[str, int], 
                          id_to_node: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """
    Construit les caractéristiques des nœuds (x) :
      1. is_merchant : 1 si le compte commence par 'M', 0 si 'C'
      2. out_degree : nombre de transactions émises
      3. in_degree : nombre de transactions reçues
      4. total_degree : in_degree + out_degree
      5. out_amount_sum : log1p(somme des montants envoyés)
      6. in_amount_sum : log1p(somme des montants reçus)
      7. out_amount_mean : log1p(montant moyen envoyé)
      8. in_amount_mean : log1p(montant moyen reçu)
      9. is_source_only : 1 si le compte n'a fait qu'émettre (out > 0 et in == 0)
     10. is_dest_only : 1 si le compte n'a fait que recevoir (in > 0 et out == 0)
     11. is_mule_candidate : 1 si le compte reçoit ET émet (in > 0 et out > 0, typique des mules)
    """
    print("[3/5] Extraction des caractéristiques des nœuds (x)...")
    start = time.perf_counter()
    num_nodes = len(id_to_node)

    # 1. Type de compte (Client vs Marchand)
    is_merchant = np.array([1.0 if name.startswith("M") else 0.0 for name in id_to_node], dtype=np.float32)

    # 2. Statistiques émetteur (out)
    orig_stats = df.groupby("nameOrig")["amount"].agg(["count", "sum", "mean"]).reset_index()
    orig_idx = [node_to_id[name] for name in orig_stats["nameOrig"]]

    out_degree = np.zeros(num_nodes, dtype=np.float32)
    out_amount_sum = np.zeros(num_nodes, dtype=np.float32)
    out_amount_mean = np.zeros(num_nodes, dtype=np.float32)

    out_degree[orig_idx] = orig_stats["count"].values
    out_amount_sum[orig_idx] = np.log1p(orig_stats["sum"].values)
    out_amount_mean[orig_idx] = np.log1p(orig_stats["mean"].values)

    # 3. Statistiques destinataire (in)
    dest_stats = df.groupby("nameDest")["amount"].agg(["count", "sum", "mean"]).reset_index()
    dest_idx = [node_to_id[name] for name in dest_stats["nameDest"]]

    in_degree = np.zeros(num_nodes, dtype=np.float32)
    in_amount_sum = np.zeros(num_nodes, dtype=np.float32)
    in_amount_mean = np.zeros(num_nodes, dtype=np.float32)

    in_degree[dest_idx] = dest_stats["count"].values
    in_amount_sum[dest_idx] = np.log1p(dest_stats["sum"].values)
    in_amount_mean[dest_idx] = np.log1p(dest_stats["mean"].values)

    # 4. Rôles structurels
    total_degree = in_degree + out_degree
    is_source_only = ((out_degree > 0) & (in_degree == 0)).astype(np.float32)
    is_dest_only = ((in_degree > 0) & (out_degree == 0)).astype(np.float32)
    is_mule = ((in_degree > 0) & (out_degree > 0)).astype(np.float32)

    features = np.column_stack([
        is_merchant,
        out_degree,
        in_degree,
        total_degree,
        out_amount_sum,
        in_amount_sum,
        out_amount_mean,
        in_amount_mean,
        is_source_only,
        is_dest_only,
        is_mule
    ])

    feature_names = [
        "is_merchant", "out_degree", "in_degree", "total_degree",
        "out_amount_sum_log", "in_amount_sum_log",
        "out_amount_mean_log", "in_amount_mean_log",
        "is_source_only", "is_dest_only", "is_mule_candidate"
    ]

    x_tensor = torch.tensor(features, dtype=torch.float32)
    elapsed = time.perf_counter() - start
    print(f"      Matrice x : {list(x_tensor.shape)} ({len(feature_names)} features par compte, {elapsed:.2f} s)")
    return x_tensor, feature_names


def extract_edges_and_features(df: pd.DataFrame, 
                               node_to_id: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Construit :
      - edge_index [2, E] : tenseur des indices [source, destination]
      - edge_attr [E, F] : caractéristiques de chaque transaction :
          1. amount_log : log1p(amount)
          2. step_norm : step normalisé (0..1)
          3-7. One-Hot encoding des types (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
          8. is_risky_type : 1 si TRANSFER ou CASH_OUT
          9. orig_balance_error : écart de solde émetteur
         10. is_drained : 1 si solde émetteur vidé
      - y_edge [E] : label de fraude transactionnelle (isFraud)
    """
    print("[4/5] Construction des arêtes (edge_index) et de leurs attributs (edge_attr)...")
    start = time.perf_counter()

    # edge_index
    src = df["nameOrig"].map(node_to_id).values
    dst = df["nameDest"].map(node_to_id).values
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    # edge_attr
    amount_log = np.log1p(df["amount"].values).astype(np.float32)
    step_norm = (df["step"].values / MAX_STEP_NORMALIZATION).astype(np.float32)

    # One-hot encoding du type
    type_dummies = pd.get_dummies(df["type"])
    for col in TRANSACTION_TYPES:
        if col not in type_dummies:
            type_dummies[col] = 0
    type_encoded = type_dummies[TRANSACTION_TYPES].values.astype(np.float32)

    is_risky = df["type"].isin(RISKY_TYPES).values.astype(np.float32)

    # Écart de solde émetteur si les colonnes de solde sont présentes
    if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        expected_new = np.where(
            df["type"] == "CASH_IN",
            df["oldbalanceOrg"] + df["amount"],
            df["oldbalanceOrg"] - df["amount"]
        )
        bal_err = np.log1p(np.abs(expected_new - df["newbalanceOrig"].values)).astype(np.float32)
        drained = ((df["oldbalanceOrg"].values > 0) & (df["newbalanceOrig"].values == 0)).astype(np.float32)
    else:
        bal_err = np.zeros(len(df), dtype=np.float32)
        drained = np.zeros(len(df), dtype=np.float32)

    edge_feats = np.column_stack([
        amount_log,
        step_norm,
        type_encoded,
        is_risky,
        bal_err,
        drained
    ])

    edge_feature_names = (
        ["amount_log", "step_norm"]
        + [f"type_{t}" for t in TRANSACTION_TYPES]
        + ["is_risky_type", "orig_balance_error_log", "is_account_drained"]
    )

    edge_attr = torch.tensor(edge_feats, dtype=torch.float32)
    y_edge = torch.tensor(df["isFraud"].values, dtype=torch.float32)

    elapsed = time.perf_counter() - start
    print(f"      edge_index : {list(edge_index.shape)} ({edge_index.shape[1]:,} arêtes)")
    print(f"      edge_attr  : {list(edge_attr.shape)} ({len(edge_feature_names)} features, {elapsed:.2f} s)")
    print(f"      y (labels) : {list(y_edge.shape)} ({(y_edge == 1).sum().item():,} fraudes)")

    return edge_index, edge_attr, y_edge, edge_feature_names


def build_paysim_graph(file_path: str = "data/paysim_sample.csv", 
                       nrows: Optional[int] = None) -> Tuple[Data, Dict]:
    """Assemble et retourne l'objet torch_geometric.data.Data et ses métadonnées."""
    df = load_dataset(file_path, nrows=nrows)
    node_to_id, id_to_node = build_node_mappings(df)
    x, node_feature_names = extract_node_features(df, node_to_id, id_to_node)
    edge_index, edge_attr, y_edge, edge_feature_names = extract_edges_and_features(df, node_to_id)

    # Label au niveau nœud : 1 si le compte a été impliqué dans au moins une fraude
    fraud_src = set(edge_index[0, y_edge == 1].tolist())
    fraud_dst = set(edge_index[1, y_edge == 1].tolist())
    fraud_nodes = fraud_src.union(fraud_dst)

    y_node = torch.zeros(x.shape[0], dtype=torch.float32)
    if fraud_nodes:
        y_node[list(fraud_nodes)] = 1.0

    # Création du conteneur PyTorch Geometric
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_edge
    )
    graph.y_node = y_node  # label au niveau nœud
    graph.num_nodes = x.shape[0]

    metadata = {
        "num_nodes": graph.num_nodes,
        "num_edges": edge_index.shape[1],
        "node_features": node_feature_names,
        "edge_features": edge_feature_names,
        "num_fraud_edges": int((y_edge == 1).sum().item()),
        "fraud_edge_ratio": float((y_edge == 1).sum().item() / max(1, len(y_edge))),
        "num_fraud_nodes": int((y_node == 1).sum().item()),
        "source_file": file_path,
        "sample_size": len(df)
    }

    return graph, metadata


def save_graph(graph: Data, metadata: Dict, 
               output_graph_path: str = "data/paysim_graph.pt",
               output_meta_path: str = "data/paysim_graph_metadata.json") -> None:
    """Sauvegarde le graphe et ses métadonnées."""
    print("[5/5] Sauvegarde du graphe PyG...")
    os.makedirs(os.path.dirname(os.path.abspath(output_graph_path)), exist_ok=True)
    
    torch.save(graph, output_graph_path)
    print(f"      Graphe enregistré : {output_graph_path} ({os.path.getsize(output_graph_path) / 1024 / 1024:.2f} Mo)")

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"      Métadonnées : {output_meta_path}")


def print_graph_summary(graph: Data, metadata: Dict) -> None:
    """Affiche un résumé complet des caractéristiques du graphe."""
    print("\n" + "=" * 65)
    print("    RÉSUMÉ DU GRAPHE PAYSIM (PyTorch Geometric)")
    print("=" * 65)
    print(f"  • Nombre de nœuds (comptes)        : {metadata['num_nodes']:,}")
    print(f"  • Nombre d'arêtes (transactions)   : {metadata['num_edges']:,}")
    print(f"  • Caractéristiques nœuds (x)       : {graph.x.shape[1]} dimensions")
    print(f"  • Caractéristiques arêtes (attr)   : {graph.edge_attr.shape[1]} dimensions")
    print(f"  • Transactions frauduleuses (edges): {metadata['num_fraud_edges']:,} ({metadata['fraud_edge_ratio']*100:.3f} %)")
    print(f"  • Comptes impliqués dans la fraude : {metadata['num_fraud_nodes']:,}")
    print(f"  • Densité du graphe                : {metadata['num_edges'] / max(1, metadata['num_nodes']):.4f} arêtes/nœud")
    print(f"  • Objet PyG valide                 : {graph.validate()}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constructeur du graphe PaySim PyTorch Geometric")
    parser.add_argument("--input", default="data/paysim_sample.csv", help="Chemin du CSV PaySim")
    parser.add_argument("--nrows", type=int, default=None, help="Nombre max de lignes à charger")
    parser.add_argument("--output", default="data/paysim_graph.pt", help="Chemin de sortie du graphe PyG (.pt)")
    parser.add_argument("--meta", default="data/paysim_graph_metadata.json", help="Chemin de sortie des métadonnées (.json)")
    args = parser.parse_args()

    graph, metadata = build_paysim_graph(file_path=args.input, nrows=args.nrows)
    save_graph(graph, metadata, output_graph_path=args.output, output_meta_path=args.meta)
    print_graph_summary(graph, metadata)


if __name__ == "__main__":
    main()

