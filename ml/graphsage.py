"""
Mission 1 - Semaine 3 : Entraînement GraphSAGE et extraction d'embeddings (64 dimensions)

Ce module entraîne un encodeur GraphSAGE (PyTorch Geometric) sur le graphe PaySim
généré par `ml/graph_builder.py` afin de produire des représentations vectorielles
de 64 dimensions pour chaque compte (`nameOrig` / `nameDest`).

Utilisation :
    # Entraînement + extraction des embeddings 64d
    python ml/graphsage.py --graph data/paysim_graph.pt --epochs 50

    # Extraction seule depuis un checkpoint existant
    python ml/graphsage.py --extract-only --graph data/paysim_graph.pt --checkpoint ml/checkpoints/graphsage_best.pt
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """Encodeur GraphSAGE à 2 couches produisant des embeddings de nœuds (64-dim par défaut)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h


class EdgeFraudClassifier(nn.Module):
    """Classifieur de fraude sur arêtes combinant embeddings GraphSAGE et edge_attr."""

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: int = 128,
        emb_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=emb_dim,
            dropout=dropout,
        )
        head_in = 2 * emb_dim + edge_channels
        self.head = nn.Sequential(
            nn.Linear(head_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        msg_edge_index: torch.Tensor,
        target_edge_index: torch.Tensor,
        target_edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_emb = self.encoder(x, msg_edge_index)
        src_emb = node_emb[target_edge_index[0]]
        dst_emb = node_emb[target_edge_index[1]]
        edge_repr = torch.cat([src_emb, dst_emb, target_edge_attr], dim=-1)
        logits = self.head(edge_repr).squeeze(-1)
        return logits, node_emb


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(graph_path: str) -> Data:
    """Charge le graphe PyTorch Geometric depuis le disque."""
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graphe introuvable : {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    if not isinstance(graph, Data):
        raise TypeError(f"Objet inattendu dans {graph_path} : {type(graph)}")
    return graph


def split_edges(
    y: torch.Tensor,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crée un split stratifié train / val / test sur les indices d'arêtes."""
    num_edges = y.shape[0]
    indices = np.arange(num_edges)
    y_np = y.cpu().numpy().astype(int)

    stratify_full = y_np if len(np.unique(y_np)) > 1 and np.bincount(y_np).min() >= 2 else None
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify_full,
    )

    rel_test = test_ratio / (val_ratio + test_ratio)
    y_temp = y_np[temp_idx]
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 and np.bincount(y_temp).min() >= 2 else None
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        random_state=seed,
        stratify=stratify_temp,
    )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def scale_features(
    x: torch.Tensor,
    edge_attr: torch.Tensor,
    train_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, StandardScaler, StandardScaler]:
    """Standardise les features des nœuds et des arêtes (fit sur train uniquement pour edge_attr)."""
    x_scaler = StandardScaler()
    x_scaled = torch.tensor(
        x_scaler.fit_transform(x.cpu().numpy()),
        dtype=torch.float32,
    )

    edge_scaler = StandardScaler()
    edge_np = edge_attr.cpu().numpy()
    edge_scaler.fit(edge_np[train_idx.cpu().numpy()])
    edge_scaled = torch.tensor(
        edge_scaler.transform(edge_np),
        dtype=torch.float32,
    )
    return x_scaled, edge_scaled, x_scaler, edge_scaler


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calcule AUC-ROC, PR-AUC, Rappel et F1 sur un split."""
    preds = (probs >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) > 1
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)) if has_both_classes else 0.0,
        "pr_auc": float(average_precision_score(y_true, probs)) if has_both_classes else 0.0,
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }


@torch.no_grad()
def evaluate(
    model: EdgeFraudClassifier,
    x: torch.Tensor,
    msg_edge_index: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
    split_idx: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    logits, _ = model(
        x,
        msg_edge_index,
        edge_index[:, split_idx],
        edge_attr[split_idx],
    )
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y[split_idx].cpu().numpy().astype(int)
    return compute_metrics(y_true, probs)


def train_graphsage(
    graph: Data,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    emb_dim: int = 64,
    dropout: float = 0.3,
    out_model: str = "ml/checkpoints/graphsage_best.pt",
    seed: int = 42,
) -> Tuple[EdgeFraudClassifier, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Entraîne GraphSAGE sur les arêtes du graphe et sauvegarde le meilleur checkpoint."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/4] Entraînement GraphSAGE sur {device} ({graph.num_nodes:,} nœuds, {graph.edge_index.shape[1]:,} arêtes)")

    train_idx, val_idx, test_idx = split_edges(graph.y, seed=seed)
    x_scaled, edge_scaled, x_scaler, edge_scaler = scale_features(
        graph.x, graph.edge_attr, train_idx
    )

    x_dev = x_scaled.to(device)
    edge_index_dev = graph.edge_index.to(device)
    edge_attr_dev = edge_scaled.to(device)
    y_dev = graph.y.to(device)

    train_idx_dev = train_idx.to(device)
    val_idx_dev = val_idx.to(device)
    test_idx_dev = test_idx.to(device)

    # Message passing uniquement sur les arêtes d'entraînement pendant l'apprentissage
    msg_edge_index = edge_index_dev[:, train_idx_dev]

    model = EdgeFraudClassifier(
        in_channels=x_dev.shape[1],
        edge_channels=edge_attr_dev.shape[1],
        hidden_channels=hidden_dim,
        emb_dim=emb_dim,
        dropout=dropout,
    ).to(device)

    y_train = y_dev[train_idx_dev]
    pos_count = float((y_train == 1).sum().item())
    neg_count = float((y_train == 0).sum().item())
    pos_weight = torch.tensor([neg_count / max(1.0, pos_count)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(os.path.dirname(os.path.abspath(out_model)), exist_ok=True)
    best_val_auc = -1.0
    best_state: Optional[Dict] = None

    print(f"[2/4] Début des {epochs} époques (pos_weight={pos_weight.item():.2f})...")
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits, _ = model(
            x_dev,
            msg_edge_index,
            edge_index_dev[:, train_idx_dev],
            edge_attr_dev[train_idx_dev],
        )
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(
            model, x_dev, msg_edge_index, edge_index_dev, edge_attr_dev, y_dev, val_idx_dev
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_state = {
                "encoder_state_dict": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "in_channels": x_dev.shape[1],
                "edge_channels": edge_attr_dev.shape[1],
                "hidden_dim": hidden_dim,
                "emb_dim": emb_dim,
                "dropout": dropout,
                "x_mean": x_scaler.mean_,
                "x_scale": x_scaler.scale_,
                "best_val_auc": best_val_auc,
            }
            torch.save(best_state, out_model)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"      Époque {epoch:03d}/{epochs:03d} | loss={loss.item():.4f} | "
                f"val_auc={val_metrics['roc_auc']:.4f} | val_pr_auc={val_metrics['pr_auc']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f}"
            )

    elapsed = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])
        model.to(device)

    test_metrics = evaluate(
        model, x_dev, msg_edge_index, edge_index_dev, edge_attr_dev, y_dev, test_idx_dev
    )
    print(f"[3/4] Entraînement terminé en {elapsed:.1f}s — Checkpoint : {out_model}")
    print(
        f"      Test | AUC-ROC={test_metrics['roc_auc']:.4f} | "
        f"PR-AUC={test_metrics['pr_auc']:.4f} | "
        f"Rappel={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f}"
    )

    return model, x_scaled, edge_index_dev.cpu(), test_metrics


@torch.no_grad()
def extract_embeddings(
    encoder: GraphSAGEEncoder,
    x_scaled: torch.Tensor,
    edge_index: torch.Tensor,
    id_to_node: Optional[List[str]],
    out_embeddings: str = "data/node_embeddings_64d.parquet",
) -> torch.Tensor:
    """Extrait les embeddings [num_nodes, 64] et les sauvegarde (.pt + .parquet/.csv)."""
    print("[4/4] Extraction des embeddings 64 dimensions...")
    encoder.eval()
    device = next(encoder.parameters()).device
    embeddings = encoder(x_scaled.to(device), edge_index.to(device)).cpu()

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Les embeddings extraits contiennent des NaN ou Inf.")

    num_nodes, emb_dim = embeddings.shape
    if id_to_node is None or len(id_to_node) != num_nodes:
        print("      Attention : id_to_node absent du graphe, utilisation d'IDs génériques.")
        id_to_node = [f"node_{i}" for i in range(num_nodes)]

    os.makedirs(os.path.dirname(os.path.abspath(out_embeddings)), exist_ok=True)
    pt_path = os.path.splitext(out_embeddings)[0] + ".pt"
    torch.save({"embeddings": embeddings, "id_to_node": id_to_node}, pt_path)
    print(f"      Tenseur PyTorch sauvegardé : {pt_path} {list(embeddings.shape)}")

    cols = [f"emb_{i}" for i in range(emb_dim)]
    df_emb = pd.DataFrame(embeddings.numpy(), columns=cols)
    df_emb.insert(0, "account_id", id_to_node)

    if out_embeddings.endswith(".parquet"):
        try:
            df_emb.to_parquet(out_embeddings, index=False)
            print(f"      Parquet sauvegardé        : {out_embeddings} ({df_emb.shape})")
        except ImportError:
            csv_fallback = os.path.splitext(out_embeddings)[0] + ".csv"
            df_emb.to_csv(csv_fallback, index=False)
            print(f"      pyarrow/fastparquet absent, CSV sauvegardé : {csv_fallback}")
    else:
        df_emb.to_csv(out_embeddings, index=False)
        print(f"      CSV sauvegardé            : {out_embeddings} ({df_emb.shape})")

    return embeddings


def load_encoder_from_checkpoint(checkpoint_path: str) -> Tuple[GraphSAGEEncoder, np.ndarray, np.ndarray]:
    """Recharge un GraphSAGEEncoder et les paramètres de normalisation depuis un checkpoint."""
    ckpt = torch.load(checkpoint_path, weights_only=False)
    encoder = GraphSAGEEncoder(
        in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["hidden_dim"],
        out_channels=ckpt["emb_dim"],
        dropout=ckpt["dropout"],
    )
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return encoder, ckpt["x_mean"], ckpt["x_scale"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement GraphSAGE & extraction embeddings 64d")
    parser.add_argument("--graph", default="data/paysim_graph.pt", help="Chemin du graphe PyG (.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage Adam")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Dimension cachée GraphSAGE")
    parser.add_argument("--emb-dim", type=int, default=64, help="Dimension des embeddings de sortie")
    parser.add_argument("--dropout", type=float, default=0.3, help="Taux de dropout")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--out-model", default="ml/checkpoints/graphsage_best.pt", help="Chemin du checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint à charger avec --extract-only")
    parser.add_argument("--out-embeddings", default="data/node_embeddings_64d.parquet", help="Chemin d'export des embeddings")
    parser.add_argument("--extract-only", action="store_true", help="Extrait les embeddings sans réentraîner")
    args = parser.parse_args()

    graph = load_graph(args.graph)
    id_to_node = getattr(graph, "id_to_node", None)

    if args.extract_only:
        ckpt_path = args.checkpoint or args.out_model
        encoder, x_mean, x_scale = load_encoder_from_checkpoint(ckpt_path)
        x_np = (graph.x.cpu().numpy() - x_mean) / np.where(x_scale == 0, 1.0, x_scale)
        x_scaled = torch.tensor(x_np, dtype=torch.float32)
        extract_embeddings(encoder, x_scaled, graph.edge_index, id_to_node, args.out_embeddings)
    else:
        model, x_scaled, edge_index_cpu, _ = train_graphsage(
            graph=graph,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            out_model=args.out_model,
            seed=args.seed,
        )
        extract_embeddings(
            model.encoder.cpu(),
            x_scaled,
            edge_index_cpu,
            id_to_node,
            args.out_embeddings,
        )


if __name__ == "__main__":
    main()
