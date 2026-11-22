"""Features de graphe « point-in-time » et branchement des embeddings GNN.

Pourquoi : la baseline ne regarde que le montant et les soldes d'une transaction isolée.
Le réseau de comptes (qui envoie à qui) apporte une information différente :
un compte destinataire qui a déjà reçu beaucoup d'argent, ou qui réémet ce qu'il reçoit
(profil « mule »), est plus suspect. C'est la version simple de ce qu'apprendra GraphSAGE.

Règle anti-fuite (point-in-time) : pour une transaction au step s, on ne compte QUE les
transactions des steps STRICTEMENT antérieurs (< s). Jamais la transaction elle-même,
jamais le futur. L'ordre des transactions à l'intérieur d'un même step étant inconnu,
le même step est exclu par prudence. En production, c'est ce que servirait un graph store
mis à jour en continu.

Features (calculées sur TOUTES les transactions, tous types, historique + flux) :
  g_dest_in_count      nb de transactions déjà reçues par le compte destinataire
  g_dest_in_amount_log log1p(montant total déjà reçu par le destinataire)
  g_dest_out_count     nb de transactions déjà émises par le destinataire (profil mule)
  g_orig_in_count      nb de transactions déjà reçues par l'émetteur
  g_orig_out_count     nb de transactions déjà émises par l'émetteur

Embeddings GNN (semaine 3, Maryam) : fichier parquet avec une colonne `account` et des
colonnes emb_0..emb_63 ; voir attach_embeddings().
"""

import time

import numpy as np
import pandas as pd

GRAPH_FEATURES = [
    "g_dest_in_count", "g_dest_in_amount_log", "g_dest_out_count",
    "g_orig_in_count", "g_orig_out_count",
]
SPLIT_FILES = ["paysim_historical.csv", "paysim_streaming.csv"]
RAW_COLUMNS = ["step", "amount", "nameOrig", "nameDest"]


def prior_cumulative(event_account, event_step, event_value, query_account, query_step):
    """Pour chaque requête (compte a, step s) : somme des event_value du compte a aux steps < s.

    Les événements sont agrégés par (compte, step), cumulés dans l'ordre des steps, puis
    chaque requête prend le dernier cumul d'un step STRICTEMENT inférieur (merge_asof avec
    allow_exact_matches=False). Les comptes sans événement antérieur valent 0.
    """
    events = pd.DataFrame({"account": event_account, "step": event_step.astype(np.int32),
                           "value": event_value})
    table = events.groupby(["account", "step"], sort=True)["value"].sum()
    table = table.groupby(level=0).cumsum().reset_index().sort_values("step", kind="stable")

    queries = pd.DataFrame({"account": query_account, "step": query_step.astype(np.int32),
                            "qid": np.arange(len(query_account))}).sort_values("step", kind="stable")
    merged = pd.merge_asof(queries, table, on="step", by="account",
                           allow_exact_matches=False, direction="backward")
    return merged.sort_values("qid")["value"].fillna(0.0).to_numpy()


def compute_graph_features(frames):
    """frames : dict {nom: DataFrame avec step, amount, nameOrig, nameDest}.

    Le graphe est construit sur l'union des frames (même timeline) ; renvoie un dict
    {nom: DataFrame des GRAPH_FEATURES, même index que la frame d'origine}.
    """
    keys = list(frames)
    all_tx = pd.concat([frames[k][RAW_COLUMNS] for k in keys], ignore_index=True)
    sizes = [len(frames[k]) for k in keys]

    # Identifiants texte -> entiers (même code pour un compte, qu'il émette ou reçoive)
    codes, _ = pd.factorize(pd.concat([all_tx["nameOrig"], all_tx["nameDest"]], ignore_index=True))
    orig, dest = codes[:len(all_tx)], codes[len(all_tx):]
    step = all_tx["step"].to_numpy()
    ones = np.ones(len(all_tx))
    amount = all_tx["amount"].to_numpy()

    feats = pd.DataFrame({
        # reçus par le destinataire / émis par le destinataire
        "g_dest_in_count": prior_cumulative(dest, step, ones, dest, step),
        "g_dest_in_amount_log": np.log1p(prior_cumulative(dest, step, amount, dest, step)),
        "g_dest_out_count": prior_cumulative(orig, step, ones, dest, step),
        # reçus par l'émetteur / émis par l'émetteur
        "g_orig_in_count": prior_cumulative(dest, step, ones, orig, step),
        "g_orig_out_count": prior_cumulative(orig, step, ones, orig, step),
    })

    out, start = {}, 0
    for key, size in zip(keys, sizes):
        part = feats.iloc[start:start + size].copy()
        part.index = frames[key].index
        out[key] = part
        start += size
    return out


def build_graph_feature_cache(data_dir, force=False):
    """Calcule (une fois) les features de graphe des deux fichiers et les met en cache parquet."""
    caches = {f: data_dir / f"graph_features_{f.replace('.csv', '')}.parquet" for f in SPLIT_FILES}
    if not force and all(p.is_file() for p in caches.values()):
        return caches
    start = time.perf_counter()
    print("Computing point-in-time graph features (one-off, cached)...")
    frames = {f: pd.read_csv(data_dir / f, usecols=RAW_COLUMNS) for f in SPLIT_FILES}
    feats = compute_graph_features(frames)
    for f, path in caches.items():
        feats[f].astype(np.float32).to_parquet(path)
    print(f"  done in {time.perf_counter() - start:.0f} s -> {', '.join(p.name for p in caches.values())}")
    return caches


def attach_graph_features(df, filename, data_dir):
    """Ajoute les GRAPH_FEATURES à df (index = numéro de ligne dans le CSV d'origine)."""
    cache = build_graph_feature_cache(data_dir)[filename]
    feats = pd.read_parquet(cache)
    return df.join(feats, how="left")


# ---------------------------------------------------------------------------
# Embeddings GNN (GraphSAGE) : à brancher dès que Maryam fournit le fichier
# ---------------------------------------------------------------------------
def embedding_columns(emb):
    return [c for c in emb.columns if c.startswith("emb_")]


def attach_embeddings_frame(df, name_orig, name_dest, emb):
    """Ajoute les embeddings de l'émetteur (eo_*) et du destinataire (ed_*) à df.

    emb : DataFrame avec `account` + emb_0..emb_k. Un compte absent (démarrage à froid :
    pas vu dans le graphe d'historique) reçoit un vecteur nul et known_* = 0.
    """
    cols = embedding_columns(emb)
    table = emb.set_index("account")[cols].astype(np.float32)
    out = df.copy()
    for prefix, names in (("eo", name_orig), ("ed", name_dest)):
        values = table.reindex(pd.Index(names))
        out[f"known_{prefix}"] = values.notna().all(axis=1).to_numpy().astype(np.int8)
        values = values.fillna(0.0).to_numpy()
        for j, col in enumerate(cols):
            out[f"{prefix}_{col}"] = values[:, j]
    return out


def attach_embeddings(df, filename, data_dir, emb_path):
    """Version fichier : relit nameOrig/nameDest du CSV (alignés par index) puis attache."""
    emb = pd.read_parquet(emb_path)
    names = pd.read_csv(data_dir / filename, usecols=["nameOrig", "nameDest"]).loc[df.index]
    return attach_embeddings_frame(df, names["nameOrig"].to_numpy(), names["nameDest"].to_numpy(), emb)


def embedding_feature_names(emb_path):
    cols = embedding_columns(pd.read_parquet(emb_path, columns=None).head(1))
    return ["known_eo", "known_ed"] + [f"eo_{c}" for c in cols] + [f"ed_{c}" for c in cols]
