# Mission 2 — Modélisation du Graphe PaySim sous PyTorch Geometric (PyG)

Ce module convertit les flux de transactions PaySim en un graphe orienté hétérogène sous le format standard `torch_geometric.data.Data`.

```
Transactions PaySim (CSV/Parquet)
      │
      ▼
Indexation des comptes (nameOrig, nameDest ➔ ID entiers 0..N-1)
      │
      ├─► x : Matrice de caractéristiques des nœuds [N, 11]
      ├─► edge_index : Connexions orientées source ➔ cible [2, E]
      ├─► edge_attr  : Attributs des transactions [E, 10]
      └─► y          : Labels de fraude isFraud [E]
      │
      ▼
Sauvegarde sous data/paysim_graph.pt (Prêt pour GraphSAGE en Semaine 3)
```

## Structure des Caractéristiques

### Nœuds / Comptes (`x` : 11 dimensions)
1. `is_merchant` : 1 si compte Marchand (`M...`), 0 si Client (`C...`)
2. `out_degree` : nombre de transactions envoyées
3. `in_degree` : nombre de transactions reçues
4. `total_degree` : `in_degree + out_degree`
5. `out_amount_sum_log` : volume total envoyé ($\log(1 + x)$)
6. `in_amount_sum_log` : volume total reçu ($\log(1 + x)$)
7. `out_amount_mean_log` : montant moyen envoyé ($\log(1 + x)$)
8. `in_amount_mean_log` : montant moyen reçu ($\log(1 + x)$)
9. `is_source_only` : 1 si le compte n'a fait qu'émettre des fonds
10. `is_dest_only` : 1 si le compte n'a fait que recevoir des fonds
11. `is_mule_candidate` : 1 si le compte reçoit ET réémet des fonds (profil type mule financière)

### Arêtes / Transactions (`edge_attr` : 10 dimensions)
1. `amount_log` : $\log(1 + \text{amount})$
2. `hour_norm` : heure de la journée, $(\text{step} \bmod 24) / 23$
3-7. Encodage One-Hot des types : `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`
8. `is_risky_type` : 1 si `TRANSFER` ou `CASH_OUT`
9. `amount_to_balance_log` : $\log(1 + \text{amount} / (\text{oldbalanceOrg} + 1))$
10. `is_account_drained` : 1 si `amount == oldbalanceOrg` (le compte est vidé exactement)

### Anti-fuite de données
- **Coupure temporelle** : par défaut le graphe ne contient que l'historique (`step <= 520`,
  même coupure que `03_split_paysim.py`). Sinon les degrés, volumes et labels des nœuds
  incluraient des transactions de la période de test et GraphSAGE « verrait le futur ».
- **Pas de soldes après transaction** (`newbalanceOrig`, `newbalanceDest`) : le score est
  produit avant validation, ces valeurs n'existent pas encore. Dans PaySim elles trahissent
  la fraude (AUC artificielle de 1,0 avec la baseline LightGBM).
- **Pas de step brut** : les steps de test (521–743) ne sont jamais vus à l'entraînement ;
  seule l'heure de la journée généralise.
- Pour la période de test, un compte absent du graphe d'historique n'a pas d'embedding
  (démarrage à froid) : prévoir un vecteur nul + un indicateur `known_in_graph`.

### Cibles / Labels
- `data.y` : label de fraude transactionnelle (`isFraud` sur chaque arête).
- `data.y_node` : indicateur binaire si le compte a participé à une transaction frauduleuse.

---

## Utilisation

### Générer le graphe depuis l'échantillon :
```powershell
python ml/graph_builder.py --input data/paysim_sample.csv
```

### Options disponibles :
* `--input` : chemin du fichier source (CSV ou Parquet).
* `--nrows` : nombre de lignes à charger (pour tests rapides).
* `--output` : chemin de sortie du fichier PyG (défaut : `data/paysim_graph.pt`).
* `--meta` : chemin du résumé JSON (défaut : `data/paysim_graph_metadata.json`).
* `--max-step` : dernier step gardé (défaut `520` = historique ; `-1` = tout, déconseillé).

### Graphe complet d'historique (pour GraphSAGE) :
```powershell
python ml/graph_builder.py --input data/processed/paysim_historical.csv
```

### Lancer les tests unitaires :
```powershell
python -m unittest tests/test_graph_builder.py -v
```
