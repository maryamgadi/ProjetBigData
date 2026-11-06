
import pandas as pd
from pathlib import Path

# Chemin du dataset
BASE_DIR = Path(__file__).resolve().parent

csv_path = (
    BASE_DIR
    / "data"
    / "archive"
    / "PS_20174392719_1491204439457_log.csv"
)

# Lecture du dataset
df = pd.read_csv(csv_path)

# Dimensions
print("Dimensions :", df.shape)

# Colonnes
print("\nColonnes :")
print(df.columns.tolist())

# Types de données
print("\nInformations :")
df.info()

# Valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Types de transactions
print("\nTransactions par type :")
print(df["type"].value_counts())

# Distribution des fraudes
print("\nDistribution des fraudes :")
print(df["isFraud"].value_counts())

# Aperçu
print("\nPremières transactions :")
print(df.head())
