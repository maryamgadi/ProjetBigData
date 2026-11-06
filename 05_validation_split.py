
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

historical = pd.read_csv(
    DATA_DIR / "paysim_historical.csv"
)

streaming = pd.read_csv(
    DATA_DIR / "paysim_streaming.csv"
)

for name, df in [
    ("Historique", historical),
    ("Streaming", streaming)
]:
    frauds = df["isFraud"].sum()
    fraud_rate = df["isFraud"].mean() * 100

    print(f"\n--- {name} ---")
    print("Transactions :", len(df))
    print("Fraudes :", frauds)
    print(f"Taux de fraude : {fraud_rate:.4f}%")
    print("Step min :", df["step"].min())
    print("Step max :", df["step"].max())

assert historical["step"].max() < streaming["step"].min()

print("\nSéparation chronologique validée !")
