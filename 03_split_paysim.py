
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

input_file = (
    BASE_DIR / "data" / "archive"
    / "PS_20174392719_1491204439457_log.csv"
)

output_dir = BASE_DIR / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

print("Lecture de PaySim...")
df = pd.read_csv(input_file)

# Séparation chronologique par heure simulée
steps = sorted(df["step"].unique())
split_index = int(len(steps) * 0.70)
cutoff_step = steps[split_index]

historical = df[df["step"] < cutoff_step]
streaming = df[df["step"] >= cutoff_step]

historical_file = output_dir / "paysim_historical.csv"
streaming_file = output_dir / "paysim_streaming.csv"

historical.to_csv(historical_file, index=False)
streaming.to_csv(streaming_file, index=False)

print("\nDécoupage terminé !")
print("Step de séparation :", cutoff_step)
print("Historique :", historical.shape)
print("Streaming :", streaming.shape)
print("Total :", len(historical) + len(streaming))

assert len(historical) + len(streaming) == len(df)
assert historical["step"].max() < streaming["step"].min()

print("\nFraudes dans l'historique :")
print(historical["isFraud"].value_counts())

print("\nFraudes dans le streaming :")
print(streaming["isFraud"].value_counts())

print("\nPériodes :")
print("Historique :", historical["step"].min(),
      "à", historical["step"].max())

print("Streaming :", streaming["step"].min(),
      "à", streaming["step"].max())