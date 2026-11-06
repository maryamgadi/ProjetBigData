
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent

source = (
    base_dir / "data" / "archive"
    / "PS_20174392719_1491204439457_log.csv"
)

destination = base_dir / "data" / "processed" / "sample_1000.csv"
destination.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, nrows=1000)
df.to_csv(destination, index=False)

print("Échantillon créé :", df.shape)
print("Fichier :", destination)
