import pandas as pd
import os

input_file = "data/paysim.csv"
output_file = "data/paysim_sample.csv"

if os.path.exists(input_file):
    print("Chargement du dataset PaySim...")
    # On charge les colonnes nécessaires
    df = pd.read_csv(input_file)
    
    # On prend toutes les fraudes existantes + un échantillon de transactions normales
    frauds = df[df['isFraud'] == 1]
    non_frauds = df[df['isFraud'] == 0].sample(n=100000 - len(frauds), random_state=42)
    
    sample = pd.concat([frauds, non_frauds]).sort_values(by='step')
    sample.to_csv(output_file, index=False)
    print(f"Échantillon créé avec succès : {output_file} ({len(sample)} lignes, dont {len(frauds)} fraudes)")
else:
    print(f"Fichier {input_file} non trouvé. Télécharge-le sur Kaggle.")