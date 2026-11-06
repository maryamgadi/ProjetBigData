import json
import time
import argparse
import pandas as pd
from kafka import KafkaProducer

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def run_producer(file_path="data/paysim_sample.csv", topic="financial-transactions", delay=0.01):
    print(f"Connexion au broker Kafka sur localhost:9092...")
    
    # Initialisation du producteur Kafka
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        key_serializer=lambda k: k.encode('utf-8'),      # Clé encodée en UTF-8
        value_serializer=json_serializer,                 # Message encodé en JSON
        acks='all',                                      # Sécurité d'écriture maximale
        retries=3
    )

    print(f"Lecture du fichier : {file_path}")
    df = pd.read_csv(file_path)
    total_tx = len(df)
    print(f"{total_tx} transactions à envoyer dans le topic '{topic}'...")

    sent_count = 0
    for index, row in df.iterrows():
        # Construction du message transactionnel
        transaction = {
            "step": int(row['step']),
            "type": str(row['type']),
            "amount": float(row['amount']),
            "nameOrig": str(row['nameOrig']),
            "oldbalanceOrg": float(row['oldbalanceOrg']),
            "newbalanceOrig": float(row['newbalanceOrig']),
            "nameDest": str(row['nameDest']),
            "oldbalanceDest": float(row['oldbalanceDest']),
            "newbalanceDest": float(row['newbalanceDest']),
            "isFraud": int(row['isFraud'])
        }

        # CRUCIAL : La clé est nameOrig (le compte émetteur) pour le partitionnement
        key = transaction["nameOrig"]

        # Envoi dans Kafka
        producer.send(topic, key=key, value=transaction)
        sent_count += 1

        if sent_count % 1000 == 0:
            print(f"[{sent_count}/{total_tx}] transactions envoyées...")

        if delay > 0:
            time.sleep(delay)

    # Vider le buffer avant de quitter
    producer.flush()
    print(f"Terminé ! Total de {sent_count} transactions envoyées.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Producteur Kafka PaySim")
    parser.add_argument("--file", default="data/paysim_sample.csv", help="Chemin du CSV")
    parser.add_argument("--delay", type=float, default=0.005, help="Délai entre messages (en sec, 0 pour max débit)")
    args = parser.parse_args()

    run_producer(file_path=args.file, delay=args.delay)