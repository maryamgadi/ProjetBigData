
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / "infra" / ".env")

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    region_name="us-east-1",
)

expected = {
    "paysim-raw":
        "PS_20174392719_1491204439457_log.csv",
    "paysim-historical":
        "paysim_historical.csv",
    "paysim-streaming":
        "paysim_streaming.csv",
}

for bucket, filename in expected.items():
    response = s3.head_object(
        Bucket=bucket,
        Key=filename,
    )

    print(f"\nBucket : {bucket}")
    print(f"Fichier : {filename}")
    print(f"Taille : {response['ContentLength']} octets")

print("\nTous les objets attendus sont accessibles !")
