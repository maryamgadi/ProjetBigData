
from pathlib import Path
import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / "infra" / ".env")

csv_path = (
    BASE_DIR / "data" / "archive"
    / "PS_20174392719_1491204439457_log.csv"
)

bucket_name = "paysim-raw"
object_name = "PS_20174392719_1491204439457_log.csv"

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)

if not csv_path.is_file():
    raise FileNotFoundError(f"CSV introuvable : {csv_path}")

# Créer le bucket s'il n'existe pas
existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]

if bucket_name not in existing:
    s3.create_bucket(Bucket=bucket_name)

print("Début de l'upload vers MinIO...")

s3.upload_file(
    str(csv_path),
    bucket_name,
    object_name,
)

# Vérification du fichier stocké
metadata = s3.head_object(
    Bucket=bucket_name,
    Key=object_name,
)

print("Upload terminé !")
print("Bucket :", bucket_name)
print("Objet :", object_name)
print("Taille :", metadata["ContentLength"], "octets")
