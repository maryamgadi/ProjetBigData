
import os
from pathlib import Path

import boto3
from botocore.client import Config
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "infra" / ".env")

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:9000",
    aws_access_key_id=os.environ["MINIO_ROOT_USER"],
    aws_secret_access_key=os.environ["MINIO_ROOT_PASSWORD"],
    region_name="us-east-1",
    config=Config(signature_version="s3v4"),
)

files = {
    "paysim-historical": "paysim_historical.csv",
    "paysim-streaming": "paysim_streaming.csv",
}

existing_buckets = {
    bucket["Name"]
    for bucket in s3.list_buckets()["Buckets"]
}

for bucket, filename in files.items():
    path = BASE_DIR / "data" / "processed" / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    if bucket not in existing_buckets:
        s3.create_bucket(Bucket=bucket)

    print(f"\nEnvoi de {filename} vers {bucket}...")

    s3.upload_file(
        str(path),
        bucket,
        filename,
    )

    metadata = s3.head_object(
        Bucket=bucket,
        Key=filename,
    )

    assert metadata["ContentLength"] == path.stat().st_size

    print("Upload vérifié :", bucket)
    print("Taille :", metadata["ContentLength"], "octets")

print("\nTous les fichiers sont stockés dans MinIO !")
