"""Création d'une SparkSession configurée pour MinIO (S3A) et Delta Lake.

Les connecteurs (jars) sont chargés par scripts/run_spark.sh via --packages ;
ce module ne fait que la configuration.
"""

import os

from pyspark.sql import SparkSession

LAKEHOUSE_BUCKET = "fraud-lakehouse"
BRONZE_TRANSACTIONS = f"s3a://{LAKEHOUSE_BUCKET}/bronze/transactions"
BRONZE_CHECKPOINT = f"s3a://{LAKEHOUSE_BUCKET}/_checkpoints/bronze_transactions"

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = "financial-transactions"


def build_spark(app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        # --- MinIO via S3A ---
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.endpoint", os.getenv("MINIO_ENDPOINT", "http://fraud-minio:9000"))
        .config("spark.hadoop.fs.s3a.endpoint.region", "us-east-1")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.access.key", os.environ["MINIO_ROOT_USER"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["MINIO_ROOT_PASSWORD"])
        # --- Delta Lake ---
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Petits micro-batchs en local : peu de partitions de shuffle
        .config("spark.sql.shuffle.partitions", "4")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
