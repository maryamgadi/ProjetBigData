
import os
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("PaySim-MinIO-S3A")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# Configuration de la connexion MinIO
hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

hadoop_conf.set(
    "fs.s3a.impl",
    "org.apache.hadoop.fs.s3a.S3AFileSystem"
)
hadoop_conf.set("fs.s3a.endpoint", "http://fraud-minio:9000")
hadoop_conf.set("fs.s3a.endpoint.region", "us-east-1")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

hadoop_conf.set(
    "fs.s3a.aws.credentials.provider",
    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
)

hadoop_conf.set(
    "fs.s3a.access.key",
    os.environ["MINIO_ROOT_USER"]
)
hadoop_conf.set(
    "fs.s3a.secret.key",
    os.environ["MINIO_ROOT_PASSWORD"]
)

# Lecture directe depuis le bucket MinIO
path = "s3a://paysim-historical/paysim_historical.csv"

print("\nConnexion à MinIO...")

df = (
    spark.read
    .option("header", "true")
    .csv(path)
)

print("\n=== PREMIERES TRANSACTIONS DEPUIS MINIO ===")
df.select(
    "step", "type", "amount",
    "nameOrig", "nameDest", "isFraud"
).show(5, truncate=False)

print("\nLecture S3A réussie !")

spark.stop()
