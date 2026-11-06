
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, DoubleType, StringType
)

# 1. Initialiser Spark
spark = (
    SparkSession.builder
    .appName("PaySim-CSV-to-Parquet")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. Configurer MinIO avec S3A
conf = spark.sparkContext._jsc.hadoopConfiguration()

conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("fs.s3a.endpoint", "http://fraud-minio:9000")
conf.set("fs.s3a.endpoint.region", "us-east-1")
conf.set("fs.s3a.path.style.access", "true")
conf.set("fs.s3a.connection.ssl.enabled", "false")

conf.set(
    "fs.s3a.aws.credentials.provider",
    "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider"
)

conf.set("fs.s3a.access.key", os.environ["MINIO_ROOT_USER"])
conf.set("fs.s3a.secret.key", os.environ["MINIO_ROOT_PASSWORD"])

# 3. Définir les types des colonnes
schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True),
])

# 4. Lire le CSV historique depuis MinIO
input_path = "s3a://paysim-historical/paysim_historical.csv"

df = (
    spark.read
    .option("header", "true")
    .schema(schema)
    .csv(input_path)
)

# 5. Prendre 10 000 transactions pour le test
sample = df.limit(10000)

# 6. Écrire au format Parquet dans MinIO
output_path = "s3a://paysim-historical/parquet_test/"

print("\nÉcriture Parquet en cours...")

(
    sample.write
    .mode("overwrite")
    .parquet(output_path)
)

# 7. Vérifier que Spark peut relire le fichier
result = spark.read.parquet(output_path)

print("\n=== VERIFICATION PARQUET ===")
print("Nombre de lignes :", result.count())

result.printSchema()
result.show(5, truncate=False)

print("\nConversion CSV vers Parquet réussie !")

spark.stop()
