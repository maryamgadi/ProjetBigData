"""Converts PaySim history (CSV in MinIO) into a Delta table."""

from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType,
)

from spark_session import LAKEHOUSE_BUCKET, build_spark

SOURCE = "s3a://paysim-historical/paysim_historical.csv"
TARGET = f"s3a://{LAKEHOUSE_BUCKET}/bronze/historical"

SCHEMA = StructType([
    StructField("step", IntegerType()),
    StructField("type", StringType()),
    StructField("amount", DoubleType()),
    StructField("nameOrig", StringType()),
    StructField("oldbalanceOrg", DoubleType()),
    StructField("newbalanceOrig", DoubleType()),
    StructField("nameDest", StringType()),
    StructField("oldbalanceDest", DoubleType()),
    StructField("newbalanceDest", DoubleType()),
    StructField("isFraud", IntegerType()),
    StructField("isFlaggedFraud", IntegerType()),
])

spark = build_spark("Historical-to-Delta")

df = spark.read.option("header", "true").schema(SCHEMA).csv(SOURCE)
expected = df.count()

(
    df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("type")          # 5 values: the model reads only TRANSFER/CASH_OUT
    .save(TARGET)
)

written = spark.read.format("delta").load(TARGET).count()
assert written == expected, f"{written} rows written, {expected} expected"
print(f"OK: {written} transactions in {TARGET}")

spark.sql(f"DESCRIBE DETAIL delta.`{TARGET}`").select("numFiles", "sizeInBytes").show()
spark.stop()