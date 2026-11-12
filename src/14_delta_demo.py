"""Demo of Delta Lake features: ACID, Time Travel, schema, compaction."""

from delta.tables import DeltaTable
from pyspark.errors import AnalysisException
from pyspark.sql.functions import lit

from spark_session import BRONZE_TRANSACTIONS, LAKEHOUSE_BUCKET, build_spark

HISTORICAL = f"s3a://{LAKEHOUSE_BUCKET}/bronze/historical"

spark = build_spark("Delta-Demo")


def count(path, version=None):
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", version)
    return reader.load(path).count()


def num_files(path):
    return spark.sql(f"DESCRIBE DETAIL delta.`{path}`").collect()[0]["numFiles"]


table = DeltaTable.forPath(spark, HISTORICAL)
v0 = table.history(1).collect()[0]["version"]
n0 = count(HISTORICAL)
print(f"\n[0] Version {v0}: {n0} rows")

# 1. ACID transaction: a DELETE creates a new version, all or nothing
table.delete("type = 'DEBIT'")
print(f"[1] After DELETE DEBIT: {count(HISTORICAL)} rows")

# 2. Time Travel: re-read the table as it was before the DELETE
print(f"[2] Time Travel version {v0}: {count(HISTORICAL, v0)} rows")
assert count(HISTORICAL, v0) == n0

# 3. Restore: undo the DELETE (itself a new version)
table.restoreToVersion(v0)
print(f"[3] After RESTORE: {count(HISTORICAL)} rows")
assert count(HISTORICAL) == n0

# 4. Schema enforcement: Delta rejects a write with an unknown column
bad = spark.read.format("delta").load(HISTORICAL).limit(1).withColumn("unknown_column", lit(1))
try:
    bad.write.format("delta").mode("append").save(HISTORICAL)
    print("[4] PROBLEM: the bad schema was accepted")
except AnalysisException:
    print("[4] Schema enforcement: write correctly rejected")

# 5. Compaction: streaming creates many small files, OPTIMIZE merges them
before = num_files(BRONZE_TRANSACTIONS)
DeltaTable.forPath(spark, BRONZE_TRANSACTIONS).optimize().executeCompaction()
print(f"[5] OPTIMIZE bronze/transactions: {before} -> {num_files(BRONZE_TRANSACTIONS)} files")

print("\n=== History (audit) ===")
table.history(5).select("version", "timestamp", "operation").show(truncate=False)
spark.stop()