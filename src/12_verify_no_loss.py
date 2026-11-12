"""Vérifie que la table Delta bronze contient tous les messages Kafka, sans doublon.

Compare, partition par partition, les offsets présents dans le topic
`financial-transactions` et ceux archivés dans la table Delta bronze.
À lancer quand le producteur a fini et que le job 11 a rattrapé le topic
(par exemple après `11_stream_kafka_to_delta.py --available-now`).

Lancement :
  docker compose exec spark /workspace/scripts/run_spark.sh src/12_verify_no_loss.py
Code de sortie 0 si OK, 1 si perte, doublon ou message illisible.
"""

import sys

from delta.tables import DeltaTable
from pyspark.sql.functions import col, count, countDistinct, max as smax, min as smin

from spark_session import BRONZE_TRANSACTIONS, KAFKA_BOOTSTRAP, KAFKA_TOPIC, build_spark


def main():
    spark = build_spark("Verify-No-Loss")

    # 1. Tous les messages actuellement dans Kafka (lecture batch)
    kafka = (
        spark.read.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("endingOffsets", "latest")
        .load()
        .groupBy(col("partition").alias("kafka_partition"))
        .agg(
            count("*").alias("kafka_msgs"),
            smin("offset").alias("kafka_min"),
            smax("offset").alias("kafka_max"),
        )
    )

    # 2. Ce qui est archivé dans Delta
    bronze = spark.read.format("delta").load(BRONZE_TRANSACTIONS)
    delta = bronze.groupBy("kafka_partition").agg(
        count("*").alias("delta_rows"),
        countDistinct("kafka_offset").alias("delta_distinct"),
        smin("kafka_offset").alias("delta_min"),
        smax("kafka_offset").alias("delta_max"),
    )

    report = (
        kafka.join(delta, "kafka_partition", "full_outer")
        .fillna(0, ["kafka_msgs", "delta_rows", "delta_distinct"])
        .withColumn("missing", col("kafka_msgs") - col("delta_distinct"))
        .withColumn("duplicates", col("delta_rows") - col("delta_distinct"))
        .orderBy("kafka_partition")
    )

    print("\n=== Kafka vs Delta, par partition ===")
    report.show(truncate=False)

    totals = report.agg(
        {"kafka_msgs": "sum", "delta_rows": "sum", "missing": "sum", "duplicates": "sum"}
    ).collect()[0]
    kafka_total = totals["sum(kafka_msgs)"] or 0
    delta_total = totals["sum(delta_rows)"] or 0
    missing = totals["sum(missing)"] or 0
    duplicates = totals["sum(duplicates)"] or 0

    # Messages dont le JSON n'a pas pu être lu (champs obligatoires à null)
    unreadable = bronze.filter(col("nameOrig").isNull() | col("amount").isNull()).count()

    print(f"Messages dans Kafka     : {kafka_total}")
    print(f"Lignes dans Delta       : {delta_total}")
    print(f"Manquants (perte)       : {missing}")
    print(f"Doublons                : {duplicates}")
    print(f"Messages illisibles     : {unreadable}")

    # 3. Historique Delta : chaque micro-batch = une version (Time Travel)
    print("\n=== Historique Delta (5 dernières versions) ===")
    (
        DeltaTable.forPath(spark, BRONZE_TRANSACTIONS)
        .history(5)
        .select("version", "timestamp", "operation", "operationMetrics.numOutputRows")
        .show(truncate=False)
    )

    ok = kafka_total > 0 and missing == 0 and duplicates == 0 and unreadable == 0
    print("RESULTAT :", "OK - aucune perte, aucun doublon" if ok else "ECHEC")
    spark.stop()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
