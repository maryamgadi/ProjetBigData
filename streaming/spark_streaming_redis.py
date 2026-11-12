"""
Mission 1 - Semaine 2 : Spark Structured Streaming (Kafka) -> Redis Feature Store

Pipeline :
    Kafka "financial-transactions"
        -> parsing JSON + features par transaction
        -> 3 agrégations par compte (nameOrig) sur fenêtres glissantes 5 min / 1 h / 24 h
        -> 1 flux "dernière transaction" par compte
        -> écriture Redis (foreachBatch + foreachPartition + pipeline, TTL)

Lancement :
    python streaming/spark_streaming_redis.py
Variables d'environnement utiles : KAFKA_BOOTSTRAP, KAFKA_TOPIC, REDIS_HOST, REDIS_PORT,
    STARTING_OFFSETS (latest|earliest), CHECKPOINT_DIR, TRIGGER_INTERVAL, WATERMARK_DELAY
"""

import os
import sys

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import redis_sink as rs  # noqa: E402

# ---------------------------------------------------------------------------
# Étape 1 : Dépendances & configuration
# ---------------------------------------------------------------------------
SPARK_VERSION = "3.5.3"  # doit correspondre à pyspark dans requirements.txt
SCALA_VERSION = "2.12"
KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}"

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "financial-transactions")
STARTING_OFFSETS = os.getenv("STARTING_OFFSETS", "latest")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints/spark_redis")
TRIGGER_INTERVAL = os.getenv("TRIGGER_INTERVAL", "10 seconds")
WATERMARK_DELAY = os.getenv("WATERMARK_DELAY", "2 minutes")
MAX_OFFSETS_PER_TRIGGER = os.getenv("MAX_OFFSETS_PER_TRIGGER", "20000")

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def configure_hadoop_on_windows() -> None:
    """Spark exige winutils.exe + hadoop.dll sous Windows (cf. scripts/setup_winutils.ps1)."""
    if os.name != "nt":
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_hadoop = os.path.join(project_root, "tools", "hadoop")
    if not os.environ.get("HADOOP_HOME") and os.path.isfile(
        os.path.join(local_hadoop, "bin", "winutils.exe")
    ):
        os.environ["HADOOP_HOME"] = local_hadoop
    hadoop_home = os.environ.get("HADOOP_HOME")
    if not hadoop_home:
        sys.exit("[ERREUR] HADOOP_HOME absent : lancez "
                 "'powershell -ExecutionPolicy Bypass -File scripts/setup_winutils.ps1'")
    hadoop_bin = os.path.join(hadoop_home, "bin")
    if hadoop_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")


configure_hadoop_on_windows()

# ---------------------------------------------------------------------------
# Étape 2 : Schéma JSON envoyé par producer/producer.py
# ---------------------------------------------------------------------------
TRANSACTION_SCHEMA = StructType([
    StructField("step", IntegerType()),
    StructField("type", StringType()),
    StructField("amount", DoubleType()),
    StructField("nameOrig", StringType()),
    StructField("oldbalanceOrg", DoubleType()),
    StructField("newbalanceOrig", DoubleType()),
    StructField("nameDest", StringType()),
    StructField("oldbalanceDest", DoubleType()),
    StructField("newbalanceDest", DoubleType()),
    StructField("isFraud", IntegerType()),  # label : jamais utilisé comme feature
])

RISKY_TYPES = ("TRANSFER", "CASH_OUT")


def create_spark_session() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("PaySim-Streaming-Redis")
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.jars.packages", KAFKA_PACKAGE)
        # Mémoire : 1 Go par défaut en local -> OutOfMemoryError avec 4 requêtes stateful
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "3g"))
        # État des fenêtres dans RocksDB (hors heap Java) au lieu de HashMaps en mémoire
        .config(
            "spark.sql.streaming.stateStore.providerClass",
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        )
        .config("spark.sql.streaming.stateStore.rocksdb.changelogCheckpointing.enabled", "true")
        .config("spark.sql.shuffle.partitions", "3")  # = nb de partitions du topic
        .config("spark.sql.session.timeZone", "UTC")
        # Limiter l'historique conservé par l'UI Spark (sinon le driver grossit à chaque batch)
        .config("spark.sql.ui.retainedExecutions", "20")
        .config("spark.ui.retainedJobs", "50")
        .config("spark.ui.retainedStages", "50")
        .config("spark.sql.streaming.numRecentProgressUpdates", "20")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    # Rend redis_sink importable par les workers Python
    spark.sparkContext.addPyFile(rs.__file__)
    return spark


# ---------------------------------------------------------------------------
# Étape 3 : Lecture Kafka + features par transaction
# ---------------------------------------------------------------------------
def read_transactions(spark: SparkSession) -> DataFrame:
    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", STARTING_OFFSETS)
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", MAX_OFFSETS_PER_TRIGGER)  # borne la taille d'un batch
        .load()
    )

    # event_time = horodatage Kafka du message (heure d'envoi par le producteur)
    tx = (
        raw.select(
            F.col("timestamp").alias("event_time"),
            F.col("offset").alias("kafka_offset"),
            F.from_json(F.col("value").cast("string"), TRANSACTION_SCHEMA).alias("tx"),
        )
        .select("event_time", "kafka_offset", "tx.*")
        .filter(F.col("nameOrig").isNotNull() & F.col("amount").isNotNull())
    )

    # Écart entre le solde attendu et le solde réel de l'émetteur (signal fort sur PaySim)
    expected_new_balance = F.when(
        F.col("type") == "CASH_IN", F.col("oldbalanceOrg") + F.col("amount")
    ).otherwise(F.col("oldbalanceOrg") - F.col("amount"))

    return (
        tx
        .withColumn("is_risky_type", F.col("type").isin(*RISKY_TYPES).cast("int"))
        .withColumn(
            "is_account_drained",
            ((F.col("oldbalanceOrg") > 0) & (F.col("newbalanceOrig") == 0)).cast("int"),
        )
        .withColumn("balance_error_orig", expected_new_balance - F.col("newbalanceOrig"))
    )


# ---------------------------------------------------------------------------
# Étape 4 : Agrégations sur fenêtres glissantes
# ---------------------------------------------------------------------------
def windowed_features(tx: DataFrame, size: str, slide: str) -> DataFrame:
    return (
        tx.withWatermark("event_time", WATERMARK_DELAY)
        .groupBy(F.window("event_time", size, slide), F.col("nameOrig"))
        .agg(
            F.count(F.lit(1)).alias("tx_count"),
            F.round(F.sum("amount"), 2).alias("amount_sum"),
            F.round(F.max("amount"), 2).alias("amount_max"),
            F.round(F.avg("amount"), 2).alias("amount_avg"),
            F.sum("is_risky_type").alias("risky_tx_count"),
            F.sum("is_account_drained").alias("drained_count"),
            F.approx_count_distinct("nameDest").alias("distinct_dest_count"),
            F.max("event_time").alias("last_event_time"),
        )
    )


# ---------------------------------------------------------------------------
# Étape 5 : Écriture Redis (foreachBatch)
# ---------------------------------------------------------------------------
def make_window_batch_writer(label: str, ttl_seconds: int):
    """
    Une transaction appartient à plusieurs fenêtres glissantes. Pour chaque compte on
    garde celle qui couvre le plus d'historique jusqu'à sa dernière transaction :
    la fenêtre de fin minimale parmi celles qui se terminent après cette transaction.
    """
    prefix = rs.window_prefix(label)

    def write_batch(batch_df: DataFrame, batch_id: int) -> None:
        per_account = Window.partitionBy("nameOrig")
        latest = (
            batch_df
            .withColumn("acc_last_event", F.max("last_event_time").over(per_account))
            .filter(F.col("window.end") > F.col("acc_last_event"))
            .withColumn("rn", F.row_number().over(per_account.orderBy(F.col("window.end").asc())))
            .filter(F.col("rn") == 1)
            .select(
                "nameOrig",
                *rs.WINDOW_FEATURES,
                F.col("window.start").alias("window_start"),
                F.col("window.end").alias("window_end"),
                "last_event_time",
            )
        )
        latest.foreachPartition(rs.make_partition_writer(prefix, ttl_seconds))

    return write_batch


def write_last_transaction(batch_df: DataFrame, batch_id: int) -> None:
    """Hash account:{nameOrig} = dernière transaction connue du compte."""
    latest_first = Window.partitionBy("nameOrig").orderBy(
        F.col("event_time").desc(), F.col("kafka_offset").desc()
    )
    latest = (
        batch_df
        .withColumn("rn", F.row_number().over(latest_first))
        .filter(F.col("rn") == 1)
        .select(
            "nameOrig",
            F.col("type").alias("last_type"),
            F.round("amount", 2).alias("last_amount"),
            F.col("step").alias("last_step"),
            F.col("nameDest").alias("last_dest"),
            F.round("balance_error_orig", 2).alias("last_balance_error_orig"),
            F.col("event_time").alias("last_event_time"),
        )
    )
    latest.foreachPartition(rs.make_partition_writer(rs.ACCOUNT_PREFIX, rs.ACCOUNT_TTL_SECONDS))


def start_queries(tx: DataFrame) -> list:
    queries = []
    for label, (size, slide, ttl) in rs.WINDOWS.items():
        queries.append(
            windowed_features(tx, size, slide)
            .writeStream
            .queryName(f"features_{label}")
            .outputMode("update")
            .foreachBatch(make_window_batch_writer(label, ttl))
            .option("checkpointLocation", f"{CHECKPOINT_DIR}/features_{label}")
            .trigger(processingTime=TRIGGER_INTERVAL)
            .start()
        )
    queries.append(
        tx.writeStream
        .queryName("last_transaction")
        .foreachBatch(write_last_transaction)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/last_transaction")
        .trigger(processingTime=TRIGGER_INTERVAL)
        .start()
    )
    return queries


def main() -> None:
    client = rs.get_client()
    client.ping()
    client.close()
    print(f"[OK] Redis joignable sur {rs.REDIS_HOST}:{rs.REDIS_PORT}")

    spark = create_spark_session()
    print(f"[OK] Spark {spark.version} | Kafka {KAFKA_BOOTSTRAP} topic '{KAFKA_TOPIC}'")

    queries = start_queries(read_transactions(spark))
    print(f"[OK] {len(queries)} requêtes streaming démarrées : "
          + ", ".join(q.name for q in queries))
    print("     Ctrl+C pour arrêter.")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("\nArrêt demandé...")
    finally:
        for q in spark.streams.active:
            q.stop()
        spark.stop()


if __name__ == "__main__":
    main()
