#!/usr/bin/env bash
# Lance un job PySpark dans le conteneur fraud-spark avec les connecteurs
# Kafka, S3A (MinIO) et Delta Lake.
# Usage : docker compose exec spark /workspace/scripts/run_spark.sh src/11_stream_kafka_to_delta.py [args]
set -euo pipefail

exec /opt/spark/bin/spark-submit \
  --master "local[${SPARK_CORES:-2}]" \
  --driver-memory "${SPARK_DRIVER_MEMORY:-1g}" \
  --packages "${SPARK_PACKAGES}" \
  "$@"
