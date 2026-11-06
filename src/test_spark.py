
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, sum as spark_sum

# 1. Démarrer Spark
spark = (
    SparkSession.builder
    .appName("PaySim-First-Test")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

# 2. Lire les transactions
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("/workspace/data/processed/sample_1000.csv")
)

# 3. Afficher les données
print("\n=== APERÇU PAYSIM ===")
df.show(5, truncate=False)

# 4. Vérifier le schéma
print("\n=== SCHÉMA ===")
df.printSchema()

# 5. Calculer des statistiques
print("\n=== ANALYSE PAR TYPE ===")

df.groupBy("type").agg(
    count("*").alias("transactions"),
    avg("amount").alias("montant_moyen"),
    spark_sum("isFraud").alias("fraudes")
).show()

# 6. Arrêter Spark
spark.stop()
