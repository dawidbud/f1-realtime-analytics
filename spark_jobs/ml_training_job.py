"""
PySpark ML pipeline: predict tire wear percentage from telemetry features.

Reads the Parquet telemetry data produced by streaming_job.py, trains a
RandomForestRegressor, and saves the fitted pipeline model to disk.

Run inside the spark cluster, e.g.:
  docker compose exec spark-master \
    spark-submit /opt/spark_jobs/ml_training_job.py
"""

import os

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession

INPUT_DIR = os.environ.get("INPUT_DIR", "/opt/data/processed/telemetry_agg")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/data/models/tire_wear_rf")

FEATURE_COLS = [
    "speed_kmh",
    "throttle_pct",
    "brake_pct",
    "gear",
    "rpm",
    "lap",
    "distance_m",
    "tire_age_laps",
]
LABEL_COL = "tire_wear_pct"


def main():
    spark = SparkSession.builder.appName("f1-ml-training").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(INPUT_DIR).dropna(subset=FEATURE_COLS + [LABEL_COL])

    compound_indexer = StringIndexer(
        inputCol="tire_compound", outputCol="tire_compound_idx", handleInvalid="keep"
    )
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS + ["tire_compound_idx"], outputCol="features"
    )
    rf = RandomForestRegressor(
        featuresCol="features", labelCol=LABEL_COL, numTrees=100, maxDepth=8
    )

    pipeline = Pipeline(stages=[compound_indexer, assembler, rf])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Test RMSE: {rmse:.3f}")

    model.write().overwrite().save(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
