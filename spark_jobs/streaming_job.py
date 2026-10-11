"""
Real-time F1 telemetry analysis with Spark Structured Streaming.

Reads telemetry events from Kafka (topic: f1-telemetry), computes rolling
per-driver aggregates (avg speed, max tire wear) over a sliding window,
and writes the results both to the console and to a Parquet sink.

Run inside the spark cluster, e.g.:
  docker compose exec spark-master \
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    /opt/spark_jobs/streaming_job.py
"""

import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, avg, max as spark_max
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "f1-telemetry")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/opt/data/checkpoints")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/data/processed/telemetry_agg")

TELEMETRY_SCHEMA = StructType(
    [
        StructField("driver_id", StringType()),
        StructField("team", StringType()),
        StructField("lap", IntegerType()),
        StructField("distance_m", IntegerType()),
        StructField("speed_kmh", DoubleType()),
        StructField("throttle_pct", DoubleType()),
        StructField("brake_pct", DoubleType()),
        StructField("gear", IntegerType()),
        StructField("rpm", IntegerType()),
        StructField("tire_compound", StringType()),
        StructField("tire_age_laps", IntegerType()),
        StructField("tire_wear_pct", DoubleType()),
        StructField("timestamp", StringType()),
    ]
)


def main():
    spark = (
        SparkSession.builder.appName("f1-realtime-analysis")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    events = raw.select(
        from_json(col("value").cast("string"), TELEMETRY_SCHEMA).alias("data")
    ).select("data.*").withColumn("event_time", col("timestamp").cast("timestamp"))

    agg = (
        events.withWatermark("event_time", "10 seconds")
        .groupBy(
            window(col("event_time"), "10 seconds", "5 seconds"),
            col("driver_id"),
            col("team"),
        )
        .agg(
            avg("speed_kmh").alias("avg_speed_kmh"),
            spark_max("speed_kmh").alias("max_speed_kmh"),
            spark_max("tire_wear_pct").alias("max_tire_wear_pct"),
        )
    )

    console_query = (
        agg.writeStream.outputMode("update")
        .format("console")
        .option("truncate", False)
        .start()
    )

    parquet_query = (
        events.writeStream.format("parquet")
        .option("path", OUTPUT_DIR)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .outputMode("append")
        .start()
    )

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
