"""
Spark batch: buduje tabele cech "kierowca x wyscig" do predykcji wynikow.

Wejscie: data/raw/fastf1 (z ingest/ingest_fastf1.py).
Wyjscie: data/processed/race_features - jeden wiersz na kierowce i runde, dla
wszystkich rozegranych wyscigow (z finish_position) oraz dla kolejnej rundy
(is_future = true, finish_position = null) - to ja przewiduje model.

Wszystkie cechy "formy" sa liczone wylacznie z rund POPRZEDZAJACYCH dany wyscig,
wiec nie ma wycieku informacji z przyszlosci. Cechy biezacej rundy (pole startowe,
kwalifikacje) sa znane przed startem wyscigu.

Uruchomienie:
  docker compose exec spark-master spark-submit /opt/spark_jobs/race_features_job.py
"""

import os

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

RAW_DIR = os.environ.get("RAW_DIR", "/opt/data/raw/fastf1")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/data/processed/race_features")
SEASON = int(os.environ.get("SEASON", "2026"))
FORM_WINDOW = 3  # liczba poprzednich wyscigow do cech "formy"
MIN_CLEAN_LAPS = 5

# Cechy biezacej rundy, znane dopiero po kwalifikacjach.
POST_QUALI_FEATURES = ["grid_position", "quali_position", "quali_gap_pct"]
# Cechy formy - znane przed weekendem.
FORM_FEATURES = [
    "drv_races", "drv_avg_finish_3", "drv_avg_grid_3", "drv_avg_gain_3",
    "drv_avg_pace_gap_3", "drv_avg_quali_gap_3", "drv_dnf_rate", "drv_points",
    "team_avg_finish_3", "team_avg_pace_gap_3", "team_avg_quali_3", "team_dnf_rate",
]


def load_race_results(spark):
    results = spark.read.parquet(f"{RAW_DIR}/results").filter(F.col("season") == SEASON)
    race = results.filter(F.col("session") == "R").select(
        F.col("round").cast("int"),
        F.col("Abbreviation").alias("driver"),
        F.col("TeamName").alias("team"),
        F.col("Position").cast("double").alias("finish_position"),
        F.col("GridPosition").cast("double").alias("grid_position"),
        F.col("Points").cast("double").alias("points"),
        (~F.col("ClassifiedPosition").rlike("^[0-9]+$")).cast("double").alias("dnf"),
    )
    quali = results.filter(F.col("session") == "Q").select(
        F.col("round").cast("int"),
        F.col("Abbreviation").alias("driver"),
        F.col("TeamName").alias("quali_team"),
        F.col("Position").cast("double").alias("quali_position"),
        F.least("Q1", "Q2", "Q3").alias("quali_best_s"),
    )
    by_round = Window.partitionBy("round")
    quali = quali.withColumn(
        "quali_gap_pct", F.col("quali_best_s") / F.min("quali_best_s").over(by_round) - 1
    ).drop("quali_best_s")
    return race, quali


def load_race_pace(spark):
    """Tempo wyscigowe: mediana czystych okrazen wzgledem najszybszego kierowcy w rundzie."""
    laps = spark.read.parquet(f"{RAW_DIR}/laps").filter(
        (F.col("season") == SEASON) & (F.col("session") == "R")
    )
    clean = laps.filter(
        F.col("LapTime").isNotNull()
        & F.col("IsAccurate")
        & (F.col("TrackStatus") == "1")  # zielona flaga
        & ~F.col("IsPitIn")
        & ~F.col("IsPitOut")
        & (F.col("LapNumber") > 1)
        & ~F.coalesce(F.col("Deleted"), F.lit(False))
    )
    pace = (
        clean.groupBy(F.col("round").cast("int").alias("round"), F.col("Driver").alias("driver"))
        .agg(
            F.percentile_approx("LapTime", 0.5).alias("median_lap_s"),
            F.count("*").alias("clean_laps"),
        )
        .filter(F.col("clean_laps") >= MIN_CLEAN_LAPS)
    )
    by_round = Window.partitionBy("round")
    return pace.withColumn(
        "pace_gap_pct", F.col("median_lap_s") / F.min("median_lap_s").over(by_round) - 1
    ).select("round", "driver", "pace_gap_pct")


def next_round_skeleton(spark, race, quali, last_round):
    """Wiersze dla kolejnej rundy: sklad z kwalifikacji (jesli juz byly) albo z ostatniego wyscigu."""
    schedule = spark.read.parquet(f"{RAW_DIR}/schedule").filter(F.col("season") == SEASON)
    upcoming = schedule.filter(F.col("RoundNumber") > last_round).agg(F.min("RoundNumber")).first()[0]
    if upcoming is None:
        return None, None
    upcoming = int(upcoming)

    next_quali = quali.filter(F.col("round") == upcoming)
    if next_quali.count() > 0:
        lineup = next_quali.select("driver", F.col("quali_team").alias("team"))
    else:
        lineup = race.filter(F.col("round") == last_round).select("driver", "team")
    return upcoming, lineup.withColumn("round", F.lit(upcoming))


def main():
    spark = (
        SparkSession.builder.appName("f1-race-features")
        .config("spark.sql.shuffle.partitions", "4")  # malo danych - 200 partycji to sam narzut
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    race, quali = load_race_results(spark)
    pace = load_race_pace(spark)
    last_round = race.agg(F.max("round")).first()[0]

    base = race.withColumn("is_future", F.lit(False))
    upcoming, lineup = next_round_skeleton(spark, race, quali, last_round)
    if lineup is not None:
        base = base.unionByName(lineup.withColumn("is_future", F.lit(True)), allowMissingColumns=True)

    df = (
        base.join(quali.drop("quali_team"), ["round", "driver"], "left")
        .join(pace, ["round", "driver"], "left")
        # Czy kwalifikacje tej rundy juz sie odbyly (dla przyszlej rundy moga jeszcze nie byc).
        .withColumn("has_quali", ~F.col("is_future") | F.col("quali_position").isNotNull())
        # Brak czasu w kwalifikacjach = ostatnie pole.
        .withColumn(
            "quali_position",
            F.when(F.col("has_quali"), F.coalesce("quali_position", F.lit(22.0))),
        )
        # Dla przyszlej rundy pole startowe = pozycja z kwalifikacji (kary nieznane).
        .withColumn(
            "grid_position",
            F.when(F.col("is_future"), F.col("quali_position")).otherwise(F.col("grid_position")),
        )
        .withColumn("gain", F.col("grid_position") - F.col("finish_position"))
    )

    # --- forma kierowcy: tylko poprzednie rundy ---
    drv_last = Window.partitionBy("driver").orderBy("round").rowsBetween(-FORM_WINDOW, -1)
    drv_all = Window.partitionBy("driver").orderBy("round").rowsBetween(Window.unboundedPreceding, -1)
    df = (
        df.withColumn("drv_races", F.count("finish_position").over(drv_all).cast("double"))
        .withColumn("drv_avg_finish_3", F.avg("finish_position").over(drv_last))
        .withColumn("drv_avg_grid_3", F.avg("grid_position").over(drv_last))
        .withColumn("drv_avg_gain_3", F.avg("gain").over(drv_last))
        .withColumn("drv_avg_pace_gap_3", F.avg("pace_gap_pct").over(drv_last))
        .withColumn("drv_avg_quali_gap_3", F.avg("quali_gap_pct").over(drv_last))
        .withColumn("drv_dnf_rate", F.avg("dnf").over(drv_all))
        .withColumn("drv_points", F.coalesce(F.sum("points").over(drv_all), F.lit(0.0)))
    )

    # --- forma zespolu: agregat na runde, potem okno po poprzednich rundach ---
    team_round = df.groupBy("round", "team").agg(
        F.avg("finish_position").alias("t_finish"),
        F.avg("pace_gap_pct").alias("t_pace_gap"),
        F.avg("quali_position").alias("t_quali"),
        F.avg("dnf").alias("t_dnf"),
    )
    team_last = Window.partitionBy("team").orderBy("round").rowsBetween(-FORM_WINDOW, -1)
    team_all = Window.partitionBy("team").orderBy("round").rowsBetween(Window.unboundedPreceding, -1)
    team_round = team_round.select(
        "round", "team",
        F.avg("t_finish").over(team_last).alias("team_avg_finish_3"),
        F.avg("t_pace_gap").over(team_last).alias("team_avg_pace_gap_3"),
        F.avg("t_quali").over(team_last).alias("team_avg_quali_3"),
        F.avg("t_dnf").over(team_all).alias("team_dnf_rate"),
    )
    df = df.join(team_round, ["round", "team"], "left")

    # Braki (np. pierwsze rundy, debiutanci) uzupelniamy srednia z calej tabeli.
    fill_cols = FORM_FEATURES + POST_QUALI_FEATURES
    means = df.agg(*[F.avg(c).alias(c) for c in fill_cols]).first().asDict()
    df = df.fillna({c: float(v) for c, v in means.items() if v is not None})

    out = df.select(
        F.lit(SEASON).alias("season"), "round", "driver", "team", "is_future", "has_quali",
        *POST_QUALI_FEATURES, *FORM_FEATURES, "finish_position", "dnf",
    ).orderBy("round", "grid_position")
    out.write.mode("overwrite").parquet(OUTPUT_DIR)

    completed = out.filter(~F.col("is_future")).count()
    print(f"Zapisano {completed} wierszy z rund 1-{last_round} do {OUTPUT_DIR}")
    if upcoming is not None:
        print(f"Kolejna runda do predykcji: {upcoming} ({lineup.count()} kierowcow)")


if __name__ == "__main__":
    main()
