"""
PySpark ML: przewiduje pozycje koncowa kazdego kierowcy w wyscigu.

Wejscie: data/processed/race_features (z race_features_job.py).

Dwa warianty modelu:
  - pre_quali  - tylko forma kierowcy i zespolu (predykcja przed weekendem),
  - post_quali - dodatkowo pole startowe i wynik kwalifikacji.

Model: regresja liniowa z regularyzacja elastic net, trenowana bez wierszy z DNF
(awarie to w duzej mierze losowy szum, ktorego cechy nie wyjasniaja). Przewidywana
wartosc sluzy do posortowania kierowcow, co daje pelna klasyfikacje 1..N.
W walk-forward wypadla lepiej niz GBT i RandomForest - przy ~20 wierszach na
runde prosty model mniej sie przeucza.

Ewaluacja walk-forward: dla kazdej rundy k (od MIN_TRAIN_ROUNDS+1) model jest
trenowany na rundach < k i przewiduje runde k - tak jak w realnym uzyciu.
Wyniki porownywane sa z baseline'em:
  - pre_quali:  kolejnosc wg sredniej pozycji z ostatnich 3 wyscigow,
  - post_quali: kolejnosc wg pola startowego.

Na koniec model trenowany jest na wszystkich rundach i przewiduje kolejna runde
(wariant post_quali, jesli kwalifikacje juz sie odbyly, inaczej pre_quali).

Uruchomienie:
  docker compose exec spark-master spark-submit /opt/spark_jobs/race_prediction_job.py
"""

import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

FEATURES_DIR = os.environ.get("FEATURES_DIR", "/opt/data/processed/race_features")
PREDICTIONS_DIR = os.environ.get("PREDICTIONS_DIR", "/opt/data/processed/race_predictions")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/data/models")
MIN_TRAIN_ROUNDS = int(os.environ.get("MIN_TRAIN_ROUNDS", "5"))
LABEL_COL = "finish_position"

FORM_FEATURES = [
    "drv_races", "drv_avg_finish_3", "drv_avg_grid_3", "drv_avg_gain_3",
    "drv_avg_pace_gap_3", "drv_avg_quali_gap_3", "drv_dnf_rate", "drv_points",
    "team_avg_finish_3", "team_avg_pace_gap_3", "team_avg_quali_3", "team_dnf_rate",
]
VARIANTS = {
    "pre_quali": {"features": FORM_FEATURES, "baseline": "drv_avg_finish_3"},
    "post_quali": {
        "features": ["grid_position", "quali_position", "quali_gap_pct"] + FORM_FEATURES,
        "baseline": "grid_position",
    },
}


def build_pipeline(features):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    # standardization=True (domyslnie) - regularyzacja dziala na cechach w tej samej skali.
    lr = LinearRegression(featuresCol="features", labelCol=LABEL_COL, regParam=1.0, elasticNetParam=0.5)
    return Pipeline(stages=[assembler, lr])


def training_rows(df):
    return df.filter(F.col("dnf") == 0)


def rank_rows(rows, score_key):
    """Sortuje kierowcow jednej rundy wg wyniku -> przewidywana pozycja 1..N."""
    ordered = sorted(rows, key=lambda r: (r[score_key], r["grid_position"]))
    return {r["driver"]: pos for pos, r in enumerate(ordered, start=1)}


def race_metrics(rows, predicted):
    n = len(rows)
    actual = {r["driver"]: int(r[LABEL_COL]) for r in rows}
    d2 = sum((predicted[d] - actual[d]) ** 2 for d in actual)
    top = lambda m, k: {d for d, p in m.items() if p <= k}
    return {
        "spearman": 1 - 6 * d2 / (n * (n**2 - 1)),
        "mae": sum(abs(predicted[d] - actual[d]) for d in actual) / n,
        "winner": float(top(predicted, 1) == top(actual, 1)),
        "podium": len(top(predicted, 3) & top(actual, 3)) / 3,
        "top10": len(top(predicted, 10) & top(actual, 10)) / 10,
    }


def mean_metrics(metrics):
    return {k: sum(m[k] for m in metrics) / len(metrics) for k in metrics[0]}


def walk_forward(df, rounds, variant):
    features, baseline = VARIANTS[variant]["features"], VARIANTS[variant]["baseline"]
    model_metrics, base_metrics = [], []
    for k in rounds[MIN_TRAIN_ROUNDS:]:
        model = build_pipeline(features).fit(training_rows(df.filter(F.col("round") < k)))
        rows = [r.asDict() for r in model.transform(df.filter(F.col("round") == k)).collect()]
        model_metrics.append(race_metrics(rows, rank_rows(rows, "prediction")))
        base_metrics.append(race_metrics(rows, rank_rows(rows, baseline)))
    return mean_metrics(model_metrics), mean_metrics(base_metrics), len(model_metrics)


def print_comparison(variant, model_m, base_m, n_races):
    print(f"\n=== {variant}: walk-forward na {n_races} wyscigach ===")
    print(f"{'metryka':<10}{'model':>10}{'baseline':>10}")
    labels = {
        "spearman": "Spearman", "mae": "MAE poz.", "winner": "zwyciezca",
        "podium": "podium", "top10": "top 10",
    }
    for key, label in labels.items():
        print(f"{label:<10}{model_m[key]:>10.3f}{base_m[key]:>10.3f}")


def print_effects(model, df, features):
    """Wplyw cech: wspolczynnik * odchylenie standardowe = zmiana pozycji przy +1 SD cechy."""
    stds = df.agg(*[F.stddev(c).alias(c) for c in features]).first().asDict()
    coefs = model.stages[-1].coefficients.toArray()
    effects = [(name, coef * (stds[name] or 0.0)) for name, coef in zip(features, coefs)]
    print("\nWplyw cech (zmiana przewidywanej pozycji przy +1 odchyleniu std.; <0 = lepiej):")
    for name, effect in sorted(effects, key=lambda x: -abs(x[1])):
        if abs(effect) >= 0.01:
            print(f"  {name:<22}{effect:+.2f}")


def main():
    spark = (
        SparkSession.builder.appName("f1-race-prediction")
        .config("spark.sql.shuffle.partitions", "4")  # malo danych - 200 partycji to sam narzut
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(FEATURES_DIR).cache()
    history = df.filter(~F.col("is_future"))
    future = df.filter(F.col("is_future"))
    rounds = sorted(r["round"] for r in history.select("round").distinct().collect())

    for variant in VARIANTS:
        model_m, base_m, n_races = walk_forward(history, rounds, variant)
        print_comparison(variant, model_m, base_m, n_races)

    future_rows = future.collect()
    if not future_rows:
        print("\nBrak kolejnej rundy do predykcji (koniec sezonu?).")
        return

    target_round = future_rows[0]["round"]
    variant = "post_quali" if all(r["has_quali"] for r in future_rows) else "pre_quali"
    features = VARIANTS[variant]["features"]
    train = training_rows(history)
    model = build_pipeline(features).fit(train)
    model.write().overwrite().save(f"{MODEL_DIR}/race_position_{variant}")
    print_effects(model, train, features)

    rows = [r.asDict() for r in model.transform(future).collect()]
    predicted = rank_rows(rows, "prediction")
    result = spark.createDataFrame(
        [(target_round, variant, r["driver"], r["team"], predicted[r["driver"]], float(r["prediction"])) for r in rows],
        "round INT, variant STRING, driver STRING, team STRING, predicted_position INT, score DOUBLE",
    ).orderBy("predicted_position")

    print(f"\nPredykcja rundy {target_round} (model {variant}):")
    result.select("predicted_position", "driver", "team", F.round("score", 2).alias("score")).show(30, truncate=False)

    out = f"{PREDICTIONS_DIR}/round={target_round}"
    result.drop("round").coalesce(1).write.mode("overwrite").option("header", True).csv(out)
    print(f"Zapisano predykcje do {out}")


if __name__ == "__main__":
    main()
