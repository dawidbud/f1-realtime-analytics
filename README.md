# F1 Real-Time Analytics & ML

Projekt analizy danych F1:

- **Predykcja wyników wyścigów na prawdziwych danych** – dane sezonu 2026
  z [FastF1](https://github.com/theOehrly/Fast-F1) → Parquet → cechy w Spark →
  model PySpark ML przewidujący pozycję końcową każdego kierowcy w kolejnym wyścigu.
  Zobacz [Predykcja wyników (FastF1)](#predykcja-wyników-fastf1).
- **Streaming w czasie rzeczywistym** – symulowana telemetria płynie przez Kafkę,
  jest przetwarzana przez Spark Structured Streaming, a na zebranych danych
  trenowany jest model zużycia opon.

Całość działa w Dockerze, eksploracja i trening odbywają się też w notebookach Jupyter.

## Struktura projektu

```
F1/
├── docker-compose.yml        # Kafka, Zookeeper, Kafka UI, Spark (master+worker), Jupyter, producer
├── producer/                 # Generator symulowanej telemetrii -> Kafka
│   └── producer.py
├── ingest/
│   └── ingest_fastf1.py      # Pobieranie danych FastF1 -> data/raw/fastf1 (Parquet)
├── spark/
│   └── Dockerfile            # Obraz Sparka (numpy, UID hosta, spark-submit w PATH)
├── spark_jobs/
│   ├── race_features_job.py  # Cechy "kierowca x wyścig" z danych FastF1
│   ├── race_prediction_job.py# Model wyników wyścigu + ewaluacja walk-forward + predykcja
│   ├── streaming_job.py      # Spark Structured Streaming: real-time agregacje
│   └── ml_training_job.py    # PySpark ML: model zużycia opon (dane symulowane)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_realtime_streaming_analysis.ipynb
│   └── 03_pyspark_ml_training.ipynb
├── config/
│   └── config.yaml
└── data/
    ├── raw/
    └── processed/
```

## Szybki start

```bash
cd F1
docker compose up --build
```

Dostępne serwisy:
- Jupyter Lab: http://localhost:8888/lab?token=f1-local
- Kafka UI: http://localhost:8080
- Spark master UI: http://localhost:8081

### Jupyter: token i kernel w VS Code

Jupyter ma **stały token `f1-local`** (ustawiony w `docker-compose.yml`), więc
nie zmienia się po restarcie kontenera. Port jest dostępny tylko z `localhost`.
Własny token ustawisz w pliku `.env` w katalogu projektu: `JUPYTER_TOKEN=twoj-token`
(potem `docker compose up -d jupyter`).

Notebooki trzeba uruchamiać na kernelu **z kontenera** (PySpark i ścieżki
`/opt/data`, `/home/jovyan/data` istnieją tylko tam), nie na lokalnym Pythonie.
W VS Code:

1. Otwórz notebook → **Select Kernel** (prawy górny róg) →
   **Existing Jupyter Server** (lub *Select Another Kernel…* → *Existing Jupyter Server*).
2. Wklej adres:
   ```
   http://localhost:8888/?token=f1-local
   ```
3. Wybierz kernel **Python 3 (ipykernel)**.

Jeśli VS Code „wisi” na kernelu albo w logach Jupytera
(`docker compose logs jupyter`) widać `403 Forbidden`, VS Code używa starego
adresu/tokena: `Ctrl+Shift+P` → **Jupyter: Clear Remote Server List**, otwórz
notebook ponownie i podłącz kernel jak wyżej.

Producer od razu zacznie wysyłać symulowaną telemetrię (prędkość, przepustnica,
hamulec, bieg, obroty, mieszanka i wiek opon, zużycie opon) do topicu `f1-telemetry`.

### Symulacja

- Tor 5 km z 7 zakrętami: prędkość, hamowanie, bieg i obroty wynikają z pozycji
  na torze (strefy hamowania przed zakrętem, przyspieszanie po nim).
- Zużycie opon narasta z każdym metrem i zależy od mieszanki (SOFT > MEDIUM > HARD),
  stylu jazdy kierowcy, hamowania i zakrętów; po ~60% przyspiesza ("klif").
- Zużyte opony obniżają prędkość. Po przekroczeniu progu (65–85%) kierowca
  zjeżdża do boksu: nowa mieszanka, zużycie i `tire_age_laps` od zera.
- Symulacja biegnie 5x szybciej niż w rzeczywistości (`TICK_INTERVAL_S=0.2`),
  żeby szybko zebrać dane z pełnego zakresu zużycia.

### Obraz Sparka

Spark działa na własnym obrazie [spark/Dockerfile](spark/Dockerfile) (bazującym na
`apache/spark:3.5.9-python3`): ma doinstalowany `numpy` (wymagany przez
`pyspark.ml`), `spark-submit` w `PATH`, a użytkownik `spark` ma UID 1000, więc
pliki zapisywane w `data/` należą do użytkownika hosta. Jeśli Twój UID na hoście
jest inny (`id -u`), zmień `SPARK_UID`/`SPARK_GID` w Dockerfile.

## Predykcja wyników (FastF1)

Trzy kroki – po każdym weekendzie wyścigowym (albo po kwalifikacjach) wystarczy
uruchomić je ponownie:

```bash
# 1. Pobierz dane sezonu (dociąga tylko nowe rundy; pierwsze uruchomienie ~kilka minut)
docker compose run --rm ingest

# 2. Zbuduj cechy "kierowca x wyścig"
docker compose exec spark-master spark-submit /opt/spark_jobs/race_features_job.py

# 3. Ewaluacja modelu + predykcja kolejnego wyścigu
docker compose exec spark-master spark-submit /opt/spark_jobs/race_prediction_job.py
```

Predykcja trafia na konsolę i do `data/processed/race_predictions/round=<N>/*.csv`,
model do `data/models/race_position_<wariant>`.

### Dane

`ingest` pobiera dla każdej rozegranej rundy kwalifikacje (Q), sprint (S) i wyścig (R):
wyniki, okrążenia (czas, mieszanka, `TyreLife`, stint, status toru) i pogodę.
Cache FastF1 trzymany jest w `data/cache/fastf1`. Opcje: `--round N` (jedna runda),
`--force` (pobierz od nowa), `--season 2025`.

FastF1 **nie udostępnia zużycia opon w %** – zespoły tego nie publikują. Degradację
opon można natomiast modelować ze spadku tempa wraz z `TyreLife`.

### Cechy

Wszystkie cechy „formy” liczone są wyłącznie z wyścigów *poprzedzających* daną
rundę (brak wycieku z przyszłości):

- **kierowca**: średnia pozycja, pole startowe, zysk pozycji, tempo wyścigowe
  (mediana czystych okrążeń vs najszybszy) i różnica w kwalifikacjach z ostatnich
  3 wyścigów; odsetek DNF i punkty w sezonie,
- **zespół**: średnia pozycja, tempo i kwalifikacje z ostatnich 3 rund; odsetek DNF,
- **po kwalifikacjach**: pole startowe, pozycja i strata do pole position.

### Model

Dwa warianty: **pre_quali** (przed weekendem – sama forma) i **post_quali**
(z wynikami kwalifikacji). Job sam wybiera wariant: jeśli kwalifikacje kolejnej
rundy są już pobrane, używa post_quali.

Regresja liniowa (elastic net) trenowana bez wierszy z DNF; kierowcy sortowani wg
przewidywanej wartości. W porównaniu walk-forward wypadła lepiej niż GBT
i RandomForest – przy ~20 wierszach na rundę prostszy model mniej się przeucza.

Ewaluacja **walk-forward** (trening na rundach 1..k-1, predykcja rundy k) na rundach
6–17 sezonu 2026, w porównaniu z baseline'em:

| wariant | | Spearman | MAE poz. | zwycięzca | podium | top 10 |
|---|---|---|---|---|---|---|
| pre_quali | model | **0.667** | **3.55** | 0.33 | 0.44 | **0.76** |
| | baseline: średnia pozycja z 3 ostatnich | 0.600 | 4.10 | 0.33 | 0.44 | 0.72 |
| post_quali | model | **0.710** | **3.15** | 0.67 | 0.61 | **0.77** |
| | baseline: pole startowe | 0.662 | 3.36 | 0.67 | 0.64 | 0.76 |

Model lepiej odtwarza całą klasyfikację, ale zwycięzcę i podium trafia tak samo
jak sam baseline. To tylko 12 wyścigów testowych – różnice są małe i będą się
zmieniać z każdą rundą.

## Uruchomienie joba strumieniowego (real-time analysis)

```bash
docker compose exec spark-master \
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
  /opt/spark_jobs/streaming_job.py
```

Wyniki agregacji (okno 10s, krok 5s, per kierowca) lecą na konsolę, a surowe
zdarzenia zapisują się jako Parquet w `data/processed/telemetry_agg`.

## Trening modelu ML

Po zebraniu danych z joba strumieniowego:

```bash
docker compose exec spark-master \
  spark-submit /opt/spark_jobs/ml_training_job.py
```

Model (predykcja `tire_wear_pct` na podstawie telemetrii i wieku opon) zapisuje
się w `data/models/tire_wear_rf`. Zbieraj dane przez co najmniej kilka minut,
żeby objęły pełne stinty i pit stopy.

## Notebooki

1. `01_data_exploration.ipynb` – wczytanie i eksploracja danych telemetrycznych.
2. `02_realtime_streaming_analysis.ipynb` – podgląd strumienia Kafka i analiza
   w czasie rzeczywistym bezpośrednio z poziomu notebooka.
3. `03_pyspark_ml_training.ipynb` – budowa i ewaluacja pipeline'u PySpark ML.
