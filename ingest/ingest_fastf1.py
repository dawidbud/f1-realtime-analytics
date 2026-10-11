"""
Pobiera dane sezonu F1 z FastF1 i zapisuje je jako Parquet do data/raw/fastf1.

Uklad (partycje w stylu Hive, czytelne bezposrednio przez Spark):
  data/raw/fastf1/schedule/season=2026/data.parquet
  data/raw/fastf1/results/season=2026/round=16/session=R/data.parquet
  data/raw/fastf1/laps/season=2026/round=16/session=R/data.parquet
  data/raw/fastf1/weather/season=2026/round=16/session=R/data.parquet

Pobierane sesje: Q (kwalifikacje), S (sprint, jesli jest) i R (wyscig).
Rundy juz pobrane sa pomijane (chyba ze --force), wiec skrypt mozna uruchamiac
po kazdym weekendzie wyscigowym i dociagnie tylko nowe dane.

Uruchomienie:
  docker compose run --rm ingest                 # sezon z SEASON (domyslnie 2026)
  docker compose run --rm ingest --round 17      # tylko jedna runda
  docker compose run --rm ingest --force         # pobierz wszystko od nowa
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RAW_DIR = DATA_DIR / "raw" / "fastf1"
CACHE_DIR = Path(os.environ.get("FASTF1_CACHE", DATA_DIR / "cache" / "fastf1"))
DEFAULT_SEASON = int(os.environ.get("SEASON", "2026"))

RESULT_COLS = [
    "DriverNumber", "Abbreviation", "FullName", "TeamName", "Position",
    "ClassifiedPosition", "GridPosition", "Q1", "Q2", "Q3", "Time", "Status",
    "Points", "Laps",
]
LAP_COLS = [
    "Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife",
    "FreshTyre", "PitInTime", "PitOutTime", "Sector1Time", "Sector2Time",
    "Sector3Time", "SpeedST", "TrackStatus", "Position", "IsAccurate", "Deleted",
]
WEATHER_COLS = ["Time", "AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall", "WindSpeed"]

log = logging.getLogger("ingest")


def _timedeltas_to_seconds(df):
    """Spark nie obsluguje typu duration z Arrow - zamieniamy na sekundy (float)."""
    for col in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].dt.total_seconds()
    return df


def _write(df, table, season, round_no=None, session=None):
    path = RAW_DIR / table / f"season={season}"
    if round_no is not None:
        path = path / f"round={round_no}" / f"session={session}"
    path.mkdir(parents=True, exist_ok=True)
    # Spark nie czyta timestampow w nanosekundach (domyslnych w pandas).
    df.to_parquet(path / "data.parquet", index=False, coerce_timestamps="us", allow_truncated_timestamps=True)


def _already_ingested(season, round_no, session):
    return (RAW_DIR / "results" / f"season={season}" / f"round={round_no}"
            / f"session={session}" / "data.parquet").exists()


def ingest_session(season, round_no, session_name):
    session = fastf1.get_session(season, round_no, session_name)
    with_laps = session_name in ("R", "S")
    session.load(laps=with_laps, telemetry=False, weather=session_name == "R", messages=False)

    results = session.results
    if results is None or results.empty or results["Position"].isna().all():
        log.info("  %s: brak wynikow (sesja sie jeszcze nie odbyla?) - pomijam", session_name)
        return False

    results = results[RESULT_COLS].copy()
    for col in ("Position", "GridPosition", "Points", "Laps"):
        results[col] = pd.to_numeric(results[col], errors="coerce")
    results["ClassifiedPosition"] = results["ClassifiedPosition"].astype(str)
    _write(_timedeltas_to_seconds(results), "results", season, round_no, session_name)

    if with_laps:
        laps = pd.DataFrame(session.laps)[LAP_COLS].copy()
        laps["IsPitIn"] = laps["PitInTime"].notna()
        laps["IsPitOut"] = laps["PitOutTime"].notna()
        laps = laps.drop(columns=["PitInTime", "PitOutTime"])
        for col in ("FreshTyre", "IsAccurate", "Deleted"):
            laps[col] = laps[col].astype("boolean")
        laps["TrackStatus"] = laps["TrackStatus"].astype(str)
        _write(_timedeltas_to_seconds(laps), "laps", season, round_no, session_name)

    if session_name == "R":
        weather = session.weather_data[WEATHER_COLS].copy()
        weather["Rainfall"] = weather["Rainfall"].astype(bool)
        _write(_timedeltas_to_seconds(weather), "weather", season, round_no, session_name)

    log.info("  %s: zapisano (%d kierowcow)", session_name, len(results))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=DEFAULT_SEASON)
    parser.add_argument("--round", type=int, dest="round_no", help="tylko ta runda")
    parser.add_argument("--force", action="store_true", help="nadpisz juz pobrane rundy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    schedule = fastf1.get_event_schedule(args.season, include_testing=False)
    schedule_out = schedule[["RoundNumber", "EventName", "Country", "Location", "EventDate", "EventFormat"]].copy()
    _write(schedule_out, "schedule", args.season)

    today = datetime.now(timezone.utc).date()
    for _, event in schedule.iterrows():
        round_no = int(event["RoundNumber"])
        if args.round_no is not None and round_no != args.round_no:
            continue
        if event["EventDate"].date() > today:
            continue

        log.info("Runda %d: %s", round_no, event["EventName"])
        sessions = ["Q", "S", "R"] if "sprint" in event["EventFormat"] else ["Q", "R"]
        for session_name in sessions:
            if not args.force and _already_ingested(args.season, round_no, session_name):
                log.info("  %s: juz pobrane - pomijam", session_name)
                continue
            try:
                ingest_session(args.season, round_no, session_name)
            except Exception as exc:  # pojedyncza sesja nie powinna przerywac calego ingestu
                log.warning("  %s: blad pobierania: %s", session_name, exc)


if __name__ == "__main__":
    main()
