import json
import math
import os
import random
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "f1-telemetry")
# Czas rzeczywisty miedzy kolejnymi probkami. Kazda probka to SIM_STEP_S sekund
# symulacji, wiec przy 0.2s symulacja biegnie 5x szybciej niz w rzeczywistosci
# (szybciej zbiera sie dane z pelnego zakresu zuzycia opon).
TICK_INTERVAL_S = float(os.environ.get("TICK_INTERVAL_S", "0.2"))
SIM_STEP_S = 1.0

# "aggression" zwieksza zuzycie opon i predkosc w zakretach, "pace" skaluje predkosc.
DRIVERS = [
    {"driver_id": "VER", "team": "Red Bull", "pace": 1.000, "aggression": 1.10},
    {"driver_id": "HAM", "team": "Mercedes", "pace": 0.992, "aggression": 0.90},
    {"driver_id": "LEC", "team": "Ferrari", "pace": 0.995, "aggression": 1.05},
    {"driver_id": "NOR", "team": "McLaren", "pace": 0.997, "aggression": 1.00},
    {"driver_id": "ALO", "team": "Aston Martin", "pace": 0.985, "aggression": 0.95},
]

TRACK_LENGTH_M = 5000
MAX_SPEED_KMH = 330.0
BRAKE_DECEL_MS2 = 40.0  # ~4g
ACCEL_MS2 = 12.0
# (pozycja apexu [m], predkosc w apexie [km/h])
CORNERS = [
    (450, 95),
    (1200, 180),
    (1750, 130),
    (2600, 250),
    (3300, 85),
    (3900, 160),
    (4600, 120),
]

# Zuzycie opon w % na metr przy neutralnej jezdzie (~3.5 / 2.4 / 1.7 % na okrazenie).
COMPOUND_WEAR_PER_M = {"SOFT": 0.00055, "MEDIUM": 0.00038, "HARD": 0.00027}
# Wplyw mieszanki na przyczepnosc (predkosc w zakretach).
COMPOUND_GRIP = {"SOFT": 1.03, "MEDIUM": 1.0, "HARD": 0.97}

GEAR_TOP_SPEEDS_KMH = [90, 125, 160, 195, 230, 265, 300, 340]
RPM_MIN, RPM_MAX = 10500, 12500


def _kmh_to_ms(v):
    return v / 3.6


def _ms_to_kmh(v):
    return v * 3.6


def target_speed_kmh(distance_m, grip):
    """Profil predkosci: najnizsza z obwiedni hamowania/przyspieszania wokol zakretow.

    Zwraca (predkosc, faza) gdzie faza to "brake", "corner" albo "accel".
    """
    best_v = _kmh_to_ms(MAX_SPEED_KMH)
    phase = "accel"
    for apex_m, apex_kmh in CORNERS:
        v_apex = _kmh_to_ms(apex_kmh * grip)
        # odleglosc do apexu po okregu toru (przed: >0, za: <0)
        delta = (apex_m - distance_m + TRACK_LENGTH_M / 2) % TRACK_LENGTH_M - TRACK_LENGTH_M / 2
        decel = BRAKE_DECEL_MS2 if delta > 0 else ACCEL_MS2
        v = math.sqrt(v_apex**2 + 2 * decel * abs(delta))
        if v < best_v:
            best_v = v
            if abs(delta) < 40:
                phase = "corner"
            elif delta > 0:
                phase = "brake"
            else:
                phase = "accel"
    return _ms_to_kmh(best_v), phase


def gear_and_rpm(speed_kmh):
    low = 0
    for gear, top in enumerate(GEAR_TOP_SPEEDS_KMH, start=1):
        if speed_kmh <= top:
            frac = (speed_kmh - low) / (top - low)
            return gear, int(RPM_MIN + frac * (RPM_MAX - RPM_MIN))
        low = top
    return 8, RPM_MAX


def new_stint(s, previous_compound=None):
    choices = [c for c in COMPOUND_WEAR_PER_M if c != previous_compound]
    s["compound"] = random.choice(choices)
    s["wear"] = 0.0
    s["tire_age_laps"] = 0
    s["pit_threshold"] = random.uniform(65, 85)


def initial_state():
    s = {"lap": 1, "distance_m": random.uniform(0, TRACK_LENGTH_M)}
    new_stint(s)
    # Start w losowym momencie stintu, zeby dane od razu pokrywaly szeroki zakres zuzycia.
    s["tire_age_laps"] = random.randint(0, 15)
    s["wear"] = s["tire_age_laps"] * COMPOUND_WEAR_PER_M[s["compound"]] * TRACK_LENGTH_M * 1.2
    return s


def step(driver, s):
    """Przesuwa bolid o SIM_STEP_S sekund i zwraca zdarzenie telemetrii."""
    grip = COMPOUND_GRIP[s["compound"]]
    v_target, phase = target_speed_kmh(s["distance_m"], grip)

    # Zuzyte opony = mniejsza predkosc (szczegolnie w zakretach).
    wear_penalty = 1 - 0.0012 * s["wear"] * (1.5 if phase != "accel" else 1.0)
    speed = v_target * driver["pace"] * wear_penalty * random.gauss(1.0, 0.01)
    speed = max(60.0, min(speed, MAX_SPEED_KMH + 5))

    if phase == "brake":
        brake = min(100.0, random.gauss(85, 8) * driver["aggression"])
        throttle = 0.0
    elif phase == "corner":
        brake = max(0.0, random.gauss(10, 6))
        throttle = max(0.0, min(100.0, random.gauss(35, 10)))
    else:
        brake = 0.0
        throttle = 100.0 if speed > 0.97 * MAX_SPEED_KMH else max(0.0, min(100.0, random.gauss(85, 10)))

    gear, rpm = gear_and_rpm(speed)
    rpm = int(rpm + random.gauss(0, 80))

    moved_m = _kmh_to_ms(speed) * SIM_STEP_S

    # Zuzycie: baza mieszanki * styl jazdy * obciazenie (hamowanie, zakrety),
    # przyspiesza po przekroczeniu ~60% ("klif" opon).
    load = 1.0 + 0.6 * brake / 100 + (0.5 if phase == "corner" else 0.0)
    cliff = 1.0 + max(0.0, s["wear"] - 60) / 40
    s["wear"] = min(100.0, s["wear"] + COMPOUND_WEAR_PER_M[s["compound"]] * moved_m * load * driver["aggression"] * cliff)

    s["distance_m"] += moved_m
    if s["distance_m"] >= TRACK_LENGTH_M:
        s["distance_m"] -= TRACK_LENGTH_M
        s["lap"] += 1
        s["tire_age_laps"] += 1
        if s["wear"] >= s["pit_threshold"]:
            new_stint(s, previous_compound=s["compound"])

    return {
        "driver_id": driver["driver_id"],
        "team": driver["team"],
        "lap": s["lap"],
        "distance_m": int(s["distance_m"]),
        "speed_kmh": round(speed, 1),
        "throttle_pct": round(throttle, 1),
        "brake_pct": round(brake, 1),
        "gear": gear,
        "rpm": rpm,
        "tire_compound": s["compound"],
        "tire_age_laps": s["tire_age_laps"],
        "tire_wear_pct": round(min(100.0, max(0.0, s["wear"] + random.gauss(0, 0.3))), 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def connect_with_retry():
    while True:
        try:
            return KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        except NoBrokersAvailable:
            print(f"Kafka not reachable yet at {BOOTSTRAP_SERVERS}, retrying in 5s...")
            time.sleep(5)


def main():
    producer = connect_with_retry()

    state = {d["driver_id"]: initial_state() for d in DRIVERS}

    print(f"Streaming F1 telemetry to topic '{TOPIC}' on {BOOTSTRAP_SERVERS}")
    try:
        while True:
            for driver in DRIVERS:
                event = step(driver, state[driver["driver_id"]])
                producer.send(TOPIC, key=driver["driver_id"], value=event)
                print(event)

            producer.flush()
            time.sleep(TICK_INTERVAL_S)
    except KeyboardInterrupt:
        pass
    finally:
        producer.close()


if __name__ == "__main__":
    main()
