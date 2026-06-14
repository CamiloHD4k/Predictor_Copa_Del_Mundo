from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from predict_match import HOST_TEAMS, MATCH_CONTEXT_WORLD_CUP_NEUTRAL, TEAM_RATINGS, get_fifa_rank, get_team_metrics, get_team_stats, normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_PATH = RAW_DIR / "world_cup_2026_fixtures.csv"
VENUES_PATH = RAW_DIR / "world_cup_2026_venues.csv"
MATCHES_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "matches_master.csv"

CITY_CLIMATE_ESTIMATES = {
    "Ciudad de México": {"temperature_c": 23.0, "humidity_pct": 45.0, "wind_kmh": 12.0, "rain_mm": 2.0, "weather": "Normal"},
    "Guadalajara": {"temperature_c": 28.0, "humidity_pct": 55.0, "wind_kmh": 11.0, "rain_mm": 3.0, "weather": "Normal"},
    "Monterrey": {"temperature_c": 34.0, "humidity_pct": 52.0, "wind_kmh": 13.0, "rain_mm": 2.0, "weather": "Calor extremo"},
    "Toronto": {"temperature_c": 24.0, "humidity_pct": 62.0, "wind_kmh": 14.0, "rain_mm": 2.0, "weather": "Normal"},
    "Vancouver": {"temperature_c": 20.0, "humidity_pct": 70.0, "wind_kmh": 10.0, "rain_mm": 3.0, "weather": "Lluvia"},
    "Nueva York/Nueva Jersey": {"temperature_c": 27.0, "humidity_pct": 65.0, "wind_kmh": 13.0, "rain_mm": 2.0, "weather": "Normal"},
    "Los Ángeles": {"temperature_c": 24.0, "humidity_pct": 55.0, "wind_kmh": 12.0, "rain_mm": 0.0, "weather": "Normal"},
    "Dallas": {"temperature_c": 34.0, "humidity_pct": 55.0, "wind_kmh": 15.0, "rain_mm": 2.0, "weather": "Calor extremo"},
    "Houston": {"temperature_c": 33.0, "humidity_pct": 75.0, "wind_kmh": 14.0, "rain_mm": 4.0, "weather": "Alta humedad"},
    "Miami": {"temperature_c": 31.0, "humidity_pct": 78.0, "wind_kmh": 16.0, "rain_mm": 5.0, "weather": "Alta humedad"},
    "Atlanta": {"temperature_c": 30.0, "humidity_pct": 66.0, "wind_kmh": 10.0, "rain_mm": 3.0, "weather": "Normal"},
    "Kansas City": {"temperature_c": 29.0, "humidity_pct": 61.0, "wind_kmh": 18.0, "rain_mm": 3.0, "weather": "Normal"},
    "Boston": {"temperature_c": 25.0, "humidity_pct": 64.0, "wind_kmh": 15.0, "rain_mm": 2.0, "weather": "Normal"},
    "Philadelphia": {"temperature_c": 28.0, "humidity_pct": 66.0, "wind_kmh": 12.0, "rain_mm": 3.0, "weather": "Normal"},
    "Seattle": {"temperature_c": 20.0, "humidity_pct": 68.0, "wind_kmh": 10.0, "rain_mm": 2.0, "weather": "Normal"},
    "San Francisco Bay Area": {"temperature_c": 21.0, "humidity_pct": 62.0, "wind_kmh": 18.0, "rain_mm": 0.0, "weather": "Normal"},
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_world_cup_fixtures() -> pd.DataFrame:
    return _read_csv(FIXTURES_PATH)


def validate_world_cup_fixtures(fixtures: Optional[pd.DataFrame] = None) -> list[str]:
    fixtures = load_world_cup_fixtures() if fixtures is None else fixtures
    errors = []
    required_columns = {"group", "matchday", "date", "team_home", "team_away", "venue", "city", "country", "phase"}
    missing_columns = sorted(required_columns - set(fixtures.columns))
    if missing_columns:
        errors.append(f"Faltan columnas obligatorias en world_cup_2026_fixtures.csv: {', '.join(missing_columns)}")
        return errors

    if len(fixtures) != 72:
        errors.append(f"Total fase de grupos inválido: se esperaban 72 partidos y hay {len(fixtures)}.")

    for column, label in [
        ("team_home", "equipos locales"),
        ("team_away", "equipos visitantes"),
        ("date", "fechas"),
        ("venue", "sedes"),
    ]:
        empty_count = int(fixtures[column].isna().sum()) + int((fixtures[column].astype(str).str.strip() == "").sum())
        if empty_count:
            errors.append(f"Hay {empty_count} valores vacíos en {label} ({column}).")

    group_count = fixtures["group"].nunique()
    if group_count != 12:
        errors.append(f"Cantidad de grupos inválida: se esperaban 12 grupos y hay {group_count}.")

    group_sizes = fixtures.groupby("group").size()
    for group, count in group_sizes.items():
        if count != 6:
            errors.append(f"Grupo {group}: se esperaban 6 partidos y hay {count}.")

    teams = pd.concat([fixtures[["group", "team_home"]].rename(columns={"team_home": "team"}), fixtures[["group", "team_away"]].rename(columns={"team_away": "team"})])
    team_counts = teams.groupby(["group", "team"]).size()
    for (group, team), count in team_counts.items():
        if count != 3:
            errors.append(f"Equipo {team} en grupo {group}: se esperaban 3 partidos y hay {count}.")

    teams_per_group = teams.drop_duplicates().groupby("group").size()
    for group, count in teams_per_group.items():
        if count != 4:
            errors.append(f"Grupo {group}: se esperaban 4 selecciones y hay {count}.")

    return errors


def _team_variants(team: str) -> set[str]:
    canonical = normalize_team_name(str(team))
    variants = {str(team).strip().lower(), canonical.lower()}
    if canonical == "Argelia":
        variants.update({"algeria", "argelia"})
    elif canonical == "Jordania":
        variants.update({"jordan", "jordania"})
    elif canonical == "Cabo Verde":
        variants.update({"cape verde", "cabo verde", "cape verde islands"})
    elif canonical == "Congo DR":
        variants.update({"dr congo", "congo dr"})
    elif canonical == "Estados Unidos":
        variants.update({"united states", "usa", "estados unidos", "us"})
    elif canonical == "Corea del Sur":
        variants.update({"south korea", "corea del sur", "korea republic"})
    elif canonical == "Czech Republic":
        variants.update({"czechia", "czech republic"})
    return {variant for variant in variants if variant}


def _load_api_key() -> str:
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    key_path = PROJECT_ROOT / "key.env"
    if api_key or not key_path.exists():
        return api_key
    for line in key_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("FOOTBALL_DATA_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _same_team(left: str, right: str) -> bool:
    left_variants = _team_variants(str(left))
    right_variants = _team_variants(str(right))
    return bool(left_variants & right_variants)


def get_related_fixtures(*teams) -> list[Dict[str, object]]:
    fixtures = load_world_cup_fixtures()
    if fixtures.empty:
        return []
    all_variants = set()
    for team in teams:
        all_variants.update(_team_variants(str(team)))

    related = []
    for _, row in fixtures.iterrows():
        home_variants = _team_variants(str(row.get("team_home", "")))
        away_variants = _team_variants(str(row.get("team_away", "")))
        if all_variants & home_variants or all_variants & away_variants:
            related.append(row.to_dict())
    return related


def _weather_label(temperature_c: float, humidity_pct: float, wind_kmh: float, rain_mm: float) -> str:
    if rain_mm >= 4:
        return "Lluvia"
    if wind_kmh >= 28:
        return "Viento fuerte"
    if temperature_c >= 32 and humidity_pct >= 65:
        return "Alta humedad"
    if temperature_c >= 32:
        return "Calor extremo"
    if temperature_c <= 4:
        return "Frío intenso"
    return "Normal"


def weather_source_label(weather_source: str) -> str:
    if weather_source == "open_meteo_forecast":
        return "Clima real/pronóstico"
    if weather_source == "not_available":
        return "No disponible"
    return "Clima estimado histórico"


def get_fixture_info(team_home, team_away):
    fixtures = _read_csv(FIXTURES_PATH)
    if not fixtures.empty:
        for _, row in fixtures.iterrows():
            direct = _same_team(row.get("team_home", ""), team_home) and _same_team(row.get("team_away", ""), team_away)
            reverse = _same_team(row.get("team_home", ""), team_away) and _same_team(row.get("team_away", ""), team_home)
            if direct or reverse:
                data = row.to_dict()
                data["fixture_found"] = True
                data["fixture_order"] = "direct" if direct else "reverse"
                data["fixture_source"] = "local_csv"
                return data

    return None


def get_general_prediction_context(team_home, team_away, competition="World Cup 2026"):
    canonical_home = normalize_team_name(team_home)
    canonical_away = normalize_team_name(team_away)
    home_stats = get_team_stats(team_home)
    away_stats = get_team_stats(team_away)
    home_metrics = get_team_metrics(team_home)
    away_metrics = get_team_metrics(team_away)
    related_fixtures = get_related_fixtures(team_home, team_away)
    related_labels = [
        f"{fixture.get('team_home')} vs {fixture.get('team_away')}"
        for fixture in related_fixtures
    ]
    return {
        "fixture_found": False,
        "official_world_cup_match": False,
        "fixture_source": "not_found",
        "data_source": "predicción_general",
        "message": "Este partido no existe en el fixture oficial del Mundial 2026. Se usará una predicción general entre selecciones.",
        "not_found_detail": f"{team_home} no juega contra {team_away} en el fixture cargado.",
        "related_fixtures": related_fixtures,
        "related_fixture_labels": related_labels,
        "competition": competition,
        "date": "No disponible",
        "time_local": "No disponible",
        "stage": "No disponible",
        "phase": "No disponible",
        "group": "No disponible",
        "matchday": "No disponible",
        "venue": "No disponible",
        "city": "Sin especificar",
        "country": "No disponible",
        "neutral_site": True,
        "match_context": "Predicción general",
        "venue_type": "Predicción general",
        "host_advantage": False,
        "host_advantage_team": "N/A",
        "host_team": None,
        "home_advantage": 0,
        "real_home_advantage": "N/A",
        "altitude_meters": "No disponible",
        "temperature_c": "No disponible",
        "humidity_pct": "No disponible",
        "wind_kmh": "No disponible",
        "rain_mm": "No disponible",
        "weather": "No disponible",
        "weather_source": "not_available",
        "weather_source_label": weather_source_label("not_available"),
        "rest_days_home": "No disponible",
        "rest_days_away": "No disponible",
        "rest": "No disponible",
        "importance": "No disponible",
        "fatigue": "Normal",
        "home_fifa_rank": get_fifa_rank(team_home),
        "away_fifa_rank": get_fifa_rank(team_away),
        "home_elo": home_metrics.get("elo_rating", home_stats.get("home_elo", TEAM_RATINGS.get(canonical_home))),
        "away_elo": away_metrics.get("elo_rating", away_stats.get("away_elo", TEAM_RATINGS.get(canonical_away))),
        "home_squad_value_eur_m": home_metrics.get("squad_value_eur_m"),
        "away_squad_value_eur_m": away_metrics.get("squad_value_eur_m"),
        "home_average_age": home_metrics.get("average_age"),
        "away_average_age": away_metrics.get("average_age"),
        "home_recent_form": home_stats.get("home_last10_form"),
        "away_recent_form": away_stats.get("away_last10_form"),
    }


def get_venue_info(venue_or_city):
    venues = _read_csv(VENUES_PATH)
    if venues.empty or not venue_or_city:
        return None
    needle = str(venue_or_city).strip().lower()
    for _, row in venues.iterrows():
        if str(row.get("venue", "")).strip().lower() == needle or str(row.get("city", "")).strip().lower() == needle:
            return row.to_dict()
    return None


def get_weather_context(city, date):
    estimate = dict(CITY_CLIMATE_ESTIMATES.get(str(city), CITY_CLIMATE_ESTIMATES["San Francisco Bay Area"]))
    estimate["weather_source"] = "city_estimate"
    estimate["weather_source_label"] = weather_source_label("city_estimate")

    venue = get_venue_info(city)
    if not venue:
        return estimate

    try:
        fixture_date = pd.to_datetime(date).date()
        today = datetime.utcnow().date()
        if abs((fixture_date - today).days) <= 14:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": venue["latitude"],
                "longitude": venue["longitude"],
                "daily": "temperature_2m_max,precipitation_sum,wind_speed_10m_max",
                "hourly": "relative_humidity_2m",
                "timezone": "auto",
                "start_date": str(fixture_date),
                "end_date": str(fixture_date),
            }
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            payload = response.json()
            temp = float(payload.get("daily", {}).get("temperature_2m_max", [estimate["temperature_c"]])[0])
            rain = float(payload.get("daily", {}).get("precipitation_sum", [estimate["rain_mm"]])[0])
            wind = float(payload.get("daily", {}).get("wind_speed_10m_max", [estimate["wind_kmh"]])[0])
            humidity_values = payload.get("hourly", {}).get("relative_humidity_2m", [])
            humidity = float(sum(humidity_values) / len(humidity_values)) if humidity_values else estimate["humidity_pct"]
            return {
                "temperature_c": temp,
                "humidity_pct": humidity,
                "wind_kmh": wind,
                "rain_mm": rain,
                "weather": _weather_label(temp, humidity, wind, rain),
                "weather_source": "open_meteo_forecast",
                "weather_source_label": weather_source_label("open_meteo_forecast"),
            }
    except Exception:
        pass
    return estimate


def get_rest_days(team, fixture_date):
    matches = _read_csv(MATCHES_MASTER_PATH)
    if matches.empty or "date" not in matches.columns:
        return None
    fixture_dt = pd.to_datetime(fixture_date, errors="coerce")
    if pd.isna(fixture_dt):
        return None
    dates = pd.to_datetime(matches["date"], errors="coerce")
    variants = {team, normalize_team_name(team)}
    mask = (matches["team_home"].isin(variants) | matches["team_away"].isin(variants)) & dates.notna() & (dates < fixture_dt)
    if not mask.any():
        return None
    last_date = dates[mask].max()
    return int((fixture_dt - last_date).days)


def _rest_label(home_rest, away_rest):
    if home_rest is None or away_rest is None:
        return "Ambos con descanso normal"
    if home_rest < 4 and away_rest < 4:
        return "Ambos con poco descanso"
    if home_rest - away_rest >= 2:
        return "Local con más descanso"
    if away_rest - home_rest >= 2:
        return "Visitante con más descanso"
    return "Ambos con descanso normal"


def _importance_from_stage(stage):
    text = str(stage or "").lower()
    if "final" in text:
        return "Final"
    if "elimin" in text or "round" in text or "octavos" in text or "cuartos" in text or "semifinal" in text:
        return "Eliminación directa"
    if "decisivo" in text:
        return "Partido decisivo de grupo"
    return "Fase de grupos"


def get_match_context(team_home, team_away, competition="World Cup 2026"):
    validation_errors = validate_world_cup_fixtures()
    fixture = get_fixture_info(team_home, team_away)
    if not fixture:
        context = get_general_prediction_context(team_home, team_away, competition=competition)
        context["fixture_validation_errors"] = validation_errors
        return context

    venue = get_venue_info(fixture.get("venue") or fixture.get("city")) or {}
    date = fixture.get("date")
    city = fixture.get("city") or venue.get("city") or "Sin especificar"
    weather = get_weather_context(city, date)

    canonical_home = normalize_team_name(team_home)
    canonical_away = normalize_team_name(team_away)
    host_team = canonical_home if canonical_home in HOST_TEAMS else canonical_away if canonical_away in HOST_TEAMS else "N/A"
    home_rest = get_rest_days(team_home, date)
    away_rest = get_rest_days(team_away, date)
    home_stats = get_team_stats(team_home)
    away_stats = get_team_stats(team_away)
    home_metrics = get_team_metrics(team_home)
    away_metrics = get_team_metrics(team_away)

    return {
        "fixture_found": True,
        "official_world_cup_match": True,
        "fixture_validation_errors": validation_errors,
        "data_source": fixture.get("fixture_source", "local_csv"),
        "message": "",
        "competition": fixture.get("competition", competition),
        "team_home": fixture.get("team_home", ""),
        "team_away": fixture.get("team_away", ""),
        "date": date,
        "time_local": fixture.get("time_local", ""),
        "stage": fixture.get("stage", "Fase de grupos"),
        "phase": fixture.get("stage", "Fase de grupos"),
        "group": fixture.get("group", ""),
        "matchday": fixture.get("matchday", ""),
        "venue": fixture.get("venue", ""),
        "city": city,
        "country": fixture.get("country") or venue.get("country", ""),
        "neutral_site": True,
        "match_context": MATCH_CONTEXT_WORLD_CUP_NEUTRAL,
        "venue_type": "Mundial 2026 neutral",
        "host_advantage_team": host_team,
        "host_team": host_team if host_team != "N/A" else None,
        "home_advantage": 0,
        "real_home_advantage": host_team if host_team != "N/A" else "N/A",
        "altitude_meters": int(float(venue.get("altitude_meters", 0) or 0)),
        "temperature_c": weather["temperature_c"],
        "humidity_pct": weather["humidity_pct"],
        "wind_kmh": weather["wind_kmh"],
        "rain_mm": weather["rain_mm"],
        "weather": weather["weather"],
        "weather_source": weather["weather_source"],
        "weather_source_label": weather.get("weather_source_label", weather_source_label(weather["weather_source"])),
        "rest_days_home": home_rest,
        "rest_days_away": away_rest,
        "rest": _rest_label(home_rest, away_rest),
        "importance": _importance_from_stage(fixture.get("stage")),
        "fatigue": "Normal",
        "home_fifa_rank": get_fifa_rank(team_home),
        "away_fifa_rank": get_fifa_rank(team_away),
        "home_elo": home_metrics.get("elo_rating", home_stats.get("home_elo", TEAM_RATINGS.get(canonical_home))),
        "away_elo": away_metrics.get("elo_rating", away_stats.get("away_elo", TEAM_RATINGS.get(canonical_away))),
        "home_squad_value_eur_m": home_metrics.get("squad_value_eur_m"),
        "away_squad_value_eur_m": away_metrics.get("squad_value_eur_m"),
        "home_average_age": home_metrics.get("average_age"),
        "away_average_age": away_metrics.get("average_age"),
        "home_recent_form": home_stats.get("home_last10_form"),
        "away_recent_form": away_stats.get("away_last10_form"),
    }


def get_auto_context(team_home, team_away):
    return get_match_context(team_home, team_away, competition="World Cup 2026")
