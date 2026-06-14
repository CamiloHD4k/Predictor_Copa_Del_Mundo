import argparse
import difflib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from context_adjustments import apply_context_adjustments

FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_last5_form",
    "away_last5_form",
    "home_last10_form",
    "away_last10_form",
    "form5_diff",
    "form10_diff",
    "home_gf5",
    "away_gf5",
    "home_gf10",
    "away_gf10",
    "home_ga5",
    "away_ga5",
    "home_ga10",
    "away_ga10",
    "gf5_diff",
    "gf10_diff",
    "ga5_diff",
    "ga10_diff",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
    "goals_avg_diff",
    "conceded_avg_diff",
    "home_win_rate",
    "away_win_rate",
    "win_rate_diff",
    "home_draw_rate",
    "away_draw_rate",
    "draw_rate_diff",
    "home_loss_rate",
    "away_loss_rate",
    "loss_rate_diff",
]

SCORE_CANDIDATES = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    (3, 0),
    (0, 3),
    (2, 2),
    (3, 1),
]

DEFAULT_TEAM_STATS = {
    "home_elo": 1500.0,
    "away_elo": 1500.0,
    "home_last5_form": 1.5,
    "home_last10_form": 1.5,
    "away_last5_form": 1.5,
    "away_last10_form": 1.5,
    "home_gf5": 6.0,
    "home_gf10": 12.0,
    "away_gf5": 6.0,
    "away_gf10": 12.0,
    "home_ga5": 5.0,
    "home_ga10": 10.0,
    "away_ga5": 5.0,
    "away_ga10": 10.0,
    "home_goals_avg": 1.5,
    "away_goals_avg": 1.5,
    "home_conceded_avg": 1.1,
    "away_conceded_avg": 1.1,
    "home_win_rate": 0.45,
    "away_win_rate": 0.45,
    "home_draw_rate": 0.25,
    "away_draw_rate": 0.25,
    "home_loss_rate": 0.30,
    "away_loss_rate": 0.30,
}

_MODEL = None
_MATCHES_MASTER = None
_FIFA_RANKINGS = None
_TEAM_METRICS = None
MISSING_FIFA_RANK_MESSAGE = "Ranking FIFA no disponible para este equipo"
FIFA_RANK_FALLBACK = 999
MATCH_CONTEXT_WORLD_CUP_NEUTRAL = "Mundial 2026 - sede neutral"
MATCH_CONTEXT_NORMAL_HOME = "Partido normal con localía"
MATCH_CONTEXT_HOST_COUNTRY = "Partido en país anfitrión"
MATCH_CONTEXT_GENERAL = "Predicción general"
HOST_TEAMS = {"Estados Unidos", "Mexico", "Canadá"}
HOME_ADVANTAGE_XG = 0.10

TEAM_NAME_ALIASES = {
    "spain": "España",
    "españa": "España",
    "cape verde": "Cabo Verde",
    "cabo verde": "Cabo Verde",
    "cape verde islands": "Cabo Verde",
    "germany": "Alemania",
    "alemania": "Alemania",
    "england": "Inglaterra",
    "inglaterra": "Inglaterra",
    "netherlands": "Países Bajos",
    "países bajos": "Países Bajos",
    "united states": "Estados Unidos",
    "estados unidos": "Estados Unidos",
    "usa": "Estados Unidos",
    "us": "Estados Unidos",
    "south korea": "Corea del Sur",
    "corea del sur": "Corea del Sur",
    "morocco": "Marruecos",
    "marruecos": "Marruecos",
    "saudi arabia": "Arabia Saudita",
    "arabia saudita": "Arabia Saudita",
    "ivory coast": "Costa de Marfil",
    "costa de marfil": "Costa de Marfil",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "curaçao": "Curacao",
    "curacao": "Curacao",
    "brazil": "Brasil",
    "brasil": "Brasil",
    "uruguay": "Uruguay",
    "france": "Francia",
    "francia": "Francia",
    "belgium": "Bélgica",
    "bélgica": "Bélgica",
    "austria": "Austria",
    "australia": "Australia",
    "canada": "Canadá",
    "canadá": "Canadá",
    "portugal": "Portugal",
    "argentina": "Argentina",
    "algeria": "Argelia",
    "argelia": "Argelia",
    "croatia": "Croacia",
    "czechia": "Czech Republic",
    "czech republic": "Czech Republic",
    "switzerland": "Switzerland",
    "suiza": "Switzerland",
    "haiti": "Haiti",
    "haití": "Haiti",
    "jordan": "Jordania",
    "jordania": "Jordania",
    "panama": "Panama",
    "panamá": "Panama",
    "mexico": "Mexico",
    "méxico": "Mexico",
}

def normalize_team_name(team: str) -> str:
    if not isinstance(team, str):
        return ""
    key = team.strip().lower()
    if key in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[key]

    for canonical in set(TEAM_NAME_ALIASES.values()):
        if canonical.lower() == key:
            return canonical

    candidates = difflib.get_close_matches(
        key,
        list(TEAM_NAME_ALIASES.keys()) + [name.lower() for name in set(TEAM_NAME_ALIASES.values())],
        n=1,
        cutoff=0.8,
    )
    if candidates:
        candidate = candidates[0]
        return TEAM_NAME_ALIASES.get(candidate, next((name for name in set(TEAM_NAME_ALIASES.values()) if name.lower() == candidate), team.strip()))

    return team.strip()


def get_team_variants(team: str) -> List[str]:
    canonical = normalize_team_name(team)
    variants = {canonical}
    for alias, canonical_name in TEAM_NAME_ALIASES.items():
        if canonical_name == canonical:
            variants.add(alias.title())
            variants.add(alias)
    return [variant for variant in variants if variant]


def load_fifa_rankings() -> Dict[str, int]:
    global _FIFA_RANKINGS
    if _FIFA_RANKINGS is None:
        rankings: Dict[str, int] = {}
        metrics = load_team_metrics()
        for team, values in metrics.items():
            if values.get("fifa_rank") is not None:
                rankings[team] = int(values["fifa_rank"])
        path = Path(__file__).resolve().parent / "data" / "raw" / "fifa_rankings_2026.csv"
        if path.exists():
            df = pd.read_csv(path)
            if {"team", "fifa_rank"}.issubset(df.columns):
                for _, row in df.dropna(subset=["team", "fifa_rank"]).iterrows():
                    rankings.setdefault(normalize_team_name(str(row["team"])), int(row["fifa_rank"]))
        _FIFA_RANKINGS = rankings
    return _FIFA_RANKINGS


def get_fifa_rank(team: str) -> Optional[int]:
    return load_fifa_rankings().get(normalize_team_name(team))


def load_team_metrics() -> Dict[str, Dict[str, object]]:
    global _TEAM_METRICS
    if _TEAM_METRICS is None:
        path = Path(__file__).resolve().parent / "data" / "raw" / "team_metrics_2026.csv"
        metrics: Dict[str, Dict[str, object]] = {}
        if path.exists():
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                team = normalize_team_name(str(row["team"]))
                metrics[team] = {
                    "fifa_rank": int(row["fifa_rank"]) if pd.notna(row.get("fifa_rank")) else None,
                    "elo_rating": float(row["elo_rating"]) if pd.notna(row.get("elo_rating")) else None,
                    "squad_value_eur_m": float(row["squad_value_eur_m"]) if pd.notna(row.get("squad_value_eur_m")) else None,
                    "average_age": float(row["average_age"]) if pd.notna(row.get("average_age")) else None,
                    "data_quality": row.get("data_quality", ""),
                    "updated_at": row.get("updated_at", ""),
                }
        _TEAM_METRICS = metrics
    return _TEAM_METRICS


def get_team_metrics(team: str) -> Dict[str, object]:
    return load_team_metrics().get(normalize_team_name(team), {})


def resolve_match_context(team_home: str, team_away: str, match_context: str = MATCH_CONTEXT_WORLD_CUP_NEUTRAL, host_team: Optional[str] = None) -> Dict[str, object]:
    context = match_context or MATCH_CONTEXT_WORLD_CUP_NEUTRAL
    canonical_home = normalize_team_name(team_home)
    canonical_away = normalize_team_name(team_away)
    canonical_host = normalize_team_name(host_team) if host_team else ""

    neutral_site = context == MATCH_CONTEXT_WORLD_CUP_NEUTRAL
    advantage_team = None

    if context == MATCH_CONTEXT_GENERAL:
        neutral_site = True
        advantage_team = None
    elif context == MATCH_CONTEXT_NORMAL_HOME:
        neutral_site = False
        advantage_team = canonical_home
    elif context == MATCH_CONTEXT_HOST_COUNTRY:
        neutral_site = False
        if canonical_host in {canonical_home, canonical_away}:
            advantage_team = canonical_host
        elif canonical_home in HOST_TEAMS:
            advantage_team = canonical_home
        elif canonical_away in HOST_TEAMS:
            advantage_team = canonical_away
    else:
        context = MATCH_CONTEXT_WORLD_CUP_NEUTRAL
        neutral_site = True
        if canonical_home in HOST_TEAMS:
            advantage_team = canonical_home
        elif canonical_away in HOST_TEAMS:
            advantage_team = canonical_away

    return {
        "match_context": context,
        "neutral_site": neutral_site,
        "home_advantage_applied": advantage_team is not None,
        "host_advantage_team": advantage_team or "N/A",
        "home_advantage_value": HOME_ADVANTAGE_XG if advantage_team is not None else 0.0,
        "advantage_side": "home" if advantage_team == canonical_home else "away" if advantage_team == canonical_away else "none",
    }


def load_matches_master() -> pd.DataFrame:
    global _MATCHES_MASTER
    if _MATCHES_MASTER is None:
        path = Path(__file__).resolve().parent / "data" / "processed" / "matches_master.csv"
        if path.exists():
            _MATCHES_MASTER = pd.read_csv(path, parse_dates=["date"])
        else:
            _MATCHES_MASTER = pd.DataFrame()
    return _MATCHES_MASTER


def team_match_count(team: str) -> int:
    df = load_matches_master()
    if df.empty:
        return 0
    variants = get_team_variants(team)
    return int(df["team_home"].isin(variants).sum() + df["team_away"].isin(variants).sum())


def compute_recent_form(matches: pd.DataFrame, is_home: bool) -> Tuple[float, float]:
    if matches.empty:
        return 1.5, 1.5
    if "date" in matches.columns:
        matches = matches.sort_values("date", ascending=False)
    result_scores = []
    for _, row in matches.iterrows():
        result = row["result"]
        if pd.isna(result):
            score = 1.0
        elif is_home:
            score = 2.0 if result == 1 else 1.0 if result == 0 else 0.0
        else:
            score = 2.0 if result == 2 else 1.0 if result == 0 else 0.0
        result_scores.append(score)
    form5 = float(pd.Series(result_scores).head(5).mean()) if result_scores else 1.5
    form10 = float(pd.Series(result_scores).head(10).mean()) if result_scores else 1.5
    return min(max(form5, 0.8), 2.0), min(max(form10, 0.8), 2.0)


def compute_team_stats_from_matches(team: str) -> Dict[str, float]:
    df = load_matches_master()
    if df.empty:
        return {}

    variants = get_team_variants(team)
    home_df = df[df["team_home"].isin(variants)].copy()
    away_df = df[df["team_away"].isin(variants)].copy()
    home_games = len(home_df)
    away_games = len(away_df)
    total_matches = home_games + away_games
    if total_matches == 0:
        return {}

    home_wins = int((home_df["result"] == 1).sum())
    home_draws = int((home_df["result"] == 0).sum())
    home_losses = int((home_df["result"] == 2).sum())
    away_wins = int((away_df["result"] == 2).sum())
    away_draws = int((away_df["result"] == 0).sum())
    away_losses = int((away_df["result"] == 1).sum())

    home_goals_for = float(home_df["goals_home"].sum())
    home_goals_against = float(home_df["goals_away"].sum())
    away_goals_for = float(away_df["goals_away"].sum())
    away_goals_against = float(away_df["goals_home"].sum())

    home_last5_form, home_last10_form = compute_recent_form(home_df, is_home=True)
    away_last5_form, away_last10_form = compute_recent_form(away_df, is_home=False)

    return {
        "home_last5_form": home_last5_form,
        "home_last10_form": home_last10_form,
        "away_last5_form": away_last5_form,
        "away_last10_form": away_last10_form,
        "home_gf5": max(1.0, home_df["goals_home"].tail(5).sum() if home_games else 5.0),
        "home_gf10": max(1.0, home_df["goals_home"].tail(10).sum() if home_games else 10.0),
        "away_gf5": max(1.0, away_df["goals_away"].tail(5).sum() if away_games else 5.0),
        "away_gf10": max(1.0, away_df["goals_away"].tail(10).sum() if away_games else 10.0),
        "home_ga5": max(1.0, home_df["goals_away"].tail(5).sum() if home_games else 5.0),
        "home_ga10": max(1.0, home_df["goals_away"].tail(10).sum() if home_games else 10.0),
        "away_ga5": max(1.0, away_df["goals_home"].tail(5).sum() if away_games else 5.0),
        "away_ga10": max(1.0, away_df["goals_home"].tail(10).sum() if away_games else 5.0),
        "home_goals_avg": float(home_goals_for / home_games) if home_games else DEFAULT_TEAM_STATS["home_goals_avg"],
        "away_goals_avg": float(away_goals_for / away_games) if away_games else DEFAULT_TEAM_STATS["away_goals_avg"],
        "home_conceded_avg": float(home_goals_against / home_games) if home_games else DEFAULT_TEAM_STATS["home_conceded_avg"],
        "away_conceded_avg": float(away_goals_against / away_games) if away_games else DEFAULT_TEAM_STATS["away_conceded_avg"],
        "home_win_rate": float(home_wins / home_games) if home_games else DEFAULT_TEAM_STATS["home_win_rate"],
        "home_draw_rate": float(home_draws / home_games) if home_games else DEFAULT_TEAM_STATS["home_draw_rate"],
        "home_loss_rate": float(home_losses / home_games) if home_games else DEFAULT_TEAM_STATS["home_loss_rate"],
        "away_win_rate": float(away_wins / away_games) if away_games else DEFAULT_TEAM_STATS["away_win_rate"],
        "away_draw_rate": float(away_draws / away_games) if away_games else DEFAULT_TEAM_STATS["away_draw_rate"],
        "away_loss_rate": float(away_losses / away_games) if away_games else DEFAULT_TEAM_STATS["away_loss_rate"],
        "match_count": total_matches,
    }


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    return joblib.load(model_path)


def get_model():
    global _MODEL
    if _MODEL is None:
        model_path = Path(__file__).resolve().parent / "models" / "best_model.pkl"
        _MODEL = load_model(model_path)
    return _MODEL


def role_value(stats: Dict[str, float], side: str, metric: str, default_key: str, neutral_site: bool) -> float:
    if neutral_site:
        home_key = f"home_{metric}"
        away_key = f"away_{metric}"
        home_value = stats.get(home_key, DEFAULT_TEAM_STATS.get(home_key, DEFAULT_TEAM_STATS[default_key]))
        away_value = stats.get(away_key, DEFAULT_TEAM_STATS.get(away_key, DEFAULT_TEAM_STATS[default_key]))
        return (home_value + away_value) / 2.0
    key = f"{side}_{metric}"
    return stats.get(key, DEFAULT_TEAM_STATS.get(key, DEFAULT_TEAM_STATS[default_key]))


def build_match_features(home_stats: Dict[str, float], away_stats: Dict[str, float], neutral_site: bool = False) -> pd.DataFrame:
    home_rank = home_stats.get("fifa_rank")
    away_rank = away_stats.get("fifa_rank")
    if home_rank is not None and away_rank is not None:
        home_rank_feature = int(home_rank)
        away_rank_feature = int(away_rank)
        rank_diff_feature = away_rank_feature - home_rank_feature
    else:
        home_rank_feature = FIFA_RANK_FALLBACK
        away_rank_feature = FIFA_RANK_FALLBACK
        rank_diff_feature = 0

    stats = {
        "home_elo": home_stats.get("home_elo", DEFAULT_TEAM_STATS["home_elo"]),
        "away_elo": away_stats.get("away_elo", DEFAULT_TEAM_STATS["away_elo"]),
        "home_last5_form": role_value(home_stats, "home", "last5_form", "home_last5_form", neutral_site),
        "home_last10_form": role_value(home_stats, "home", "last10_form", "home_last10_form", neutral_site),
        "away_last5_form": role_value(away_stats, "away", "last5_form", "away_last5_form", neutral_site),
        "away_last10_form": role_value(away_stats, "away", "last10_form", "away_last10_form", neutral_site),
        "home_gf5": role_value(home_stats, "home", "gf5", "home_gf5", neutral_site),
        "home_gf10": role_value(home_stats, "home", "gf10", "home_gf10", neutral_site),
        "away_gf5": role_value(away_stats, "away", "gf5", "away_gf5", neutral_site),
        "away_gf10": role_value(away_stats, "away", "gf10", "away_gf10", neutral_site),
        "home_ga5": role_value(home_stats, "home", "ga5", "home_ga5", neutral_site),
        "home_ga10": role_value(home_stats, "home", "ga10", "home_ga10", neutral_site),
        "away_ga5": role_value(away_stats, "away", "ga5", "away_ga5", neutral_site),
        "away_ga10": role_value(away_stats, "away", "ga10", "away_ga10", neutral_site),
        "home_goals_avg": role_value(home_stats, "home", "goals_avg", "home_goals_avg", neutral_site),
        "away_goals_avg": role_value(away_stats, "away", "goals_avg", "away_goals_avg", neutral_site),
        "home_conceded_avg": role_value(home_stats, "home", "conceded_avg", "home_conceded_avg", neutral_site),
        "away_conceded_avg": role_value(away_stats, "away", "conceded_avg", "away_conceded_avg", neutral_site),
        "home_win_rate": role_value(home_stats, "home", "win_rate", "home_win_rate", neutral_site),
        "home_draw_rate": role_value(home_stats, "home", "draw_rate", "home_draw_rate", neutral_site),
        "home_loss_rate": role_value(home_stats, "home", "loss_rate", "home_loss_rate", neutral_site),
        "away_win_rate": role_value(away_stats, "away", "win_rate", "away_win_rate", neutral_site),
        "away_draw_rate": role_value(away_stats, "away", "draw_rate", "away_draw_rate", neutral_site),
        "away_loss_rate": role_value(away_stats, "away", "loss_rate", "away_loss_rate", neutral_site),
    }

    stats["elo_diff"] = stats["home_elo"] - stats["away_elo"]
    stats["home_rank"] = home_rank_feature
    stats["away_rank"] = away_rank_feature
    stats["rank_diff"] = rank_diff_feature
    stats["form5_diff"] = stats["home_last5_form"] - stats["away_last5_form"]
    stats["form10_diff"] = stats["home_last10_form"] - stats["away_last10_form"]
    stats["gf5_diff"] = stats["home_gf5"] - stats["away_gf5"]
    stats["gf10_diff"] = stats["home_gf10"] - stats["away_gf10"]
    stats["ga5_diff"] = stats["home_ga5"] - stats["away_ga5"]
    stats["ga10_diff"] = stats["home_ga10"] - stats["away_ga10"]
    stats["goals_avg_diff"] = stats["home_goals_avg"] - stats["away_goals_avg"]
    stats["conceded_avg_diff"] = stats["home_conceded_avg"] - stats["away_conceded_avg"]
    stats["win_rate_diff"] = stats["home_win_rate"] - stats["away_win_rate"]
    stats["draw_rate_diff"] = stats["home_draw_rate"] - stats["away_draw_rate"]
    stats["loss_rate_diff"] = stats["home_loss_rate"] - stats["away_loss_rate"]

    return pd.DataFrame([{col: stats[col] for col in FEATURE_COLUMNS}])


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def expected_goals_from_features(features: pd.Series, context_info: Optional[Dict[str, object]] = None) -> Tuple[float, float]:
    context_info = context_info or {}
    elo_diff = float(features["elo_diff"])
    rank_diff = float(features["rank_diff"])
    home_form = (float(features["home_last5_form"]) + float(features["home_last10_form"])) / 2.0
    away_form = (float(features["away_last5_form"]) + float(features["away_last10_form"])) / 2.0
    form_diff = home_form - away_form

    home_xg = (float(features["home_goals_avg"]) + float(features["away_conceded_avg"])) / 2.0
    away_xg = (float(features["away_goals_avg"]) + float(features["home_conceded_avg"])) / 2.0

    advantage_side = context_info.get("advantage_side", "home")
    if advantage_side == "home":
        home_xg *= 1.0 + HOME_ADVANTAGE_XG
        away_xg *= 0.98
    elif advantage_side == "away":
        away_xg *= 1.0 + HOME_ADVANTAGE_XG
        home_xg *= 0.98

    elo_scale = clamp(elo_diff / 400.0, -0.9, 0.9)
    home_xg *= 1.0 + 0.28 * elo_scale
    away_xg *= 1.0 - 0.28 * elo_scale
    if elo_diff < -200:
        extra = clamp((-elo_diff - 200.0) / 500.0, 0.0, 0.30)
        away_xg *= 1.18 + extra
        home_xg *= 0.92
    elif elo_diff > 200:
        extra = clamp((elo_diff - 200.0) / 500.0, 0.0, 0.30)
        home_xg *= 1.18 + extra
        away_xg *= 0.92

    rank_scale = clamp(rank_diff / 80.0, -1.0, 1.0)
    home_xg *= 1.0 + 0.35 * rank_scale
    away_xg *= 1.0 - 0.35 * rank_scale
    if rank_diff < -40:
        extra = clamp((-rank_diff - 40.0) / 100.0, 0.0, 0.35)
        away_xg *= 1.20 + extra
        home_xg *= 0.85
    elif rank_diff > 40:
        extra = clamp((rank_diff - 40.0) / 100.0, 0.0, 0.35)
        home_xg *= 1.20 + extra
        away_xg *= 0.85

    form_scale = clamp(form_diff / 1.2, -0.6, 0.6)
    home_xg *= 1.0 + 0.12 * form_scale
    away_xg *= 1.0 - 0.12 * form_scale

    if elo_diff < -200 or rank_diff < -40:
        away_xg = max(away_xg, home_xg + 0.55)
    elif elo_diff > 200 or rank_diff > 40:
        home_xg = max(home_xg, away_xg + 0.55)

    return clamp(home_xg, 0.15, 4.5), clamp(away_xg, 0.15, 4.5)


def score_probability_distribution(features: pd.Series, max_goals: int = 5, context_info: Optional[Dict[str, object]] = None) -> Dict[Tuple[int, int], float]:
    home_expected, away_expected = expected_goals_from_features(features, context_info=context_info)
    elo_diff = float(features["elo_diff"])
    rank_diff = float(features["rank_diff"])
    strong_favorite = abs(elo_diff) > 200 or abs(rank_diff) > 40

    probs = {}
    total = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_pmf(home_expected, home_goals) * poisson_pmf(away_expected, away_goals)
            if strong_favorite and home_goals == away_goals:
                prob *= 0.72
            if (elo_diff > 200 or rank_diff > 40) and home_goals > away_goals:
                prob *= 1.08
            elif (elo_diff < -200 or rank_diff < -40) and away_goals > home_goals:
                prob *= 1.08
            probs[(home_goals, away_goals)] = prob
            total += prob

    if total <= 0:
        uniform = 1.0 / ((max_goals + 1) ** 2)
        return {score: uniform for score in probs}
    return {score: value / total for score, value in probs.items()}


def score_probability_distribution_from_expected(home_expected: float, away_expected: float, max_goals: int = 5) -> Dict[Tuple[int, int], float]:
    probs = {}
    total = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = poisson_pmf(home_expected, home_goals) * poisson_pmf(away_expected, away_goals)
            probs[(home_goals, away_goals)] = prob
            total += prob
    if total <= 0:
        uniform = 1.0 / ((max_goals + 1) ** 2)
        return {score: uniform for score in probs}
    return {score: value / total for score, value in probs.items()}


def result_probabilities_from_features(features: pd.Series, context_info: Optional[Dict[str, object]] = None) -> np.ndarray:
    score_probs = score_probability_distribution(features, context_info=context_info)
    draw = sum(value for (home_goals, away_goals), value in score_probs.items() if home_goals == away_goals)
    home_win = sum(value for (home_goals, away_goals), value in score_probs.items() if home_goals > away_goals)
    away_win = sum(value for (home_goals, away_goals), value in score_probs.items() if home_goals < away_goals)
    prob = np.array([draw, home_win, away_win], dtype=float)
    return prob / prob.sum() if prob.sum() > 0 else np.array([0.25, 0.50, 0.25], dtype=float)


def calibrate_probabilities(proba: np.ndarray, features: pd.Series) -> np.ndarray:
    calibrated = np.asarray(proba, dtype=float)
    if calibrated.sum() <= 0:
        return np.array([0.25, 0.50, 0.25], dtype=float)
    calibrated /= calibrated.sum()

    elo_diff = float(features["elo_diff"])
    rank_diff = float(features["rank_diff"])
    home = calibrated[1]
    draw = calibrated[0]
    away = calibrated[2]

    if elo_diff >= 150:
        home = max(home, 0.52)
        away = min(away, 0.28)
    elif elo_diff >= 100:
        home = max(home, 0.50)
        away = min(away, 0.30)
    elif elo_diff <= -150:
        away = max(away, 0.52)
        home = min(home, 0.28)
    elif elo_diff <= -100:
        away = max(away, 0.50)
        home = min(home, 0.30)

    if elo_diff >= 120 and away > 0.35:
        away = 0.35
    if elo_diff <= -120 and home > 0.35:
        home = 0.35

    if rank_diff >= 50:
        home = max(home, 0.55)
        away = min(away, 0.25)
    elif rank_diff <= -50:
        away = max(away, 0.55)
        home = min(home, 0.25)

    if elo_diff >= 200 or rank_diff >= 55:
        draw = min(draw, 0.22)
    elif elo_diff <= -200 or rank_diff <= -55:
        draw = min(draw, 0.22)

    if elo_diff >= 300 or rank_diff >= 70:
        draw = min(draw, 0.18)
        home = max(home, 0.62)
    elif elo_diff <= -300 or rank_diff <= -70:
        draw = min(draw, 0.18)
        away = max(away, 0.62)

    remainder = 1.0 - home - away
    draw = max(0.08, min(remainder, draw))
    if remainder - draw > 0:
        if elo_diff >= 0:
            home += remainder - draw
        else:
            away += remainder - draw
    calibrated = np.array([draw, home, away], dtype=float)
    return calibrated / calibrated.sum()


def apply_team_metrics(stats: Dict[str, float], team: str) -> Dict[str, float]:
    metrics = get_team_metrics(team)
    if metrics.get("elo_rating") is not None:
        stats["home_elo"] = float(metrics["elo_rating"])
        stats["away_elo"] = float(metrics["elo_rating"])
    else:
        stats["home_elo"] = stats.get("home_elo", TEAM_RATINGS.get(team, DEFAULT_TEAM_STATS["home_elo"]))
        stats["away_elo"] = stats.get("away_elo", stats["home_elo"])
    stats["fifa_rank"] = metrics.get("fifa_rank", get_fifa_rank(team))
    stats["squad_value_eur_m"] = metrics.get("squad_value_eur_m")
    stats["average_age"] = metrics.get("average_age")
    stats["team_metrics_quality"] = metrics.get("data_quality", "")
    stats["team_metrics_updated_at"] = metrics.get("updated_at", "")
    return stats


def get_team_stats(team: str) -> Dict[str, float]:
    canonical = normalize_team_name(team)
    team_stats = load_team_stats()
    if canonical in team_stats:
        stats = dict(team_stats[canonical])
        stats["team_name"] = canonical
        stats["matched_name"] = canonical
        stats["match_count"] = team_match_count(canonical)
        stats["warning"] = ""
        if stats["match_count"] < 20:
            stats["warning"] = "Datos insuficientes para este equipo"
        return apply_team_metrics(stats, canonical)

    match_stats = compute_team_stats_from_matches(canonical)
    if match_stats:
        stats = dict(match_stats)
        stats["team_name"] = canonical
        stats["matched_name"] = canonical
        stats["warning"] = ""
        if stats["match_count"] < 20:
            stats["warning"] = "Datos insuficientes para este equipo"
        return apply_team_metrics(stats, canonical)

    stats = estimate_team_stats(canonical)
    stats["team_name"] = canonical
    stats["matched_name"] = canonical
    stats["warning"] = "Datos insuficientes para este equipo"
    stats["match_count"] = 0
    return apply_team_metrics(stats, canonical)


def poisson_pmf(lmbda: float, k: int) -> float:
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)


def predict_exact_scores(
    team_home: str,
    team_away: str,
    max_goals: int = 5,
    match_context: str = MATCH_CONTEXT_WORLD_CUP_NEUTRAL,
    host_team: Optional[str] = None,
    external_context: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    home_stats = get_team_stats(team_home)
    away_stats = get_team_stats(team_away)
    context_info = resolve_match_context(team_home, team_away, match_context=match_context, host_team=host_team)
    features = build_match_features(home_stats, away_stats, neutral_site=bool(context_info["neutral_site"])).iloc[0]
    if external_context:
        home_xg, away_xg = expected_goals_from_features(features, context_info=context_info)
        adjustment_context = dict(external_context)
        adjustment_context.setdefault("venue_type", match_context)
        _, adjusted_goals, _ = apply_context_adjustments(
            result_probabilities_from_features(features, context_info=context_info),
            home_xg,
            away_xg,
            {
                "home_team": normalize_team_name(team_home),
                "away_team": normalize_team_name(team_away),
                **context_info,
            },
            adjustment_context,
        )
        score_probs = score_probability_distribution_from_expected(adjusted_goals[0], adjusted_goals[1], max_goals=max_goals)
    else:
        score_probs = score_probability_distribution(features, max_goals=max_goals, context_info=context_info)

    scores: List[Dict[str, object]] = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            if home_goals > away_goals:
                result = "Local"
            elif home_goals < away_goals:
                result = "Visitante"
            else:
                result = "Empate"
            scores.append(
                {
                    "marcador": f"{home_goals}-{away_goals}",
                    "probabilidad": float(score_probs[(home_goals, away_goals)]),
                    "resultado": result,
                }
            )

    scores.sort(key=lambda item: item["probabilidad"], reverse=True)
    return scores


def save_exact_score_predictions(team_home: str, team_away: str, exact_scores: List[Dict[str, object]], output_path: Path = None) -> None:
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "reports" / "exact_score_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "home_team": team_home,
            "away_team": team_away,
            "marcador": row["marcador"],
            "probabilidad": row["probabilidad"],
            "resultado": row["resultado"],
        }
        for row in exact_scores
    ]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def format_exact_score_table(exact_scores: List[Dict[str, object]], top_n: int = 20) -> str:
    lines = ["Marcadores exactos probables:"]
    for row in exact_scores[:top_n]:
        lines.append(f"{row['marcador']} | {row['probabilidad'] * 100:.1f}% | {row['resultado']}")
    return "\n".join(lines)


def format_1x2_table(prediction: Dict[str, object]) -> str:
    lines = [
        "Tabla 1X2:",
        f"Victoria local: {prediction['probabilidad_victoria_local'] * 100:.1f}%",
        f"Empate: {prediction['probabilidad_empate'] * 100:.1f}%",
        f"Victoria visitante: {prediction['probabilidad_victoria_visitante'] * 100:.1f}%",
        f"Resultado más probable: {prediction['resultado_mas_probable']}",
        f"Confianza: {prediction['confianza_modelo'] * 100:.1f}%",
    ]
    return "\n".join(lines)


TEAM_RATINGS: Dict[str, int] = {
    "Argentina": 1860,
    "Brasil": 1850,
    "España": 1805,
    "Francia": 1840,
    "Uruguay": 1790,
    "Inglaterra": 1830,
    "Croacia": 1785,
    "Alemania": 1810,
    "Portugal": 1815,
    "Marruecos": 1730,
    "Australia": 1740,
    "Canadá": 1720,
    "Arabia Saudita": 1605,
    "Austria": 1745,
    "Uruguay": 1790,
    "Uruguay": 1790,
}

def estimate_team_stats(team: str) -> Dict[str, float]:
    rating = float(TEAM_RATINGS.get(team, 1650))
    home_elo = rating
    away_elo = rating
    form_base = 1.1 + (rating - DEFAULT_TEAM_STATS["home_elo"]) / 800.0
    last5_form = min(max(form_base, 0.8), 2.0)
    last10_form = min(max(form_base - 0.05, 0.8), 2.0)
    goals_avg = min(max(1.1 + (rating - DEFAULT_TEAM_STATS["home_elo"]) / 1000.0, 0.8), 2.5)
    conceded_avg = min(max(1.4 - (rating - DEFAULT_TEAM_STATS["home_elo"]) / 1400.0, 0.6), 1.8)
    gf5 = max(2.5, goals_avg * 5.0)
    gf10 = max(5.0, goals_avg * 10.0)
    ga5 = max(1.5, conceded_avg * 5.0)
    ga10 = max(3.0, conceded_avg * 10.0)
    win_rate = min(max(0.30 + (rating - DEFAULT_TEAM_STATS["home_elo"]) / 1400.0, 0.20), 0.75)
    draw_rate = min(max(0.25 - abs(rating - DEFAULT_TEAM_STATS["home_elo"]) / 2200.0, 0.15), 0.35)
    loss_rate = max(0.05, 1.0 - win_rate - draw_rate)
    return {
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_last5_form": last5_form,
        "home_last10_form": last10_form,
        "away_last5_form": last5_form,
        "away_last10_form": last10_form,
        "home_gf5": gf5,
        "home_gf10": gf10,
        "away_gf5": gf5,
        "away_gf10": gf10,
        "home_ga5": ga5,
        "home_ga10": ga10,
        "away_ga5": ga5,
        "away_ga10": ga10,
        "home_goals_avg": goals_avg,
        "away_goals_avg": goals_avg,
        "home_conceded_avg": conceded_avg,
        "away_conceded_avg": conceded_avg,
        "home_win_rate": win_rate,
        "away_win_rate": win_rate,
        "home_draw_rate": draw_rate,
        "away_draw_rate": draw_rate,
        "home_loss_rate": loss_rate,
        "away_loss_rate": loss_rate,
    }


def load_team_stats() -> Dict[str, Dict[str, float]]:
    return {
        "Brasil": {
            "home_elo": 1840.0,
            "away_elo": 1840.0,
            "home_last5_form": 2.4,
            "home_last10_form": 2.0,
            "away_last5_form": 1.4,
            "away_last10_form": 1.5,
            "home_gf5": 12.0,
            "home_gf10": 22.0,
            "away_gf5": 7.0,
            "away_gf10": 14.0,
            "home_ga5": 3.0,
            "home_ga10": 7.0,
            "away_ga5": 5.0,
            "away_ga10": 10.0,
            "home_goals_avg": 2.2,
            "away_goals_avg": 1.3,
            "home_conceded_avg": 0.7,
            "away_conceded_avg": 1.1,
            "home_win_rate": 0.75,
            "home_draw_rate": 0.15,
            "home_loss_rate": 0.10,
            "away_win_rate": 0.55,
            "away_draw_rate": 0.20,
            "away_loss_rate": 0.25,
        },
        "Marruecos": {
            "home_elo": 1730.0,
            "away_elo": 1730.0,
            "home_last5_form": 1.7,
            "home_last10_form": 1.6,
            "away_last5_form": 1.8,
            "away_last10_form": 1.7,
            "home_gf5": 8.0,
            "home_gf10": 15.0,
            "away_gf5": 10.0,
            "away_gf10": 18.0,
            "home_ga5": 4.0,
            "home_ga10": 8.0,
            "away_ga5": 3.0,
            "away_ga10": 6.0,
            "home_goals_avg": 1.6,
            "away_goals_avg": 1.8,
            "home_conceded_avg": 0.9,
            "away_conceded_avg": 0.7,
            "home_win_rate": 0.55,
            "home_draw_rate": 0.25,
            "home_loss_rate": 0.20,
            "away_win_rate": 0.60,
            "away_draw_rate": 0.20,
            "away_loss_rate": 0.20,
        },
        "Argentina": {
            "home_elo": 1820.0,
            "away_elo": 1820.0,
            "home_last5_form": 2.2,
            "home_last10_form": 1.9,
            "away_last5_form": 1.2,
            "away_last10_form": 1.3,
            "home_gf5": 11.0,
            "home_gf10": 20.0,
            "away_gf5": 6.0,
            "away_gf10": 12.0,
            "home_ga5": 3.0,
            "home_ga10": 6.0,
            "away_ga5": 5.0,
            "away_ga10": 10.0,
            "home_goals_avg": 2.0,
            "away_goals_avg": 1.2,
            "home_conceded_avg": 0.8,
            "away_conceded_avg": 1.1,
            "home_win_rate": 0.70,
            "home_draw_rate": 0.20,
            "home_loss_rate": 0.10,
            "away_win_rate": 0.50,
            "away_draw_rate": 0.25,
            "away_loss_rate": 0.25,
        },
        "Austria": {
            "home_elo": 1745.0,
            "away_elo": 1745.0,
            "home_last5_form": 1.5,
            "home_last10_form": 1.6,
            "away_last5_form": 1.3,
            "away_last10_form": 1.4,
            "home_gf5": 7.0,
            "home_gf10": 14.0,
            "away_gf5": 8.0,
            "away_gf10": 15.0,
            "home_ga5": 5.0,
            "home_ga10": 9.0,
            "away_ga5": 4.0,
            "away_ga10": 8.0,
            "home_goals_avg": 1.4,
            "away_goals_avg": 1.6,
            "home_conceded_avg": 1.1,
            "away_conceded_avg": 0.9,
            "home_win_rate": 0.45,
            "home_draw_rate": 0.30,
            "home_loss_rate": 0.25,
            "away_win_rate": 0.55,
            "away_draw_rate": 0.25,
            "away_loss_rate": 0.20,
        },
        "España": {
            "home_elo": 1805.0,
            "away_elo": 1805.0,
            "home_last5_form": 2.0,
            "home_last10_form": 1.8,
            "away_last5_form": 1.2,
            "away_last10_form": 1.3,
            "home_gf5": 10.0,
            "home_gf10": 18.0,
            "away_gf5": 6.0,
            "away_gf10": 12.0,
            "home_ga5": 3.0,
            "home_ga10": 7.0,
            "away_ga5": 5.0,
            "away_ga10": 10.0,
            "home_goals_avg": 1.9,
            "away_goals_avg": 1.3,
            "home_conceded_avg": 0.8,
            "away_conceded_avg": 1.0,
            "home_win_rate": 0.68,
            "home_draw_rate": 0.20,
            "home_loss_rate": 0.12,
            "away_win_rate": 0.48,
            "away_draw_rate": 0.28,
            "away_loss_rate": 0.24,
        },
        "Arabia Saudita": {
            "home_elo": 1605.0,
            "away_elo": 1605.0,
            "home_last5_form": 1.1,
            "home_last10_form": 1.2,
            "away_last5_form": 1.4,
            "away_last10_form": 1.3,
            "home_gf5": 5.0,
            "home_gf10": 9.0,
            "away_gf5": 8.0,
            "away_gf10": 13.0,
            "home_ga5": 8.0,
            "home_ga10": 14.0,
            "away_ga5": 5.0,
            "away_ga10": 10.0,
            "home_goals_avg": 1.1,
            "away_goals_avg": 1.6,
            "home_conceded_avg": 1.6,
            "away_conceded_avg": 1.0,
            "home_win_rate": 0.30,
            "home_draw_rate": 0.30,
            "home_loss_rate": 0.40,
            "away_win_rate": 0.45,
            "away_draw_rate": 0.25,
            "away_loss_rate": 0.30,
        },
    }


def serialize_prediction(prediction: Dict[str, object]) -> Dict[str, object]:
    return {
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "probabilidad_victoria_local": prediction["probabilidad_victoria_local"],
        "probabilidad_empate": prediction["probabilidad_empate"],
        "probabilidad_victoria_visitante": prediction["probabilidad_victoria_visitante"],
        "resultado_mas_probable": prediction["resultado_mas_probable"],
        "confianza_modelo": prediction["confianza_modelo"],
        "marcador_esperado_home": prediction["marcador_esperado"]["home"],
        "marcador_esperado_away": prediction["marcador_esperado"]["away"],
        "top_5_marcadores": json.dumps(prediction["top_5_marcadores"], ensure_ascii=False),
    }


def save_prediction(prediction: Dict[str, object], output_path: Path = None) -> None:
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "reports" / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = serialize_prediction(prediction)
    df = pd.DataFrame([row])
    df.to_csv(output_path, mode="a", index=False, header=not output_path.exists())


def predict_match(
    team_home: str,
    team_away: str,
    match_context: str = MATCH_CONTEXT_WORLD_CUP_NEUTRAL,
    host_team: Optional[str] = None,
    external_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    canonical_home = normalize_team_name(team_home)
    canonical_away = normalize_team_name(team_away)
    home_stats = get_team_stats(team_home)
    away_stats = get_team_stats(team_away)
    context_info = resolve_match_context(team_home, team_away, match_context=match_context, host_team=host_team)
    model = get_model()
    X = build_match_features(home_stats, away_stats, neutral_site=bool(context_info["neutral_site"]))

    ml_proba = model.predict_proba(X)[0]
    if context_info["neutral_site"]:
        swapped_X = build_match_features(away_stats, home_stats, neutral_site=True)
        swapped_proba = model.predict_proba(swapped_X)[0]
        ml_proba = 0.5 * ml_proba + 0.5 * np.array([swapped_proba[0], swapped_proba[2], swapped_proba[1]], dtype=float)

    poisson_proba = result_probabilities_from_features(X.iloc[0], context_info=context_info)
    hybrid_proba = 0.5 * ml_proba + 0.5 * poisson_proba
    calibrated_proba = calibrate_probabilities(hybrid_proba, X.iloc[0])

    expected_goals_home, expected_goals_away = expected_goals_from_features(X.iloc[0], context_info=context_info)
    adjustment_context = dict(external_context or {})
    adjustment_context.setdefault("venue_type", match_context)
    adjustment_context.setdefault("match_context", match_context)
    adjusted_proba, adjusted_goals, context_explanations = apply_context_adjustments(
        calibrated_proba,
        expected_goals_home,
        expected_goals_away,
        {
            "home_team": canonical_home,
            "away_team": canonical_away,
            **context_info,
        },
        adjustment_context,
    )
    adjusted_proba = np.array(adjusted_proba, dtype=float)
    adjusted_goals_home, adjusted_goals_away = adjusted_goals
    confidence = float(np.max(adjusted_proba))
    score_probs = score_probability_distribution_from_expected(adjusted_goals_home, adjusted_goals_away, max_goals=5)
    top_scores = sorted(score_probs.items(), key=lambda item: item[1], reverse=True)[:5]

    warnings = []
    if home_stats.get("warning"):
        warnings.append(f"Local: {home_stats['warning']}")
    if away_stats.get("warning"):
        warnings.append(f"Visitante: {away_stats['warning']}")
    if home_stats.get("fifa_rank") is None:
        warnings.append(f"Local: {MISSING_FIFA_RANK_MESSAGE}")
    if away_stats.get("fifa_rank") is None:
        warnings.append(f"Visitante: {MISSING_FIFA_RANK_MESSAGE}")

    home_rank = home_stats.get("fifa_rank")
    away_rank = away_stats.get("fifa_rank")
    rank_diff = away_rank - home_rank if home_rank is not None and away_rank is not None else None

    diagnostic = {
        "home_team": canonical_home,
        "away_team": canonical_away,
        "match_context": context_info["match_context"],
        "neutral_site": bool(context_info["neutral_site"]),
        "home_advantage_applied": bool(context_info["home_advantage_applied"]),
        "host_advantage_team": context_info["host_advantage_team"],
        "home_advantage_value": float(context_info["home_advantage_value"]),
        "context_city": adjustment_context.get("city", "Sin especificar"),
        "context_weather": adjustment_context.get("weather", "Normal"),
        "context_rest": adjustment_context.get("rest", "Ambos con descanso normal"),
        "context_importance": adjustment_context.get("importance", "Fase de grupos"),
        "context_fatigue": adjustment_context.get("fatigue", "Normal"),
        "home_elo": float(X.iloc[0]["home_elo"]),
        "away_elo": float(X.iloc[0]["away_elo"]),
        "elo_diff": float(X.iloc[0]["elo_diff"]),
        "home_rank": int(home_rank) if home_rank is not None else MISSING_FIFA_RANK_MESSAGE,
        "away_rank": int(away_rank) if away_rank is not None else MISSING_FIFA_RANK_MESSAGE,
        "rank_diff": int(rank_diff) if rank_diff is not None else MISSING_FIFA_RANK_MESSAGE,
        "home_squad_value_eur_m": home_stats.get("squad_value_eur_m", "N/A"),
        "away_squad_value_eur_m": away_stats.get("squad_value_eur_m", "N/A"),
        "squad_value_diff_eur_m": (
            float(home_stats["squad_value_eur_m"]) - float(away_stats["squad_value_eur_m"])
            if home_stats.get("squad_value_eur_m") is not None and away_stats.get("squad_value_eur_m") is not None
            else "N/A"
        ),
        "home_average_age": home_stats.get("average_age", "N/A"),
        "away_average_age": away_stats.get("average_age", "N/A"),
        "average_age_diff": (
            float(home_stats["average_age"]) - float(away_stats["average_age"])
            if home_stats.get("average_age") is not None and away_stats.get("average_age") is not None
            else "N/A"
        ),
        "team_metrics_quality_home": home_stats.get("team_metrics_quality", ""),
        "team_metrics_quality_away": away_stats.get("team_metrics_quality", ""),
        "team_metrics_updated_at": home_stats.get("team_metrics_updated_at") or away_stats.get("team_metrics_updated_at", ""),
        "base_expected_goals_home": float(expected_goals_home),
        "base_expected_goals_away": float(expected_goals_away),
        "base_expected_goals_diff": float(expected_goals_home - expected_goals_away),
        "expected_goals_home": float(adjusted_goals_home),
        "expected_goals_away": float(adjusted_goals_away),
        "expected_goals_diff": float(adjusted_goals_home - adjusted_goals_away),
        "home_win_rate": float(X.iloc[0]["home_win_rate"]),
        "away_win_rate": float(X.iloc[0]["away_win_rate"]),
        "home_goals_avg": float(X.iloc[0]["home_goals_avg"]),
        "away_goals_avg": float(X.iloc[0]["away_goals_avg"]),
        "home_conceded_avg": float(X.iloc[0]["home_conceded_avg"]),
        "away_conceded_avg": float(X.iloc[0]["away_conceded_avg"]),
        "form10_diff": float(X.iloc[0]["form10_diff"]),
        "home_match_count": int(home_stats.get("match_count", 0)),
        "away_match_count": int(away_stats.get("match_count", 0)),
        "ml_prob_draw": float(ml_proba[0]),
        "ml_prob_home": float(ml_proba[1]),
        "ml_prob_away": float(ml_proba[2]),
        "poisson_prob_draw": float(poisson_proba[0]),
        "poisson_prob_home": float(poisson_proba[1]),
        "poisson_prob_away": float(poisson_proba[2]),
        "base_prob_draw": float(calibrated_proba[0]),
        "base_prob_home": float(calibrated_proba[1]),
        "base_prob_away": float(calibrated_proba[2]),
        "adjusted_prob_draw": float(adjusted_proba[0]),
        "adjusted_prob_home": float(adjusted_proba[1]),
        "adjusted_prob_away": float(adjusted_proba[2]),
        "context_adjustments": context_explanations,
        "warnings": warnings,
    }

    explanation_lines = [
        f"Predicción para {canonical_home} vs {canonical_away}",
        f"Modelo ML: Local {ml_proba[1] * 100:.1f}%, Empate {ml_proba[0] * 100:.1f}%, Visitante {ml_proba[2] * 100:.1f}%",
        f"Poisson/ELO: Local {poisson_proba[1] * 100:.1f}%, Empate {poisson_proba[0] * 100:.1f}%, Visitante {poisson_proba[2] * 100:.1f}%",
        f"Híbrido 50/50: Local {calibrated_proba[1] * 100:.1f}%, Empate {calibrated_proba[0] * 100:.1f}%, Visitante {calibrated_proba[2] * 100:.1f}%",
        f"Contexto: {context_info['match_context']}",
        f"Diferencia ELO: {diagnostic['elo_diff']:+.0f}",
        f"xG base: {expected_goals_home:.2f} - {expected_goals_away:.2f}",
        f"xG ajustado por contexto: {adjusted_goals_home:.2f} - {adjusted_goals_away:.2f}",
    ]
    if context_explanations:
        explanation_lines.append("Ajustes contextuales: " + "; ".join(context_explanations))
    if warnings:
        explanation_lines.append("Advertencias: " + "; ".join(warnings))
    explanation_lines.append("Se utiliza calibración ELO para evitar probabilidades exageradas cuando hay una gran diferencia de nivel.")

    return {
        "home_team": team_home,
        "away_team": team_away,
        "base_probabilities": {
            "draw": float(calibrated_proba[0]),
            "home": float(calibrated_proba[1]),
            "away": float(calibrated_proba[2]),
        },
        "adjusted_probabilities": {
            "draw": float(adjusted_proba[0]),
            "home": float(adjusted_proba[1]),
            "away": float(adjusted_proba[2]),
        },
        "probabilidad_victoria_local": float(adjusted_proba[1]),
        "probabilidad_empate": float(adjusted_proba[0]),
        "probabilidad_victoria_visitante": float(adjusted_proba[2]),
        "resultado_mas_probable": {0: "Empate", 1: "Victoria local", 2: "Victoria visitante"}[int(np.argmax(adjusted_proba))],
        "confianza_modelo": confidence,
        "base_expected_goals": {"home": float(expected_goals_home), "away": float(expected_goals_away)},
        "marcador_esperado": {"home": float(adjusted_goals_home), "away": float(adjusted_goals_away)},
        "top_5_marcadores": [
            {"marcador": f"{h}-{a}", "probabilidad": float(p)}
            for (h, a), p in top_scores
        ],
        "diagnostic": diagnostic,
        "explanation": "\n".join(explanation_lines),
    }


def explain_prediction(team_home: str, team_away: str) -> str:
    return predict_match(team_home, team_away)["explanation"]


def format_prediction(prediction: Dict[str, object]) -> str:
    lines = [
        f"Partido: {prediction['home_team']} vs {prediction['away_team']}",
        f"Probabilidad victoria local: {prediction['probabilidad_victoria_local']:.3f}",
        f"Probabilidad empate: {prediction['probabilidad_empate']:.3f}",
        f"Probabilidad victoria visitante: {prediction['probabilidad_victoria_visitante']:.3f}",
        f"Resultado más probable: {prediction['resultado_mas_probable']}",
        f"Confianza del modelo: {prediction['confianza_modelo']:.3f}",
        f"Marcador esperado: {prediction['marcador_esperado']['home']:.2f}-{prediction['marcador_esperado']['away']:.2f}",
        "Top 5 marcadores más probables:",
    ]
    for entry in prediction["top_5_marcadores"]:
        lines.append(f"  {entry['marcador']}: {entry['probabilidad']:.3f}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Predecir un partido con el modelo best_model.pkl")
    parser.add_argument("--team_home", "-H", help="Equipo local")
    parser.add_argument("--team_away", "-A", help="Equipo visitante")
    return parser.parse_args()


def main():
    args = parse_args()
    home = args.team_home or input("Equipo local: ").strip()
    away = args.team_away or input("Equipo visitante: ").strip()
    prediction = predict_match(home, away)
    exact_scores = predict_exact_scores(home, away, max_goals=5)

    print(f"Partido: {home} vs {away}\n")
    print(format_1x2_table(prediction))
    print()
    print(format_exact_score_table(exact_scores, top_n=10))

    save_prediction(prediction)
    save_exact_score_predictions(home, away, exact_scores)
    print(f"\nResultado guardado en reports/predictions.csv")
    print(f"Marcadores exactos guardados en reports/exact_score_predictions.csv")


if __name__ == "__main__":
    main()
