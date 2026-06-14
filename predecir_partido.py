import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

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

MATCHES = [
    ("Brasil", "Marruecos"),
    ("Argentina", "Austria"),
    ("España", "Arabia Saudita"),
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

MODEL_FEATURE_ORDER = FEATURE_COLUMNS


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    return joblib.load(model_path)


def compute_rank_from_elo(home_elo: float, away_elo: float) -> Tuple[int, int]:
    if home_elo >= away_elo:
        return 1, 2
    return 2, 1


def build_match_features(home_stats: Dict[str, float], away_stats: Dict[str, float]) -> pd.DataFrame:
    stats = {
        "home_elo": home_stats.get("home_elo", DEFAULT_TEAM_STATS["home_elo"]),
        "away_elo": away_stats.get("away_elo", DEFAULT_TEAM_STATS["away_elo"]),
        "home_last5_form": home_stats.get("home_last5_form", DEFAULT_TEAM_STATS["home_last5_form"]),
        "home_last10_form": home_stats.get("home_last10_form", DEFAULT_TEAM_STATS["home_last10_form"]),
        "away_last5_form": away_stats.get("away_last5_form", DEFAULT_TEAM_STATS["away_last5_form"]),
        "away_last10_form": away_stats.get("away_last10_form", DEFAULT_TEAM_STATS["away_last10_form"]),
        "home_gf5": home_stats.get("home_gf5", DEFAULT_TEAM_STATS["home_gf5"]),
        "home_gf10": home_stats.get("home_gf10", DEFAULT_TEAM_STATS["home_gf10"]),
        "away_gf5": away_stats.get("away_gf5", DEFAULT_TEAM_STATS["away_gf5"]),
        "away_gf10": away_stats.get("away_gf10", DEFAULT_TEAM_STATS["away_gf10"]),
        "home_ga5": home_stats.get("home_ga5", DEFAULT_TEAM_STATS["home_ga5"]),
        "home_ga10": home_stats.get("home_ga10", DEFAULT_TEAM_STATS["home_ga10"]),
        "away_ga5": away_stats.get("away_ga5", DEFAULT_TEAM_STATS["away_ga5"]),
        "away_ga10": away_stats.get("away_ga10", DEFAULT_TEAM_STATS["away_ga10"]),
        "home_goals_avg": home_stats.get("home_goals_avg", DEFAULT_TEAM_STATS["home_goals_avg"]),
        "away_goals_avg": away_stats.get("away_goals_avg", DEFAULT_TEAM_STATS["away_goals_avg"]),
        "home_conceded_avg": home_stats.get("home_conceded_avg", DEFAULT_TEAM_STATS["home_conceded_avg"]),
        "away_conceded_avg": away_stats.get("away_conceded_avg", DEFAULT_TEAM_STATS["away_conceded_avg"]),
        "home_win_rate": home_stats.get("home_win_rate", DEFAULT_TEAM_STATS["home_win_rate"]),
        "home_draw_rate": home_stats.get("home_draw_rate", DEFAULT_TEAM_STATS["home_draw_rate"]),
        "home_loss_rate": home_stats.get("home_loss_rate", DEFAULT_TEAM_STATS["home_loss_rate"]),
        "away_win_rate": away_stats.get("away_win_rate", DEFAULT_TEAM_STATS["away_win_rate"]),
        "away_draw_rate": away_stats.get("away_draw_rate", DEFAULT_TEAM_STATS["away_draw_rate"]),
        "away_loss_rate": away_stats.get("away_loss_rate", DEFAULT_TEAM_STATS["away_loss_rate"]),
    }

    stats["elo_diff"] = stats["home_elo"] - stats["away_elo"]
    stats["home_rank"], stats["away_rank"] = compute_rank_from_elo(stats["home_elo"], stats["away_elo"])
    stats["rank_diff"] = stats["home_rank"] - stats["away_rank"]
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

    return pd.DataFrame([{col: stats[col] for col in MODEL_FEATURE_ORDER}])


def predict_match(model, home: str, away: str, home_stats: Dict[str, float], away_stats: Dict[str, float]) -> Dict[str, object]:
    X = build_match_features(home_stats, away_stats)
    proba = model.predict_proba(X)[0]
    confidence = float(np.max(proba))
    score_probs = score_probability_distribution(X.iloc[0])
    top_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    expected_score = (
        sum(h * p for (h, _), p in score_probs.items()),
        sum(a * p for (_, a), p in score_probs.items()),
    )

    return {
        "home_team": home,
        "away_team": away,
        "probabilidad_victoria_local": float(proba[1]),
        "probabilidad_empate": float(proba[0]),
        "probabilidad_victoria_visitante": float(proba[2]),
        "marcador_esperado": {"home": float(expected_score[0]), "away": float(expected_score[1])},
        "top_5_marcadores": [
            {"marcador": f"{h}-{a}", "probabilidad": float(p)}
            for (h, a), p in top_scores
        ],
        "confianza_modelo": confidence,
        "resultado_mas_probable": {0: "Empate", 1: "Victoria local", 2: "Victoria visitante"}[int(np.argmax(proba))],
    }


def score_probability_distribution(features: pd.Series) -> Dict[Tuple[int, int], float]:
    home_expected = max(0.1, features["home_goals_avg"] + 0.05 * features["elo_diff"] / 100)
    away_expected = max(0.1, features["away_goals_avg"] - 0.03 * features["elo_diff"] / 100)
    home_var = max(0.5, features["home_ga5"] / 5.0)
    away_var = max(0.5, features["away_ga5"] / 5.0)

    probs = {}
    total = 0.0
    for home_goals, away_goals in SCORE_CANDIDATES:
        prob = np.exp(-0.5 * (((home_goals - home_expected) ** 2) / home_var + ((away_goals - away_expected) ** 2) / away_var))
        if home_goals > away_goals:
            prob *= 1.02
        elif home_goals < away_goals:
            prob *= 0.98
        probs[(home_goals, away_goals)] = prob
        total += prob

    if total <= 0:
        return {score: 1.0 / len(SCORE_CANDIDATES) for score in SCORE_CANDIDATES}
    return {score: value / total for score, value in probs.items()}


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


def main() -> None:
    model_path = Path(__file__).resolve().parent / "models" / "best_model.pkl"
    model = load_model(model_path)
    team_stats = load_team_stats()

    for home, away in MATCHES:
        home_stats = team_stats.get(home, {})
        away_stats = team_stats.get(away, {})
        prediction = predict_match(model, home, away, home_stats, away_stats)

        print("===================================")
        print(f"Partido: {prediction['home_team']} vs {prediction['away_team']}")
        print(f"Probabilidad victoria local: {prediction['probabilidad_victoria_local']:.3f}")
        print(f"Probabilidad empate: {prediction['probabilidad_empate']:.3f}")
        print(f"Probabilidad victoria visitante: {prediction['probabilidad_victoria_visitante']:.3f}")
        print(f"Marcador esperado: {prediction['marcador_esperado']['home']:.2f}-{prediction['marcador_esperado']['away']:.2f}")
        print("Top 5 marcadores más probables:")
        for entry in prediction["top_5_marcadores"]:
            print(f"  {entry['marcador']}: {entry['probabilidad']:.3f}")
        print(f"Confianza del modelo: {prediction['confianza_modelo']:.3f}")


if __name__ == "__main__":
    main()
