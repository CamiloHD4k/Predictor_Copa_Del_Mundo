import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import predict_match as pm
from groups_2026 import GROUPS_2026

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(42)
PREDICTION_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}

TEAM_RATINGS: Dict[str, int] = {
    "Argentina": 1860,
    "Brazil": 1850,
    "France": 1840,
    "England": 1830,
    "Spain": 1825,
    "Portugal": 1815,
    "Germany": 1810,
    "Netherlands": 1805,
    "Belgium": 1795,
    "Uruguay": 1790,
    "Croatia": 1785,
    "Morocco": 1780,
    "Senegal": 1770,
    "Japan": 1765,
    "Switzerland": 1760,
    "United States": 1755,
    "Mexico": 1750,
    "Colombia": 1745,
    "Australia": 1740,
    "Scotland": 1735,
    "South Korea": 1730,
    "Argentina": 1860,
    "Austria": 1725,
    "Canada": 1720,
    "Ireland": 1715,
    "Czech Republic": 1710,
    "Norway": 1705,
    "Turkey": 1700,
    "Denmark": 1695,
    "Iraq": 1690,
    "Ivory Coast": 1685,
    "Ecuador": 1680,
    "Paraguay": 1675,
    "Egypt": 1670,
    "Iran": 1665,
    "New Zealand": 1660,
    "Curaçao": 1655,
    "Costa Rica": 1650,
    "Jordan": 1645,
    "Algeria": 1640,
    "Panama": 1635,
    "DR Congo": 1630,
    "Uzbekistan": 1625,
    "Haiti": 1620,
    "Cape Verde": 1615,
    "Saudi Arabia": 1610,
    "Qatar": 1605,
    "South Africa": 1600,
    "Tunisia": 1595,
    "Sweden": 1590,
}

DEFAULT_RATING = 1550


def estimate_team_stats(team: str) -> Dict[str, float]:
    rating = TEAM_RATINGS.get(team, DEFAULT_RATING)
    home_elo = float(rating)
    away_elo = float(rating)
    form_base = 1.0 + (rating - DEFAULT_RATING) / 800.0
    last5_form = min(max(form_base + RNG.uniform(-0.1, 0.1), 0.8), 2.0)
    last10_form = min(max(form_base + RNG.uniform(-0.12, 0.12), 0.8), 2.0)
    goals_avg = min(max(1.1 + (rating - DEFAULT_RATING) / 1000.0, 0.8), 2.5)
    conceded_avg = min(max(1.4 - (rating - DEFAULT_RATING) / 1400.0, 0.6), 1.8)
    gf5 = max(2.5, goals_avg * 5 + RNG.uniform(-1.0, 1.0))
    gf10 = max(5.0, goals_avg * 10 + RNG.uniform(-2.0, 2.0))
    ga5 = max(1.5, conceded_avg * 5 + RNG.uniform(-1.0, 1.0))
    ga10 = max(3.0, conceded_avg * 10 + RNG.uniform(-2.0, 2.0))
    win_rate = min(max(0.30 + (rating - DEFAULT_RATING) / 1400.0 + RNG.uniform(-0.03, 0.03), 0.20), 0.75)
    draw_rate = min(max(0.25 - abs(rating - DEFAULT_RATING) / 2200.0 + RNG.uniform(-0.03, 0.03), 0.15), 0.35)
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


def _patch_predict_match_team_stats() -> None:
    original_load_team_stats = pm.load_team_stats

    def merged_load_team_stats():
        stats = original_load_team_stats()
        for group in GROUPS_2026.values():
            for team in group:
                if team not in stats:
                    stats[team] = estimate_team_stats(team)
        return stats

    pm.load_team_stats = merged_load_team_stats


def get_match_prediction(home: str, away: str) -> Dict[str, object]:
    key = (home, away)
    if key not in PREDICTION_CACHE:
        PREDICTION_CACHE[key] = pm.predict_match(home, away)
    return PREDICTION_CACHE[key]


def _round_robin_matches() -> List[Tuple[str, str, str]]:
    matches = []
    for group, teams in GROUPS_2026.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matches.append((group, teams[i], teams[j]))
    return matches


def pick_score(prediction: Dict[str, object], outcome: str) -> Tuple[int, int]:
    scoring = {
        "home": [],
        "draw": [],
        "away": [],
    }
    for entry in prediction["top_5_marcadores"]:
        home_goals, away_goals = map(int, entry["marcador"].split("-"))
        if home_goals > away_goals:
            scoring["home"].append((home_goals, away_goals, entry["probabilidad"]))
        elif home_goals < away_goals:
            scoring["away"].append((home_goals, away_goals, entry["probabilidad"]))
        else:
            scoring["draw"].append((home_goals, away_goals, entry["probabilidad"]))

    candidates = scoring[outcome]
    if candidates:
        weights = [prob for _, _, prob in candidates]
        total = sum(weights)
        normalized = [w / total for w in weights]
        choice = RNG.choices(candidates, weights=normalized, k=1)[0]
        return choice[0], choice[1]

    home_expected = round(prediction["marcador_esperado"]["home"])
    away_expected = round(prediction["marcador_esperado"]["away"])
    if outcome == "home" and home_expected <= away_expected:
        home_expected = away_expected + 1
    if outcome == "away" and away_expected <= home_expected:
        away_expected = home_expected + 1
    if outcome == "draw" and home_expected != away_expected:
        away_expected = home_expected
    return max(home_expected, 0), max(away_expected, 0)


def simulate_match(home: str, away: str) -> Tuple[int, int, str]:
    prediction = get_match_prediction(home, away)
    p_home = prediction["probabilidad_victoria_local"]
    p_draw = prediction["probabilidad_empate"]
    p_away = prediction["probabilidad_victoria_visitante"]
    outcome = RNG.choices(["home", "draw", "away"], weights=[p_home, p_draw, p_away], k=1)[0]
    home_goals, away_goals = pick_score(prediction, outcome)
    if outcome == "home" and home_goals <= away_goals:
        home_goals = away_goals + 1
    if outcome == "away" and away_goals <= home_goals:
        away_goals = home_goals + 1
    if outcome == "draw" and home_goals != away_goals:
        away_goals = home_goals
    winner = "home" if home_goals > away_goals else "away" if away_goals > home_goals else "draw"
    return home_goals, away_goals, winner


def ordinal_suffix(rank: int) -> str:
    return "th" if 10 <= rank % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rank % 10, "th")


def build_group_table(match_results: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for result in match_results:
        group = result["group"]
        home = result["home_team"]
        away = result["away_team"]
        home_goals = result["home_goals"]
        away_goals = result["away_goals"]
        rows.append((group, home, home_goals, away_goals))
        rows.append((group, away, away_goals, home_goals))

    table = pd.DataFrame(
        rows,
        columns=["group", "team", "goals_for", "goals_against"],
    )
    summary = table.groupby(["group", "team"]).sum().reset_index()
    summary["goal_difference"] = summary["goals_for"] - summary["goals_against"]
    summary["played"] = 3
    summary["wins"] = 0
    summary["draws"] = 0
    summary["losses"] = 0
    summary["points"] = 0

    for result in match_results:
        home_mask = (summary["group"] == result["group"]) & (summary["team"] == result["home_team"])
        away_mask = (summary["group"] == result["group"]) & (summary["team"] == result["away_team"])
        home_goals = result["home_goals"]
        away_goals = result["away_goals"]
        if home_goals > away_goals:
            summary.loc[home_mask, ["wins", "points"]] += [1, 3]
            summary.loc[away_mask, "losses"] += 1
        elif home_goals < away_goals:
            summary.loc[away_mask, ["wins", "points"]] += [1, 3]
            summary.loc[home_mask, "losses"] += 1
        else:
            summary.loc[home_mask, ["draws", "points"]] += [1, 1]
            summary.loc[away_mask, ["draws", "points"]] += [1, 1]

    summary["goal_difference"] = summary["goals_for"] - summary["goals_against"]
    summary = summary.sort_values(
        ["group", "points", "goal_difference", "goals_for", "team"],
        ascending=[True, False, False, False, True],
    )
    summary["rank"] = summary.groupby("group").cumcount() + 1
    return summary[
        ["group", "rank", "team", "points", "played", "wins", "draws", "losses", "goals_for", "goals_against", "goal_difference"]
    ]


def select_qualifiers(group_table: pd.DataFrame) -> List[Tuple[str, int, str]]:
    qualifiers: List[Tuple[str, int, str]] = []
    third_place_rows = []
    for group, group_df in group_table.groupby("group"):
        qualifiers.extend([(row["team"], row["rank"], group) for _, row in group_df[group_df["rank"] <= 2].iterrows()])
        third_place_rows.append(group_df[group_df["rank"] == 3].iloc[0])

    third_place_df = pd.DataFrame(third_place_rows)
    third_place_df = third_place_df.sort_values(["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True])
    best_third = [(row["team"], int(row["rank"]), row["group"]) for _, row in third_place_df.head(8).iterrows()]
    qualifiers.extend(best_third)
    return qualifiers


def seed_bracket(qualifiers: List[Tuple[str, int, str]]) -> List[Tuple[str, str]]:
    sorted_by_seed = sorted(
        qualifiers,
        key=lambda item: (item[1], -group_seed(item[2]), item[0]),
    )
    teams = [item[0] for item in sorted_by_seed]
    pairings = []
    while teams:
        pairings.append((teams.pop(0), teams.pop(-1)))
    return pairings


def group_seed(group: str) -> int:
    return ord(group) - ord("A")


def simulate_knockout_round(round_name: str, pairings: List[Tuple[str, str]]) -> Tuple[List[str], List[Dict[str, object]]]:
    winners: List[str] = []
    match_rows: List[Dict[str, object]] = []
    for home, away in pairings:
        home_goals, away_goals, winner_flag = simulate_match(home, away)
        winner = home if winner_flag == "home" else away if winner_flag == "away" else RNG.choice([home, away])
        match_rows.append(
            {
                "stage": round_name,
                "group": "",
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "winner": winner,
            }
        )
        winners.append(winner)
    return winners, match_rows


def run_single_tournament() -> Tuple[pd.DataFrame, List[Dict[str, object]], str]:
    group_matches = _round_robin_matches()
    match_results: List[Dict[str, object]] = []
    for group, home, away in group_matches:
        home_goals, away_goals, winner_flag = simulate_match(home, away)
        winner = home if winner_flag == "home" else away if winner_flag == "away" else "draw"
        match_results.append(
            {
                "stage": "group",
                "group": group,
                "home_team": home,
                "away_team": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "winner": winner,
            }
        )

    group_table = build_group_table(match_results)
    qualifiers = select_qualifiers(group_table)
    knockout_pairs = seed_bracket(qualifiers)
    knockout_results: List[Dict[str, object]] = []

    for round_name in ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]:
        winners, round_matches = simulate_knockout_round(round_name, knockout_pairs)
        knockout_results.extend(round_matches)
        if round_name != "Final":
            knockout_pairs = []
            for i in range(0, len(winners), 2):
                knockout_pairs.append((winners[i], winners[i + 1]))

    champion = knockout_results[-1]["winner"]
    return group_table, match_results + knockout_results, champion


def save_simulation_results(group_table: pd.DataFrame, match_rows: List[Dict[str, object]], output_path: Path) -> None:
    simulation_rows = []
    for _, row in group_table.iterrows():
        simulation_rows.append(
            {
                "stage": "group_standings",
                "group": row["group"],
                "home_team": row["team"],
                "away_team": "",
                "home_goals": row["points"],
                "away_goals": row["goals_for"],
                "winner": "",
                "points": row["points"],
                "goals_for": row["goals_for"],
                "goals_against": row["goals_against"],
                "goal_difference": row["goal_difference"],
            }
        )
    for match in match_rows:
        simulation_rows.append(
            {
                "stage": match["stage"],
                "group": match["group"],
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "winner": match["winner"],
                "points": "",
                "goals_for": "",
                "goals_against": "",
                "goal_difference": "",
            }
        )
    keys = [
        "stage",
        "group",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "winner",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(simulation_rows)


def save_champion_probabilities(champion_counts: Counter, n_simulations: int, output_path: Path) -> None:
    rows = []
    for team, count in champion_counts.most_common():
        rows.append(
            {
                "team": team,
                "champion_count": count,
                "champion_probability": count / n_simulations,
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["team", "champion_count", "champion_probability"])
        writer.writeheader()
        writer.writerows(rows)


def run_simulation(n_simulations: int = 10000) -> Counter:
    champion_counts: Counter = Counter()
    for _ in range(n_simulations):
        _, _, champion = run_single_tournament()
        champion_counts[champion] += 1
    return champion_counts


def print_top_favorites(champion_counts: Counter, n_simulations: int, top_n: int = 10) -> None:
    print("Top 10 favoritos al título:")
    for i, (team, count) in enumerate(champion_counts.most_common(top_n), start=1):
        probability = count / n_simulations
        print(f"{i}. {team}: {probability:.3%} ({count}/{n_simulations})")


def main() -> None:
    _patch_predict_match_team_stats()
    representative_table, representative_matches, champion = run_single_tournament()
    simulation_path = REPORTS_DIR / "world_cup_simulation.csv"
    champion_path = REPORTS_DIR / "champion_probabilities.csv"
    save_simulation_results(representative_table, representative_matches, simulation_path)

    n_simulations = 10000
    champion_counts = run_simulation(n_simulations)
    save_champion_probabilities(champion_counts, n_simulations, champion_path)

    print(f"Simulación representativa guardada en: {simulation_path}")
    print(f"Probabilidades de campeón guardadas en: {champion_path}")
    print(f"Campeón proyectado de la simulación representativa: {champion}")
    print_top_favorites(champion_counts, n_simulations)


if __name__ == "__main__":
    main()
