"""Define los grupos oficiales del Mundial 2026 y simula la fase de grupos."""

import random
from typing import Dict, List

import pandas as pd

GROUPS_2026: Dict[str, List[str]] = {
    "A": ["Mexico", "South Korea", "Czech Republic", "South Africa"],
    "B": ["Canada", "Bosnia and Herzegovina", "Switzerland", "Qatar"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


class GroupStageSimulator:
    """Simula la fase de grupos del Mundial 2026 con posiciones y estadísticas."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.team_strengths = self._generate_team_strengths()

    def _generate_team_strengths(self) -> Dict[str, float]:
        strengths: Dict[str, float] = {}
        for group_teams in GROUPS_2026.values():
            for team in group_teams:
                if team not in strengths:
                    strengths[team] = round(self.rng.uniform(0.55, 1.0), 3)
        return strengths

    def _group_match_pairs(self) -> List[Dict[str, str]]:
        matches: List[Dict[str, str]] = []
        for group, teams in GROUPS_2026.items():
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    matches.append({"group": group, "team_a": teams[i], "team_b": teams[j]})
        return matches

    def _match_outcome(self, team_a: str, team_b: str) -> str:
        strength_a = self.team_strengths.get(team_a, 0.7)
        strength_b = self.team_strengths.get(team_b, 0.7)
        diff = strength_a - strength_b

        home_win = 0.35 + 0.2 * diff
        away_win = 0.35 - 0.2 * diff
        draw = 1.0 - home_win - away_win

        home_win = max(0.05, min(home_win, 0.8))
        away_win = max(0.05, min(away_win, 0.8))
        draw = max(0.05, min(draw, 0.8))

        total = home_win + draw + away_win
        weights = [home_win / total, draw / total, away_win / total]
        return self.rng.choices(["home", "draw", "away"], weights=weights, k=1)[0]

    def _match_score(self, team_a: str, team_b: str, outcome: str) -> tuple[int, int]:
        if outcome == "draw":
            goals = self.rng.randint(0, 2)
            return goals, goals

        if outcome == "home":
            away_goals = self.rng.randint(0, 2)
            home_goals = away_goals + self.rng.randint(1, 3)
            return home_goals, away_goals

        home_goals = self.rng.randint(0, 2)
        away_goals = home_goals + self.rng.randint(1, 3)
        return home_goals, away_goals

    def _initialize_table(self) -> pd.DataFrame:
        table_rows = []
        for group, teams in GROUPS_2026.items():
            for team in teams:
                table_rows.append(
                    {
                        "group": group,
                        "team": team,
                        "points": 0,
                        "played": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "goals_for": 0,
                        "goals_against": 0,
                        "goal_difference": 0,
                    }
                )
        return pd.DataFrame(table_rows)

    def simulate(self) -> pd.DataFrame:
        """Simula todos los partidos de la fase de grupos y devuelve la clasificación."""
        table = self._initialize_table()
        matches = self._group_match_pairs()

        for match in matches:
            home, away = match["team_a"], match["team_b"]
            outcome = self._match_outcome(home, away)
            home_goals, away_goals = self._match_score(home, away, outcome)
            group = match["group"]

            table_mask_home = (table["group"] == group) & (table["team"] == home)
            table_mask_away = (table["group"] == group) & (table["team"] == away)

            table.loc[table_mask_home, "played"] += 1
            table.loc[table_mask_away, "played"] += 1
            table.loc[table_mask_home, "goals_for"] += home_goals
            table.loc[table_mask_home, "goals_against"] += away_goals
            table.loc[table_mask_away, "goals_for"] += away_goals
            table.loc[table_mask_away, "goals_against"] += home_goals

            if home_goals > away_goals:
                table.loc[table_mask_home, "wins"] += 1
                table.loc[table_mask_away, "losses"] += 1
                table.loc[table_mask_home, "points"] += 3
            elif home_goals < away_goals:
                table.loc[table_mask_away, "wins"] += 1
                table.loc[table_mask_home, "losses"] += 1
                table.loc[table_mask_away, "points"] += 3
            else:
                table.loc[table_mask_home, "draws"] += 1
                table.loc[table_mask_away, "draws"] += 1
                table.loc[table_mask_home, "points"] += 1
                table.loc[table_mask_away, "points"] += 1

        table["goal_difference"] = table["goals_for"] - table["goals_against"]
        table = table.sort_values(
            by=["group", "points", "goal_difference", "goals_for", "team"],
            ascending=[True, False, False, False, True],
        )
        table["rank"] = table.groupby("group").cumcount() + 1
        columns = [
            "group",
            "rank",
            "team",
            "points",
            "played",
            "wins",
            "draws",
            "losses",
            "goals_for",
            "goals_against",
            "goal_difference",
        ]
        return table[columns]
