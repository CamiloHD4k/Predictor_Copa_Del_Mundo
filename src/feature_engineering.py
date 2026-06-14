import pandas as pd
from typing import Dict

HOST_TEAMS = {"United States", "Estados Unidos", "USA", "Mexico", "México", "Canada", "Canadá"}
MATCH_CONTEXT_WORLD_CUP_NEUTRAL = "Mundial 2026 - sede neutral"
MATCH_CONTEXT_NORMAL_HOME = "Partido normal con localía"
MATCH_CONTEXT_HOST_COUNTRY = "Partido en país anfitrión"


class FeatureEngineer:
    """Generador de variables predictivas para los modelos de predicción."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config

    def _ensure_columns(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for column in columns:
            if column not in df.columns:
                df[column] = 0
        return df

    def build_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Construye un conjunto de características a partir de los datos cargados."""
        matches = data.get("matches", pd.DataFrame()).copy()
        rankings = data.get("rankings", pd.DataFrame())
        market = data.get("market", pd.DataFrame())

        if matches.empty:
            print("[WARN] No hay datos de partidos para generar variables.")
            return pd.DataFrame()

        matches = matches.rename(
            columns={
                "team_home": "home_team",
                "team_away": "away_team",
                "goals_home": "home_goals",
                "goals_away": "away_goals",
                "fifa_rank_home": "home_fifa_rank",
                "fifa_rank_away": "away_fifa_rank",
                "elo_home": "home_elo",
                "elo_away": "away_elo",
                "form_home": "home_form",
                "form_away": "away_form",
                "goals_scored_avg_home": "home_goals_scored_avg",
                "goals_scored_avg_away": "away_goals_scored_avg",
                "goals_conceded_avg_home": "home_goals_conceded_avg",
                "goals_conceded_avg_away": "away_goals_conceded_avg",
            }
        )

        expected_columns = [
            "home_team",
            "away_team",
            "home_fifa_rank",
            "away_fifa_rank",
            "home_elo",
            "away_elo",
            "home_goals",
            "away_goals",
            "home_form",
            "away_form",
            "home_goals_scored_avg",
            "away_goals_scored_avg",
            "home_goals_conceded_avg",
            "away_goals_conceded_avg",
            "home_average_age",
            "away_average_age",
            "home_market_value",
            "away_market_value",
        ]
        matches = self._ensure_columns(matches, expected_columns)

        features = matches.copy()
        features["fifa_rank_diff"] = features["home_fifa_rank"] - features["away_fifa_rank"]
        features["elo_diff"] = features["home_elo"] - features["away_elo"]
        features["form_diff"] = features["home_form"] - features["away_form"]
        features["goals_scored_avg_diff"] = features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
        features["goals_conceded_avg_diff"] = features["home_goals_conceded_avg"] - features["away_goals_conceded_avg"]
        # Avoid using current match final scores (leakage). Use recent averages instead.
        features["goal_difference"] = features["home_goals_scored_avg"] - features["away_goals_scored_avg"]
        features["average_age_diff"] = features["home_average_age"] - features["away_average_age"]
        features["market_value_diff"] = features["home_market_value"] - features["away_market_value"]
        match_context = self.config.get("match_context", MATCH_CONTEXT_NORMAL_HOME)
        host_team = self.config.get("host_team")
        features["neutral_site"] = match_context == MATCH_CONTEXT_WORLD_CUP_NEUTRAL
        if match_context == MATCH_CONTEXT_WORLD_CUP_NEUTRAL:
            features["home_advantage"] = features["home_team"].isin(HOST_TEAMS).astype(int)
        elif match_context == MATCH_CONTEXT_HOST_COUNTRY:
            if host_team:
                features["home_advantage"] = (features["home_team"] == host_team).astype(int)
            else:
                features["home_advantage"] = features["home_team"].isin(HOST_TEAMS).astype(int)
        else:
            features["home_advantage"] = 1

        if not rankings.empty and "team" in rankings.columns:
            rank_home = rankings.rename(columns={"team": "rank_team"}).add_suffix("_home")
            rank_away = rankings.rename(columns={"team": "rank_team"}).add_suffix("_away")
            features = features.merge(rank_home, how="left", left_on="home_team", right_on="rank_team_home")
            features = features.merge(rank_away, how="left", left_on="away_team", right_on="rank_team_away")

        if not market.empty and "team" in market.columns:
            market_home = market.rename(columns={"team": "market_team"}).add_suffix("_home")
            market_away = market.rename(columns={"team": "market_team"}).add_suffix("_away")
            features = features.merge(market_home, how="left", left_on="home_team", right_on="market_team_home")
            features = features.merge(market_away, how="left", left_on="away_team", right_on="market_team_away")
        # Drop raw current-match goal columns to avoid leaking the target
        for col in ["home_goals", "away_goals"]:
            if col in features.columns:
                features = features.drop(columns=[col])

        return features
