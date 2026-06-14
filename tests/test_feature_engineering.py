import pandas as pd
from src.feature_engineering import FeatureEngineer


def test_build_features_crea_columnas():
    config = {}
    engineer = FeatureEngineer(config)
    matches = pd.DataFrame(
        {
            "home_fifa_rank": [1, 10],
            "away_fifa_rank": [5, 12],
            "home_elo": [1900, 1750],
            "away_elo": [1880, 1700],
            "home_goals": [2, 0],
            "away_goals": [1, 1],
            "home_average_age": [27.5, 26.1],
            "away_average_age": [28.0, 25.8],
            "home_market_value": [900, 400],
            "away_market_value": [850, 350],
            "home_team": ["Team A", "Team B"],
            "away_team": ["Team C", "Team D"],
        }
    )
    rankings = pd.DataFrame()
    market = pd.DataFrame()
    features = engineer.build_features({"matches": matches, "rankings": rankings, "market": market})

    assert "fifa_rank_diff" in features.columns
    assert "elo_diff" in features.columns
    assert "goal_difference" in features.columns
    assert "home_advantage" in features.columns


def test_build_features_with_ranking_merge():
    config = {}
    engineer = FeatureEngineer(config)
    matches = pd.DataFrame(
        {
            "home_fifa_rank": [1],
            "away_fifa_rank": [5],
            "home_elo": [1900],
            "away_elo": [1880],
            "home_goals": [2],
            "away_goals": [1],
            "home_average_age": [27.5],
            "away_average_age": [28.0],
            "home_market_value": [900],
            "away_market_value": [850],
            "home_team": ["Team A"],
            "away_team": ["Team B"],
        }
    )
    rankings = pd.DataFrame(
        {
            "team": ["Team A", "Team B"],
            "fifa_rank": [1, 5],
            "elo": [1900, 1880],
        }
    )
    features = engineer.build_features({"matches": matches, "rankings": rankings, "market": pd.DataFrame()})

    assert "fifa_rank_diff" in features.columns
    assert "fifa_rank_home" in features.columns or "fifa_rank" in features.columns
