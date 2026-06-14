import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=True)
    return df


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "team_home", "team_away", "goals_home", "goals_away", "result"])
    df = df.sort_values("date").reset_index(drop=True)

    teams = pd.unique(df[["team_home", "team_away"]].values.ravel("K"))
    default_elo = 1500
    elo = {team: default_elo for team in teams}
    team_history = {
        team: {
            "matches": [],
            "goals_for": [],
            "goals_against": [],
            "results": [],
            "points": [],
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "total_goals_for": 0,
            "total_goals_against": 0,
        }
        for team in teams
    }

    rows = []
    k_factor = 20
    home_advantage = 100

    for _, row in df.iterrows():
        home = row["team_home"]
        away = row["team_away"]
        gh = int(row["goals_home"])
        ga = int(row["goals_away"])
        result = int(row["result"])
        date = row["date"]

        home_elo = elo.get(home, default_elo)
        away_elo = elo.get(away, default_elo)
        home_expected = 1 / (1 + 10 ** (-(home_elo + home_advantage - away_elo) / 400))
        away_expected = 1 - home_expected

        home_records = team_history[home]
        away_records = team_history[away]

        def last_n_metrics(records, n):
            subset = records[-n:]
            if not subset:
                return 0.0
            return float(np.mean(subset))

        def last_n_sum(records, n):
            return float(np.sum(records[-n:])) if records else 0.0

        def last_n_count(results, n, target):
            return float(sum(1 for r in results[-n:] if r == target))

        home_matches = len(home_records["results"])
        away_matches = len(away_records["results"])

        home_last5_form = last_n_metrics(home_records["points"], 5)
        home_last10_form = last_n_metrics(home_records["points"], 10)
        away_last5_form = last_n_metrics(away_records["points"], 5)
        away_last10_form = last_n_metrics(away_records["points"], 10)

        home_gf5 = last_n_sum(home_records["goals_for"], 5)
        home_gf10 = last_n_sum(home_records["goals_for"], 10)
        away_gf5 = last_n_sum(away_records["goals_for"], 5)
        away_gf10 = last_n_sum(away_records["goals_for"], 10)

        home_ga5 = last_n_sum(home_records["goals_against"], 5)
        home_ga10 = last_n_sum(home_records["goals_against"], 10)
        away_ga5 = last_n_sum(away_records["goals_against"], 5)
        away_ga10 = last_n_sum(away_records["goals_against"], 10)

        home_goals_avg = home_records["total_goals_for"] / home_matches if home_matches else 0.0
        away_goals_avg = away_records["total_goals_for"] / away_matches if away_matches else 0.0
        home_conceded_avg = home_records["total_goals_against"] / home_matches if home_matches else 0.0
        away_conceded_avg = away_records["total_goals_against"] / away_matches if away_matches else 0.0

        home_win_rate = home_records["wins"] / home_matches if home_matches else 0.0
        home_draw_rate = home_records["draws"] / home_matches if home_matches else 0.0
        home_loss_rate = home_records["losses"] / home_matches if home_matches else 0.0
        away_win_rate = away_records["wins"] / away_matches if away_matches else 0.0
        away_draw_rate = away_records["draws"] / away_matches if away_matches else 0.0
        away_loss_rate = away_records["losses"] / away_matches if away_matches else 0.0

        # dynamic rank at time of match, based on current Elo values
        current_elos = sorted([(team, rating) for team, rating in elo.items()], key=lambda x: x[1], reverse=True)
        team_rank = {team: rank + 1 for rank, (team, _) in enumerate(current_elos)}
        home_rank = team_rank.get(home, len(teams) + 1)
        away_rank = team_rank.get(away, len(teams) + 1)

        rows.append(
            {
                "team_home": home,
                "team_away": away,
                "goals_home": gh,
                "goals_away": ga,
                "result": result,
                "date": date,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "home_rank": home_rank,
                "away_rank": away_rank,
                "home_last5_form": home_last5_form,
                "home_last10_form": home_last10_form,
                "away_last5_form": away_last5_form,
                "away_last10_form": away_last10_form,
                "home_gf5": home_gf5,
                "home_gf10": home_gf10,
                "away_gf5": away_gf5,
                "away_gf10": away_gf10,
                "home_ga5": home_ga5,
                "home_ga10": home_ga10,
                "away_ga5": away_ga5,
                "away_ga10": away_ga10,
                "home_goals_avg": home_goals_avg,
                "away_goals_avg": away_goals_avg,
                "home_conceded_avg": home_conceded_avg,
                "away_conceded_avg": away_conceded_avg,
                "home_win_rate": home_win_rate,
                "home_draw_rate": home_draw_rate,
                "home_loss_rate": home_loss_rate,
                "away_win_rate": away_win_rate,
                "away_draw_rate": away_draw_rate,
                "away_loss_rate": away_loss_rate,
                "home_match_count": home_matches,
                "away_match_count": away_matches,
            }
        )

        # Update history after reading the match
        home_points = 3 if result == 1 else 1 if result == 0 else 0
        away_points = 3 if result == 2 else 1 if result == 0 else 0
        home_records["matches"].append(row.name)
        home_records["goals_for"].append(gh)
        home_records["goals_against"].append(ga)
        home_records["points"].append(home_points)
        home_records["results"].append(result)
        home_records["total_goals_for"] += gh
        home_records["total_goals_against"] += ga
        if result == 1:
            home_records["wins"] += 1
        elif result == 0:
            home_records["draws"] += 1
        else:
            home_records["losses"] += 1

        away_records["matches"].append(row.name)
        away_records["goals_for"].append(ga)
        away_records["goals_against"].append(gh)
        away_records["points"].append(away_points)
        away_records["results"].append(2 if result == 1 else 0 if result == 0 else 1)
        away_records["total_goals_for"] += ga
        away_records["total_goals_against"] += gh
        if result == 2:
            away_records["wins"] += 1
        elif result == 0:
            away_records["draws"] += 1
        else:
            away_records["losses"] += 1

        # update elo ratings
        home_score = 1.0 if result == 1 else 0.5 if result == 0 else 0.0
        away_score = 1.0 if result == 2 else 0.5 if result == 0 else 0.0
        elo[home] = home_elo + k_factor * (home_score - home_expected)
        elo[away] = away_elo + k_factor * (away_score - away_expected)

    feature_df = pd.DataFrame(rows)
    feature_df["elo_diff"] = feature_df["home_elo"] - feature_df["away_elo"]
    feature_df["rank_diff"] = feature_df["home_rank"] - feature_df["away_rank"]
    feature_df["form5_diff"] = feature_df["home_last5_form"] - feature_df["away_last5_form"]
    feature_df["form10_diff"] = feature_df["home_last10_form"] - feature_df["away_last10_form"]
    feature_df["gf5_diff"] = feature_df["home_gf5"] - feature_df["away_gf5"]
    feature_df["gf10_diff"] = feature_df["home_gf10"] - feature_df["away_gf10"]
    feature_df["ga5_diff"] = feature_df["home_ga5"] - feature_df["away_ga5"]
    feature_df["ga10_diff"] = feature_df["home_ga10"] - feature_df["away_ga10"]
    feature_df["goals_avg_diff"] = feature_df["home_goals_avg"] - feature_df["away_goals_avg"]
    feature_df["conceded_avg_diff"] = feature_df["home_conceded_avg"] - feature_df["away_conceded_avg"]
    feature_df["win_rate_diff"] = feature_df["home_win_rate"] - feature_df["away_win_rate"]
    feature_df["draw_rate_diff"] = feature_df["home_draw_rate"] - feature_df["away_draw_rate"]
    feature_df["loss_rate_diff"] = feature_df["home_loss_rate"] - feature_df["away_loss_rate"]
    return feature_df


def get_models(random_state: int):
    models = {
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1500),
        "random_forest": RandomForestClassifier(random_state=random_state, n_estimators=200),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = xgb.XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            use_label_encoder=False,
            num_class=3,
            random_state=random_state,
        )
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = lgb.LGBMClassifier(objective="multiclass", num_class=3, random_state=random_state)
    if CATBOOST_AVAILABLE:
        models["catboost"] = cb.CatBoostClassifier(verbose=0, random_state=random_state)
    return models


def evaluate_models(df: pd.DataFrame, config: dict):
    features = [
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

    X = df[features].fillna(0)
    y = df["result"].astype(int)

    models = get_models(config["model"]["random_state"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["model"]["random_state"])
    report_rows = []
    best_model_name = None
    best_f1 = -1.0
    best_model = None
    best_metrics = None

    for name, model in models.items():
        y_oof = np.zeros_like(y)
        y_oof_proba = np.zeros((len(y), 3))
        cm_total = np.zeros((3, 3), dtype=int)
        metrics_folds = []
        print(f"[INFO] Evaluando modelo {name}")
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf = model.__class__(**model.get_params())
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(y_test), 3))
                y_proba[np.arange(len(y_test)), y_pred] = 1.0
            y_oof[test_idx] = y_pred
            y_oof_proba[test_idx] = y_proba
            cm_total += confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
            metrics_folds.append(
                {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "log_loss": log_loss(y_test, y_proba, labels=[0, 1, 2]),
                }
            )
        fold_avg = {k: np.mean([m[k] for m in metrics_folds]) for k in metrics_folds[0]}
        report_rows.append(
            {
                "model": name,
                **fold_avg,
            }
        )
        print(f"[INFO] Modelo {name} metrics: {fold_avg}")
        if fold_avg["f1"] > best_f1:
            best_f1 = fold_avg["f1"]
            best_model_name = name
            best_metrics = fold_avg
            best_model = model.__class__(**model.get_params())
            best_model.fit(X, y)

        feature_importances = None
        if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
            pass

    report_df = pd.DataFrame(report_rows)
    reports_dir = Path(config["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "model_comparison.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[INFO] Saved model comparison to {report_path}")

    model_dir = Path(config["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    if best_model is not None:
        best_path = model_dir / "best_model.pkl"
        joblib.dump(best_model, best_path)
        print(f"[INFO] Saved best model ({best_model_name}) to {best_path}")

    return best_model_name, best_metrics, best_model, features


def top_features(model, feature_names):
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.array(model.coef_))
        if coef.ndim > 1:
            imp = coef.sum(axis=0)
        else:
            imp = coef
    else:
        return []
    idx = np.argsort(imp)[::-1][:20]
    return [(feature_names[i], float(imp[i])) for i in idx]


def main():
    config = load_config()
    master_path = Path(config["paths"]["data_processed"]) / "matches_master.csv"
    print(f"[INFO] Loading master dataset from {master_path}")
    df = load_master(master_path)
    print(f"[INFO] Master dataset shape: {df.shape}")
    feature_df = build_historical_features(df)
    print(f"[INFO] Feature dataset shape: {feature_df.shape}")
    best_name, best_metrics, best_model, feature_names = evaluate_models(feature_df, config)
    print(f"[INFO] Best model: {best_name}")
    print(f"[INFO] Best metrics: {best_metrics}")
    if best_model is not None:
        top20 = top_features(best_model, feature_names)
        print("[INFO] Top 20 features:")
        for name, value in top20:
            print(f"  {name}: {value:.6f}")

if __name__ == "__main__":
    main()
