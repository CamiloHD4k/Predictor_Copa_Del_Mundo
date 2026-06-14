import os
import sys
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api_clients import FootballDataClient
from src.config import load_config

config = load_config()

sources = [
    {
        "name": "matches_combined",
        "path": Path(config["paths"]["data_raw"]) / "matches_combined.csv",
        "type": "local",
    },
    {
        "name": "openfootball_worldcup",
        "url": "https://raw.githubusercontent.com/openfootball/world-cup/master/worldcup.csv",
        "type": "csv",
    },
    {
        "name": "openfootball_worldcup_2018",
        "url": "https://raw.githubusercontent.com/openfootball/world-cup/master/2018/worldcup.csv",
        "type": "csv",
    },
    {
        "name": "international_results",
        "url": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
        "type": "csv",
    },
    {
        "name": "kaggle_world_cup_2018",
        "url": "https://raw.githubusercontent.com/zamzahn/football-data/master/worldcup/matches.csv",
        "type": "csv",
    },
]

# attempt football-data.co.uk public URLs from international/history
football_data_urls = [
    "https://www.football-data.co.uk/mmz4281/2223/EC.csv",
    "https://www.football-data.co.uk/mmz4281/2122/EC.csv",
    "https://www.football-data.co.uk/mmz4281/2021/E1.csv",
    "https://www.football-data.co.uk/mmz4281/2223/WC.csv",
]


def download_csv(url: str) -> pd.DataFrame:
    print(f"[DOWNLOAD] {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as exc:
        print(f"[WARN] No se pudo descargar {url}: {exc}")
        return pd.DataFrame()


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    cols = {c.lower(): c for c in normalized.columns}

    def find(colname):
        for k, v in cols.items():
            if k == colname:
                return v
        return None

    mapping = {}
    home_team_col = find("team_home") or find("home_team") or find("home")
    away_team_col = find("team_away") or find("away_team") or find("away")
    home_goals_col = find("goals_home") or find("home_goals") or find("home_score") or find("home")
    away_goals_col = find("goals_away") or find("away_goals") or find("away_score") or find("away")
    date_col = find("date") or find("match_date") or find("game_date")
    score_col = find("score") or find("ft_score")
    result_col = find("result") or find("outcome")

    if home_team_col is not None:
        mapping[home_team_col] = "team_home"
    if away_team_col is not None:
        mapping[away_team_col] = "team_away"

    if home_goals_col is not None and home_goals_col != home_team_col:
        mapping[home_goals_col] = "goals_home"
    if away_goals_col is not None and away_goals_col != away_team_col:
        mapping[away_goals_col] = "goals_away"
    if date_col is not None:
        mapping[date_col] = "date"
    if result_col is not None:
        mapping[result_col] = "result"
    if score_col is not None:
        mapping[score_col] = "score"

    normalized = normalized.rename(columns=mapping)

    if "score" in normalized.columns and "goals_home" not in normalized.columns:
        def parse_score(s):
            if pd.isna(s):
                return pd.Series([None, None])
            try:
                left, right = str(s).split("-")
                return pd.Series([int(left.strip()), int(right.strip())])
            except Exception:
                return pd.Series([None, None])

        normalized[["goals_home", "goals_away"]] = normalized["score"].apply(parse_score)

    if "goals_home" not in normalized.columns or "goals_away" not in normalized.columns:
        return pd.DataFrame()

    for col in ["goals_home", "goals_away"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "result" not in normalized.columns:
        def encode_result(row):
            if pd.isna(row["goals_home"]) or pd.isna(row["goals_away"]):
                return None
            if row["goals_home"] > row["goals_away"]:
                return 1
            if row["goals_home"] < row["goals_away"]:
                return 2
            return 0

        normalized["result"] = normalized.apply(encode_result, axis=1)
    else:
        normalized["result"] = pd.to_numeric(normalized["result"], errors="coerce")

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce", dayfirst=True)
    else:
        normalized["date"] = pd.NaT

    normalized["source"] = source
    normalized = normalized[["team_home", "team_away", "goals_home", "goals_away", "result", "date", "source"]]
    return normalized


def load_local(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        print(f"[WARN] No se pudo leer {path}: {exc}")
        return pd.DataFrame()


def is_finalized(row):
    if pd.isna(row["goals_home"]) or pd.isna(row["goals_away"]):
        return False
    if row["date"] and row["date"] > datetime.utcnow():
        if row["goals_home"] == 0 and row["goals_away"] == 0:
            return False
    return True


all_frames = []
for source in sources:
    if source["type"] == "local":
        df = load_local(source["path"])
    else:
        df = download_csv(source["url"])
    norm = normalize(df, source["name"])
    if norm.empty:
        print(f"[WARN] Normalización vacía para {source['name']}")
    else:
        print(f"[INFO] {source['name']} normalizado: {len(norm)} filas")
        all_frames.append(norm)

for url in football_data_urls:
    df = download_csv(url)
    name = Path(url).stem
    norm = normalize(df, name)
    if not norm.empty:
        print(f"[INFO] {name} normalizado: {len(norm)} filas")
        all_frames.append(norm)

if not all_frames:
    print("[ERROR] No se obtuvieron datos de ninguna fuente.")
    sys.exit(1)

combined = pd.concat(all_frames, ignore_index=True)
print(f"[INFO] Total raw rows before cleaning: {len(combined)}")

combined = combined.dropna(subset=["team_home", "team_away"])
combined = combined[(combined["team_home"].astype(str).str.strip() != "") & (combined["team_away"].astype(str).str.strip() != "")]
combined = combined[combined.apply(is_finalized, axis=1)]

before = len(combined)
combined = combined.drop_duplicates(subset=["team_home", "team_away", "date", "goals_home", "goals_away", "result"])
after = len(combined)
print(f"[INFO] Finalized rows after cleaning and duplicate removal: {after} (removed {before-after})")

processed_path = Path(config["paths"]["data_processed"])
processed_path.mkdir(parents=True, exist_ok=True)
master_path = processed_path / "matches_master.csv"
combined.to_csv(master_path, index=False)
print(f"[INFO] Master dataset saved to {master_path}")

print(combined["result"].value_counts(dropna=False).to_dict())
print(combined.head(50).to_string(index=False))
print(f"[INFO] Unique sources: {combined['source'].unique().tolist()}")
