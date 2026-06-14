import os
import sys
from copy import deepcopy
# ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.api_clients import FootballDataClient
from src.config import load_config
import pandas as pd

config = load_config()
# candidate international competitions (try many; failures are logged)
competitions = [
    "WC",
    "EC",
    "UNL",
    "WCQ",
    "FIFA",
    "EURO",
    "CONMEBOL",
    "CONMEBOL_PRELIM",
    "FRIENDLY",
    "INT",
]

cfg = deepcopy(config)
cfg.setdefault("football_data", {})
cfg["football_data"]["competitions"] = competitions

client = FootballDataClient(cfg)
all_matches = []

print("[SCRIPT] Intentando descargar competiciones:")
for comp in competitions:
    try:
        print(f"[SCRIPT] -> {comp}")
        # call client for single competition by temporarily setting config
        client.config["football_data"]["competitions"] = [comp]
        df = client.download_matches()
        if df is None:
            df = pd.DataFrame()
        df["_source_competition"] = comp
        all_matches.append(df)
    except Exception as e:
        print(f"[SCRIPT] Error descargando {comp}: {e}")

if not all_matches:
    print("[SCRIPT] No se descargaron partidos de ninguna competición.")
    raise SystemExit(0)

combined = pd.concat(all_matches, ignore_index=True)
print(f"[SCRIPT] Total rows downloaded before cleaning: {len(combined)}")

# Clean: drop rows missing team names or empty strings
before_clean = len(combined)
combined = combined.dropna(subset=["team_home", "team_away"]) if "team_home" in combined.columns else combined
combined = combined[(combined["team_home"] != "") & (combined["team_away"] != "")] if "team_home" in combined.columns else combined
after_clean = len(combined)
print(f"[SCRIPT] Rows after cleaning: {after_clean} (discarded {before_clean - after_clean})")

# Save combined
out_path = os.path.join(config["paths"]["data_raw"], "matches_combined.csv")
combined.to_csv(out_path, index=False)
print(f"[SCRIPT] Combined saved to {out_path}")

# Show first 50 rows with requested cols if present
cols = ["team_home", "team_away", "goals_home", "goals_away", "result"]
available = [c for c in cols if c in combined.columns]
print(f"[SCRIPT] Available columns: {available}")
print("[SCRIPT] First 50 rows:\n")
print(combined[available].head(50).to_string(index=False))

# Verify encode_result by recomputing
from src.api_clients import FootballDataClient
client2 = FootballDataClient(config)
if "goals_home" in combined.columns and "goals_away" in combined.columns:
    recomputed = combined.apply(lambda r: client2._encode_result(r.get("goals_home"), r.get("goals_away")), axis=1)
    if "result" in combined.columns:
        mism = (combined["result"].fillna(-1).astype(int) != recomputed.astype(int)).sum()
        print(f"[SCRIPT] Mismatches between file 'result' and recomputed: {mism}")

# Show distribution
if "result" in combined.columns:
    dist = combined["result"].value_counts().to_dict()
    print(f"[SCRIPT] Distribution (result): {dist}")

# If not enough rows, report and exit (we'll attempt many competitions already)
print(f"[SCRIPT] Combined rows final: {len(combined)}")

