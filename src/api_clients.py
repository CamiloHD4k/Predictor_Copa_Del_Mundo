"""Clientes de API externos para el proyecto World Cup Predictor."""

import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv


class FootballDataClient:
    """Cliente para Football-Data API."""

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent
        self._load_env()
        self.api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
        self.headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
        self.raw_path = Path(self.config["paths"]["data_raw"])
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def _load_env(self) -> None:
        env_path = self.project_root / ".env"
        fallback_path = self.project_root / "key.env"

        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"[INFO] Cargando variables de entorno desde {env_path}")
        elif fallback_path.exists():
            load_dotenv(dotenv_path=fallback_path)
            print(f"[INFO] Cargando variables de entorno desde {fallback_path}")
        else:
            print("[WARN] No se encontró .env ni key.env; la API key debe estar en el entorno.")

    def test_connection(self) -> bool:
        if not self.api_key:
            print("[ERROR] FOOTBALL_DATA_API_KEY no encontrado en .env")
            return False

        try:
            response = requests.get(
                f"{self.BASE_URL}/competitions",
                headers=self.headers,
                timeout=15,
            )
            if response.status_code == 200:
                return True
            print(f"[ERROR] Football-Data respondió con status {response.status_code}: {response.text}")
            return False
        except requests.RequestException as exc:
            print(f"[ERROR] Error de conexión a Football-Data: {exc}")
            return False

    def download_matches(self) -> pd.DataFrame:
        competitions = self.config.get("football_data", {}).get("competitions", ["WC"])
        matches_frames = []

        for competition in competitions:
            try:
                response = requests.get(
                    f"{self.BASE_URL}/competitions/{competition}/matches",
                    headers=self.headers,
                    timeout=30,
                )
            except requests.RequestException as exc:
                print(f"[ERROR] No se pudo obtener partidos de {competition}: {exc}")
                continue

            if response.status_code != 200:
                print(f"[ERROR] Football-Data {competition} responded with {response.status_code}: {response.text}")
                continue

            payload = response.json()
            matches = payload.get("matches", [])
            if not matches:
                print(f"[WARN] No se encontraron partidos para competencia {competition}")
                continue

            records = []
            for match in matches:
                home = match["homeTeam"]["name"]
                away = match["awayTeam"]["name"]
                score = match.get("score", {})
                full_time = score.get("fullTime", {})
                home_goals = full_time.get("home")
                away_goals = full_time.get("away")

                records.append(
                    {
                        "team_home": home,
                        "team_away": away,
                        "goals_home": home_goals if home_goals is not None else 0,
                        "goals_away": away_goals if away_goals is not None else 0,
                        "fifa_rank_home": 0,
                        "fifa_rank_away": 0,
                        "elo_home": 0,
                        "elo_away": 0,
                        "form_home": 0,
                        "form_away": 0,
                        "goals_scored_avg_home": 0,
                        "goals_scored_avg_away": 0,
                        "goals_conceded_avg_home": 0,
                        "goals_conceded_avg_away": 0,
                        "home_average_age": 0,
                        "away_average_age": 0,
                        "home_market_value": 0,
                        "away_market_value": 0,
                        "result": self._encode_result(home_goals, away_goals),
                    }
                )

            matches_frames.append(pd.DataFrame.from_records(records))

        if not matches_frames:
            return pd.DataFrame()

        matches_df = pd.concat(matches_frames, ignore_index=True)
        file_path = self.raw_path / "matches_real.csv"
        matches_df.to_csv(file_path, index=False)
        print(f"[INFO] Guardado de partidos reales en {file_path}")
        return matches_df

    def _encode_result(self, home_goals: Any, away_goals: Any) -> int:
        if home_goals is None or away_goals is None:
            return 0
        if home_goals > away_goals:
            return 1
        if home_goals < away_goals:
            return 2
        return 0
