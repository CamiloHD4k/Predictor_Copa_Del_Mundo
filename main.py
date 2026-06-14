"""Punto de entrada del proyecto World Cup Predictor.

Carga la configuración, ejecuta la canalización de datos, entrena modelos y realiza simulaciones.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.api_clients import FootballDataClient
from src.config import load_config
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.modeling import ModelTrainer
from src.simulation import TournamentSimulator
from src.evaluation import Evaluator
from groups_2026 import GroupStageSimulator


def main() -> None:
    """Ejecuta el flujo principal del sistema de predicción."""
    print("[INFO] Inicio de ejecución principal")
    config = load_config()

    print("[INFO] Probando conexión con Football-Data...")
    api_client = FootballDataClient(config)
    use_real_data = False
    real_matches = pd.DataFrame()

    if api_client.test_connection():
        print("[INFO] Conexión a Football-Data exitosa.")
        real_matches = api_client.download_matches()
        # Clean downloaded data: drop rows without valid team names
        if not real_matches.empty:
            real_matches = real_matches.dropna(subset=["team_home", "team_away"])
            real_matches = real_matches[(real_matches["team_home"] != "") & (real_matches["team_away"] != "")]
        print(f"[INFO] Partidos reales descargados: {len(real_matches)}")
        if len(real_matches) >= 100:
            print("[INFO] Usando datos reales de Football-Data para entrenamiento.")
            use_real_data = True
        else:
            print("[ERROR] Se descargaron menos de 100 partidos reales. Se detendrá el entrenamiento.")
            return
    else:
        print("[WARN] No se pudo conectar a Football-Data. Usando datos de ejemplo local.")

    print("[INFO] Cargando datos locales...")
    loader = DataLoader(config)
    data = loader.load_data()

    if use_real_data:
        data["matches"] = real_matches

    print("[INFO] Generando variables predictivas...")
    engineer = FeatureEngineer(config)
    features = engineer.build_features(data)
    print(f"[INFO] Features generados: {features.shape if hasattr(features, 'shape') else 'no dataframe'}")

    print("[INFO] Entrenando modelos...")
    trainer = ModelTrainer(config)
    models = trainer.train_all(features)
    print(f"[INFO] Modelos entrenados: {list(models.keys())}")

    print("[INFO] Evaluando resultados...")
    evaluator = Evaluator(config)
    evaluator.compare_models(models, features)

    if config.get("simulation", {}).get("group_stage", False):
        print("[INFO] Simulando fase de grupos 2026...")
        group_simulator = GroupStageSimulator(seed=config["model"]["random_state"])
        group_table = group_simulator.simulate()
        print("[INFO] Clasificación de la fase de grupos 2026:")
        print(group_table.to_string(index=False))

    print("[INFO] Iniciando simulación de torneo...")
    simulator = TournamentSimulator(config, models)
    simulator.run_monte_carlo(n_simulations=config["simulation"].get("n_simulations", 1000))

    print("[INFO] Proceso finalizado.")


if __name__ == "__main__":
    main()
