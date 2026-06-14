import os
import pandas as pd
from typing import Dict


class DataLoader:
    """Carga y prepara datos para el sistema de predicción."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.data_path = config["paths"]["data_raw"]

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carga los conjuntos de datos disponibles desde la carpeta de datos raw."""
        data_files = {
            "matches": "matches.csv",
            "rankings": "rankings.csv",
            "market": "market.csv",
        }

        loaded_data = {}
        for key, filename in data_files.items():
            file_path = os.path.join(self.data_path, filename)
            if os.path.exists(file_path):
                try:
                    loaded_data[key] = pd.read_csv(file_path)
                    print(f"[INFO] Cargado {filename} ({loaded_data[key].shape[0]} filas)")
                except Exception as exc:
                    print(f"[ERROR] No se pudo cargar {filename}: {exc}")
                    loaded_data[key] = pd.DataFrame()
            else:
                print(f"[WARN] No se encontró {filename} en {self.data_path}")
                loaded_data[key] = pd.DataFrame()

        return loaded_data
