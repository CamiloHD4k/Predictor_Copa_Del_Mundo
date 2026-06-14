import os
import yaml
from typing import Any, Dict


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Carga la configuración del proyecto desde un archivo YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    project_root = os.path.abspath(os.path.dirname(__file__))
    config["paths"]["data_raw"] = os.path.normpath(
        os.path.join(project_root, "..", config["paths"]["data_raw"])
    )
    config["paths"]["data_processed"] = os.path.normpath(
        os.path.join(project_root, "..", config["paths"]["data_processed"])
    )
    config["paths"]["models"] = os.path.normpath(
        os.path.join(project_root, "..", config["paths"]["models"])
    )
    config["paths"]["reports"] = os.path.normpath(
        os.path.join(project_root, "..", config["paths"]["reports"])
    )

    return config
