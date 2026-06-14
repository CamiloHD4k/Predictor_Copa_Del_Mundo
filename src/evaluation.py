from typing import Dict, Any


class Evaluator:
    """Evalúa y compara modelos entrenados."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def compare_models(self, models: Dict[str, Dict[str, Any]], data: Any) -> None:
        """Imprime métricas comparativas para cada modelo."""
        if not models:
            print("[WARN] No hay modelos entrenados para evaluar.")
            return

        print("\n[RESULTADOS] Comparativa de modelos:")
        for name, model_data in models.items():
            metrics = model_data.get("metrics", {})
            print(f"- {name}")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")
