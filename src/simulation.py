import random
import math
from typing import Dict, Any


class TournamentSimulator:
    """Simula el torneo con un enfoque Monte Carlo para estimar probabilidades."""

    def __init__(self, config: Dict[str, Any], models: Dict[str, Any]) -> None:
        self.config = config
        self.models = models
        self.n_simulations = config["simulation"]["n_simulations"]

    def _softmax(self, scores: list[float]) -> list[float]:
        exp_scores = [math.exp(score) for score in scores]
        total = sum(exp_scores)
        return [s / total for s in exp_scores]

    def _predict_match_outcome(self, home_strength: float, away_strength: float) -> str:
        home_score = 1.2 * home_strength - 0.5 * away_strength
        draw_score = 0.6 - abs(home_strength - away_strength) * 0.3
        away_score = 1.2 * away_strength - 0.5 * home_strength
        probs = self._softmax([home_score, draw_score, away_score])
        outcome = random.choices(["home", "draw", "away"], weights=probs, k=1)[0]
        return outcome

    def run_monte_carlo(self, n_simulations: int = None) -> None:
        """Ejecuta simulaciones Monte Carlo y muestra conteos de resultados."""
        n = n_simulations if n_simulations is not None else self.n_simulations
        if not self.models:
            print("[WARN] No hay modelos disponibles para simular.")
            return

        print(f"[INFO] Ejecutando {n} simulaciones Monte Carlo...")
        outcomes = {"home": 0, "away": 0, "draw": 0}

        for _ in range(n):
            home_strength = random.uniform(0.2, 1.0)
            away_strength = random.uniform(0.2, 1.0)
            result = self._predict_match_outcome(home_strength, away_strength)
            outcomes[result] += 1

        print("[INFO] Resultados simulados:")
        print(outcomes)
