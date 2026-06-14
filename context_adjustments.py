from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple


ALTITUDE_ADAPTED_TEAMS = {
    "Mexico",
    "México",
    "Estados Unidos",
    "United States",
    "USA",
    "Ecuador",
    "Colombia",
}


def _normalize_probabilities(probabilities: Iterable[float]) -> Tuple[float, float, float]:
    values = [max(0.01, float(value)) for value in probabilities]
    total = sum(values)
    if total <= 0:
        return 0.33, 0.34, 0.33
    return values[0] / total, values[1] / total, values[2] / total


def _move_probability(probabilities: List[float], from_index: int, to_index: int, amount: float) -> None:
    shift = min(max(amount, 0.0), max(0.0, probabilities[from_index] - 0.03))
    probabilities[from_index] -= shift
    probabilities[to_index] += shift


def _team_is_altitude_adapted(team: str) -> bool:
    return team in ALTITUDE_ADAPTED_TEAMS


def apply_context_adjustments(
    base_probabilities,
    expected_goals_home: float,
    expected_goals_away: float,
    diagnostics: Mapping[str, object],
    context: Mapping[str, object],
):
    """Apply moderate external-context adjustments after the base model.

    Probabilities are ordered as [draw, home win, away win].
    """
    probabilities = list(_normalize_probabilities(base_probabilities))
    home_xg = float(expected_goals_home)
    away_xg = float(expected_goals_away)
    explanations: List[str] = []

    home_team = str(diagnostics.get("home_team", ""))
    away_team = str(diagnostics.get("away_team", ""))
    venue_type = str(context.get("venue_type", context.get("match_context", "Mundial 2026 neutral")))
    city = str(context.get("city", "Sin especificar"))
    weather = str(context.get("weather", "Normal"))
    rest = str(context.get("rest", "Ambos con descanso normal"))
    importance = str(context.get("importance", "Fase de grupos"))
    fatigue = str(context.get("fatigue", "Normal"))

    if venue_type == "Mundial 2026 neutral":
        explanations.append("Sede neutral: no se agrega localía al primer equipo, salvo anfitriones del Mundial 2026.")

    if city == "Ciudad de México":
        if not _team_is_altitude_adapted(home_team):
            home_xg *= 0.96
            _move_probability(probabilities, 1, 0, 0.010)
            explanations.append(f"Altitud en Ciudad de México: leve reducción física para {home_team}.")
        if not _team_is_altitude_adapted(away_team):
            away_xg *= 0.96
            _move_probability(probabilities, 2, 0, 0.010)
            explanations.append(f"Altitud en Ciudad de México: leve reducción física para {away_team}.")

    if weather == "Calor extremo":
        home_xg *= 0.95
        away_xg *= 0.95
        probabilities[0] *= 1.025
        explanations.append("Calor extremo: baja moderada del ritmo y de los goles esperados totales.")
    elif weather == "Alta humedad":
        home_xg *= 0.96
        away_xg *= 0.96
        probabilities[0] *= 1.020
        explanations.append("Alta humedad: leve aumento de fatiga y partido algo más lento.")
    elif weather == "Frío intenso":
        home_xg *= 0.97
        away_xg *= 0.97
        probabilities[0] *= 1.015
        explanations.append("Frío intenso: ligera reducción de producción ofensiva.")
    elif weather == "Lluvia":
        home_xg *= 0.94
        away_xg *= 0.94
        probabilities[0] *= 1.035
        explanations.append("Lluvia/campo pesado: menor precisión ofensiva y mayor probabilidad de marcador bajo.")
    elif weather == "Viento fuerte":
        home_xg *= 0.95
        away_xg *= 0.95
        probabilities[0] *= 1.020
        explanations.append("Viento fuerte: menor efectividad ofensiva y mayor incertidumbre en centros/remates.")

    if rest == "Local con más descanso":
        away_xg *= 0.96
        _move_probability(probabilities, 2, 1, 0.012)
        explanations.append("Descanso: el local llega con leve ventaja física.")
    elif rest == "Visitante con más descanso":
        home_xg *= 0.96
        _move_probability(probabilities, 1, 2, 0.012)
        explanations.append("Descanso: el visitante llega con leve ventaja física.")
    elif rest == "Ambos con poco descanso":
        home_xg *= 0.96
        away_xg *= 0.96
        probabilities[0] *= 1.025
        explanations.append("Poco descanso para ambos: menor ritmo y algo más de empate.")

    if fatigue == "Local más fatigado":
        home_xg *= 0.95
        _move_probability(probabilities, 1, 2, 0.015)
        explanations.append("Fatiga/viaje: reducción moderada del rendimiento local.")
    elif fatigue == "Visitante más fatigado":
        away_xg *= 0.95
        _move_probability(probabilities, 2, 1, 0.015)
        explanations.append("Fatiga/viaje: reducción moderada del rendimiento visitante.")
    elif fatigue == "Ambos fatigados":
        home_xg *= 0.96
        away_xg *= 0.96
        probabilities[0] *= 1.025
        explanations.append("Fatiga en ambos equipos: partido más lento y de menor volumen ofensivo.")

    if importance == "Partido decisivo de grupo":
        home_xg *= 0.96
        away_xg *= 0.96
        probabilities[0] *= 1.025
        explanations.append("Partido decisivo: tensión alta y enfoque algo más conservador.")
    elif importance == "Eliminación directa":
        home_xg *= 0.93
        away_xg *= 0.93
        probabilities[0] *= 1.045
        explanations.append("Eliminación directa: aumenta la cautela y el empate en 90 minutos.")
    elif importance == "Final":
        home_xg *= 0.91
        away_xg *= 0.91
        probabilities[0] *= 1.055
        explanations.append("Final: máxima tensión, menor riesgo y goles esperados ligeramente menores.")

    adjusted_probabilities = _normalize_probabilities(probabilities)
    if not explanations:
        explanations.append("Sin ajustes contextuales adicionales relevantes.")

    return adjusted_probabilities, (home_xg, away_xg), explanations
