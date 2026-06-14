import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

from auto_context import get_auto_context, load_world_cup_fixtures, validate_world_cup_fixtures
from groups_2026 import GROUPS_2026
from predict_match import (
    HOST_TEAMS,
    MATCH_CONTEXT_HOST_COUNTRY,
    MATCH_CONTEXT_GENERAL,
    MATCH_CONTEXT_NORMAL_HOME,
    MATCH_CONTEXT_WORLD_CUP_NEUTRAL,
    explain_prediction,
    load_team_stats,
    predict_exact_scores,
    predict_match,
)

st.set_page_config(
    page_title="Predicción Mundial 2026",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --wc-surface: var(--background-color);
            --wc-panel-bg: var(--secondary-background-color);
            --wc-text: var(--text-color);
            --wc-muted: color-mix(in srgb, var(--text-color) 68%, transparent);
            --wc-line: color-mix(in srgb, var(--text-color) 20%, transparent);
            --wc-soft-line: color-mix(in srgb, var(--text-color) 12%, transparent);
            --wc-accent: var(--primary-color);
            --wc-accent-soft: color-mix(in srgb, var(--primary-color) 20%, var(--secondary-background-color));
            --wc-home: color-mix(in srgb, var(--primary-color) 82%, var(--text-color));
            --wc-draw: color-mix(in srgb, var(--primary-color) 48%, var(--secondary-background-color));
            --wc-away: color-mix(in srgb, var(--primary-color) 72%, var(--text-color));
        }
        .stApp {background: var(--wc-surface);}
        .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1320px;}
        .wc-hero {
            background: linear-gradient(135deg, var(--wc-panel-bg) 0%, var(--wc-accent-soft) 100%);
            color: var(--wc-text);
            border: 1px solid var(--wc-line);
            border-radius: 10px;
            padding: 30px 34px;
            margin: 0 0 22px 0;
            box-shadow: 0 18px 40px color-mix(in srgb, var(--wc-text) 12%, transparent);
        }
        .wc-hero h1 {
            margin: 0 0 8px 0;
            font-size: clamp(34px, 5vw, 58px);
            line-height: 1;
            letter-spacing: 0;
        }
        .wc-hero p {margin: 0; color: var(--wc-muted); font-size: 18px;}
        .wc-section-title {
            color: var(--wc-text);
            border-left: 5px solid var(--wc-accent);
            padding-left: 12px;
            margin: 24px 0 12px 0;
            font-size: 22px;
            font-weight: 750;
        }
        .wc-card {
            background: var(--wc-panel-bg);
            border: 1px solid var(--wc-line);
            border-radius: 8px;
            padding: 18px;
            min-height: 130px;
            box-shadow: 0 10px 26px color-mix(in srgb, var(--wc-text) 9%, transparent);
        }
        .wc-card-label {
            color: var(--wc-muted);
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }
        .wc-card-main {
            color: var(--wc-text);
            font-size: 30px;
            font-weight: 850;
            line-height: 1.05;
            overflow-wrap: anywhere;
        }
        .wc-card-sub {color: var(--wc-muted); font-size: 14px; margin-top: 10px;}
        .wc-card-home {border-top: 5px solid var(--wc-home);}
        .wc-card-draw {border-top: 5px solid var(--wc-draw);}
        .wc-card-away {border-top: 5px solid var(--wc-away);}
        .wc-card-neutral {border-top: 5px solid var(--wc-accent);}
        .wc-probbar {
            display: flex;
            width: 100%;
            height: 40px;
            overflow: hidden;
            border-radius: 8px;
            border: 1px solid var(--wc-line);
            background: var(--wc-panel-bg);
            margin: 10px 0 8px 0;
        }
        .wc-probseg {
            color: var(--wc-text);
            font-weight: 800;
            font-size: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            min-width: 44px;
        }
        .wc-prob-home {background: color-mix(in srgb, var(--wc-home) 70%, var(--wc-panel-bg));}
        .wc-prob-draw {background: color-mix(in srgb, var(--wc-draw) 72%, var(--wc-panel-bg));}
        .wc-prob-away {background: color-mix(in srgb, var(--wc-away) 70%, var(--wc-panel-bg));}
        .wc-panel {
            background: var(--wc-panel-bg);
            border: 1px solid var(--wc-line);
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 8px 22px color-mix(in srgb, var(--wc-text) 8%, transparent);
        }
        .wc-footer {
            border-top: 1px solid var(--wc-line);
            color: var(--wc-muted);
            text-align: center;
            padding-top: 20px;
            margin-top: 34px;
            font-size: 14px;
        }
        div[data-testid="stMetric"] {
            background: var(--wc-panel-bg);
            border: 1px solid var(--wc-line);
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--wc-line);
            border-radius: 8px;
        }
        .wc-theme-note {
            color: var(--wc-muted);
            font-size: 13px;
            margin-top: -8px;
            margin-bottom: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="wc-hero">
        <h1>Predicción Mundial 2026</h1>
        <p>Dashboard deportivo con Machine Learning, ELO, ranking FIFA, Poisson/xG y contexto real del partido.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

theme_auto_mode = st.toggle(
    "Modo automático",
    value=True,
    help="Respeta automáticamente el tema claro u oscuro configurado en Streamlit.",
)
if theme_auto_mode:
    st.markdown(
        '<div class="wc-theme-note">Tema adaptativo activo: colores y contraste se ajustan al tema de Streamlit.</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="wc-theme-note">Tema manual desactivado: la interfaz mantiene colores adaptativos para preservar accesibilidad.</div>',
        unsafe_allow_html=True,
    )

team_stats = load_team_stats()
world_cup_teams = {team for group in GROUPS_2026.values() for team in group}
team_list = sorted(set(team_stats.keys()) | world_cup_teams)
fixtures_df = load_world_cup_fixtures()
fixture_validation_errors = validate_world_cup_fixtures(fixtures_df)
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
PREDICTION_HISTORY_PATH = REPORTS_DIR / "prediction_history.csv"

VENUE_TYPE_OPTIONS = ["Mundial 2026 neutral", "Localía normal", "País anfitrión", "Predicción general"]
CITY_OPTIONS = [
    "Sin especificar",
    "Ciudad de México",
    "Guadalajara",
    "Monterrey",
    "Toronto",
    "Vancouver",
    "Nueva York/Nueva Jersey",
    "Los Ángeles",
    "Dallas",
    "Houston",
    "Miami",
    "Atlanta",
    "Kansas City",
    "Boston",
    "Philadelphia",
    "Seattle",
    "San Francisco Bay Area",
]
WEATHER_OPTIONS = ["Normal", "Calor extremo", "Frío intenso", "Lluvia", "Viento fuerte", "Alta humedad", "No disponible"]
REST_OPTIONS = ["Ambos con descanso normal", "Local con más descanso", "Visitante con más descanso", "Ambos con poco descanso", "No disponible"]
IMPORTANCE_OPTIONS = ["Fase de grupos", "Partido decisivo de grupo", "Eliminación directa", "Final", "No disponible"]
FATIGUE_OPTIONS = ["Normal", "Local más fatigado", "Visitante más fatigado", "Ambos fatigados"]


def option_index(options, value, default=0):
    return options.index(value) if value in options else default


def confidence_label(confidence):
    if confidence >= 0.60:
        return "Alta"
    if confidence >= 0.45:
        return "Media"
    return "Baja"


def pct(value, decimals=1):
    return f"{value * 100:.{decimals}f}%"


def section_title(title):
    st.markdown(f'<div class="wc-section-title">{title}</div>', unsafe_allow_html=True)


def metric_card(label, main_value, sub_value="", class_name="wc-card-neutral"):
    st.markdown(
        f"""
        <div class="wc-card {class_name}">
            <div class="wc-card-label">{label}</div>
            <div class="wc-card-main">{main_value}</div>
            <div class="wc-card-sub">{sub_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bar(prediction):
    home_pct = prediction["probabilidad_victoria_local"] * 100
    draw_pct = prediction["probabilidad_empate"] * 100
    away_pct = prediction["probabilidad_victoria_visitante"] * 100
    st.markdown(
        f"""
        <div class="wc-probbar">
            <div class="wc-probseg wc-prob-home" style="width:{home_pct:.2f}%">Local {home_pct:.1f}%</div>
            <div class="wc-probseg wc-prob-draw" style="width:{draw_pct:.2f}%">Empate {draw_pct:.1f}%</div>
            <div class="wc-probseg wc-prob-away" style="width:{away_pct:.2f}%">Visitante {away_pct:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(prediction, exact_scores):
    best_score = exact_scores[0]["marcador"] if exact_scores else "N/A"
    card_cols = st.columns(5)
    with card_cols[0]:
        metric_card(
            prediction["home_team"],
            pct(prediction["probabilidad_victoria_local"]),
            "Victoria local",
            "wc-card-home",
        )
    with card_cols[1]:
        metric_card("Empate", pct(prediction["probabilidad_empate"]), "Resultado X", "wc-card-draw")
    with card_cols[2]:
        metric_card(
            prediction["away_team"],
            pct(prediction["probabilidad_victoria_visitante"]),
            "Victoria visitante",
            "wc-card-away",
        )
    with card_cols[3]:
        metric_card("Marcador más probable", best_score, prediction["resultado_mas_probable"], "wc-card-neutral")
    with card_cols[4]:
        metric_card(
            "Confianza del modelo",
            confidence_label(prediction["confianza_modelo"]),
            pct(prediction["confianza_modelo"]),
            "wc-card-neutral",
        )


def exact_scores_table(exact_scores, limit=10):
    exact_df = pd.DataFrame(exact_scores[:limit]).copy()
    exact_df["Probabilidad (%)"] = exact_df["probabilidad"] * 100
    exact_df = exact_df[["marcador", "Probabilidad (%)", "resultado"]]
    exact_df.columns = ["Marcador", "Probabilidad (%)", "Resultado"]
    exact_df = exact_df.sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)

    def highlight_top(row):
        if row.name == 0:
            return [
                "background-color: var(--secondary-background-color); color: var(--text-color); font-weight: 800"
            ] * len(row)
        return [""] * len(row)

    return exact_df.style.format({"Probabilidad (%)": "{:.2f}"}).apply(highlight_top, axis=1)


def load_prediction_history():
    if PREDICTION_HISTORY_PATH.exists():
        return pd.read_csv(PREDICTION_HISTORY_PATH)
    return pd.DataFrame()


def append_prediction_history(prediction, exact_scores, external_context):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    diagnostic = prediction.get("diagnostic", {})
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "home_team": prediction.get("home_team"),
        "away_team": prediction.get("away_team"),
        "fixture_found": external_context.get("fixture_found"),
        "competition": external_context.get("competition"),
        "date": external_context.get("date"),
        "venue": external_context.get("venue"),
        "city": external_context.get("city"),
        "prob_home": prediction.get("probabilidad_victoria_local"),
        "prob_draw": prediction.get("probabilidad_empate"),
        "prob_away": prediction.get("probabilidad_victoria_visitante"),
        "confidence": prediction.get("confianza_modelo"),
        "confidence_label": confidence_label(prediction.get("confianza_modelo", 0)),
        "expected_goals_home": prediction.get("marcador_esperado", {}).get("home"),
        "expected_goals_away": prediction.get("marcador_esperado", {}).get("away"),
        "result": prediction.get("resultado_mas_probable"),
        "top_score": exact_scores[0]["marcador"] if exact_scores else "",
        "home_rank": diagnostic.get("home_rank"),
        "away_rank": diagnostic.get("away_rank"),
        "home_elo": diagnostic.get("home_elo"),
        "away_elo": diagnostic.get("away_elo"),
    }
    history = load_prediction_history()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(PREDICTION_HISTORY_PATH, index=False)


def prediction_export_frame(prediction, exact_scores, external_context):
    diagnostic = prediction.get("diagnostic", {})
    return pd.DataFrame(
        [
            {
                "home_team": prediction.get("home_team"),
                "away_team": prediction.get("away_team"),
                "fixture_found": external_context.get("fixture_found"),
                "date": external_context.get("date"),
                "venue": external_context.get("venue"),
                "city": external_context.get("city"),
                "prob_home": prediction.get("probabilidad_victoria_local"),
                "prob_draw": prediction.get("probabilidad_empate"),
                "prob_away": prediction.get("probabilidad_victoria_visitante"),
                "confidence": prediction.get("confianza_modelo"),
                "confidence_label": confidence_label(prediction.get("confianza_modelo", 0)),
                "xg_home": prediction.get("marcador_esperado", {}).get("home"),
                "xg_away": prediction.get("marcador_esperado", {}).get("away"),
                "top_scores": "; ".join(f"{row['marcador']} ({row['probabilidad'] * 100:.1f}%)" for row in exact_scores[:5]),
                "home_rank": diagnostic.get("home_rank"),
                "away_rank": diagnostic.get("away_rank"),
                "home_elo": diagnostic.get("home_elo"),
                "away_elo": diagnostic.get("away_elo"),
                "explanation": prediction.get("explanation", ""),
            }
        ]
    )


def build_match_report(prediction, exact_scores, external_context):
    diagnostic = prediction.get("diagnostic", {})
    lines = [
        f"Informe: {prediction['home_team']} vs {prediction['away_team']}",
        f"Contexto: {diagnostic.get('match_context', 'N/A')} | Sede: {external_context.get('venue', 'N/A')} | Ciudad: {external_context.get('city', 'N/A')}",
        f"Probabilidades ajustadas: local {prediction['probabilidad_victoria_local'] * 100:.1f}%, empate {prediction['probabilidad_empate'] * 100:.1f}%, visitante {prediction['probabilidad_victoria_visitante'] * 100:.1f}%",
        f"xG esperado: {prediction['marcador_esperado']['home']:.2f} - {prediction['marcador_esperado']['away']:.2f}",
        f"Confianza: {confidence_label(prediction['confianza_modelo'])} ({prediction['confianza_modelo'] * 100:.1f}%)",
        f"Marcador más probable: {exact_scores[0]['marcador'] if exact_scores else 'N/A'}",
        f"Ranking FIFA: {diagnostic.get('home_rank', 'N/A')} vs {diagnostic.get('away_rank', 'N/A')}",
        f"ELO: {diagnostic.get('home_elo', 'N/A')} vs {diagnostic.get('away_elo', 'N/A')}",
    ]
    if diagnostic.get("context_adjustments"):
        lines.append("Ajustes contextuales: " + "; ".join(diagnostic["context_adjustments"]))
    return "\n".join(lines)


def team_comparison_frame(prediction):
    diagnostic = prediction.get("diagnostic", {})
    return pd.DataFrame(
        {
            "Métrica": ["Ranking FIFA", "ELO", "Valor plantilla (€m)", "Edad promedio", "xG esperado"],
            prediction["home_team"]: [
                diagnostic.get("home_rank", "N/A"),
                diagnostic.get("home_elo", "N/A"),
                diagnostic.get("home_squad_value_eur_m", "N/A"),
                diagnostic.get("home_average_age", "N/A"),
                prediction.get("marcador_esperado", {}).get("home", "N/A"),
            ],
            prediction["away_team"]: [
                diagnostic.get("away_rank", "N/A"),
                diagnostic.get("away_elo", "N/A"),
                diagnostic.get("away_squad_value_eur_m", "N/A"),
                diagnostic.get("away_average_age", "N/A"),
                prediction.get("marcador_esperado", {}).get("away", "N/A"),
            ],
        }
    )


def data_warnings(prediction, external_context):
    warnings = []
    diagnostic = prediction.get("diagnostic", {})
    if not external_context.get("fixture_found"):
        warnings.append("No hay fixture oficial cargado para este partido.")
    for key in ["home_rank", "away_rank", "home_elo", "away_elo", "home_squad_value_eur_m", "away_squad_value_eur_m"]:
        if diagnostic.get(key) in [None, "", "N/A"]:
            warnings.append(f"Falta dato: {key}")
    return warnings


def simulate_group(group, external_context):
    group_fixtures = fixtures_df[fixtures_df["group"] == group].copy()
    teams = sorted(set(group_fixtures["team_home"]) | set(group_fixtures["team_away"]))
    table = {team: {"Equipo": team, "Pts esperados": 0.0, "GF esperados": 0.0, "GC esperados": 0.0} for team in teams}
    for _, row in group_fixtures.iterrows():
        context = dict(external_context)
        context.update(
            {
                "fixture_found": True,
                "competition": row.get("competition"),
                "date": row.get("date"),
                "venue": row.get("venue"),
                "city": row.get("city"),
                "country": row.get("country"),
                "importance": row.get("phase", "Fase de grupos"),
                "venue_type": "Mundial 2026 neutral",
                "weather": context.get("weather", "Normal"),
                "rest": context.get("rest", "Ambos con descanso normal"),
                "fatigue": context.get("fatigue", "Normal"),
            }
        )
        prediction = predict_match(row["team_home"], row["team_away"], match_context=MATCH_CONTEXT_WORLD_CUP_NEUTRAL, external_context=context)
        home = row["team_home"]
        away = row["team_away"]
        p_home = prediction["probabilidad_victoria_local"]
        p_draw = prediction["probabilidad_empate"]
        p_away = prediction["probabilidad_victoria_visitante"]
        table[home]["Pts esperados"] += 3 * p_home + p_draw
        table[away]["Pts esperados"] += 3 * p_away + p_draw
        table[home]["GF esperados"] += prediction["marcador_esperado"]["home"]
        table[home]["GC esperados"] += prediction["marcador_esperado"]["away"]
        table[away]["GF esperados"] += prediction["marcador_esperado"]["away"]
        table[away]["GC esperados"] += prediction["marcador_esperado"]["home"]
    result = pd.DataFrame(table.values())
    result["DG esperado"] = result["GF esperados"] - result["GC esperados"]
    return result.sort_values(["Pts esperados", "DG esperado"], ascending=False).reset_index(drop=True)


def render_confidence(confidence):
    label = confidence_label(confidence)
    message = f"Confianza {label}: {confidence * 100:.1f}%"
    if label == "Alta":
        st.success(message)
    elif label == "Media":
        st.warning(message)
    else:
        st.error(message)


def render_prediction_actions(prediction, exact_scores, external_context):
    render_confidence(prediction.get("confianza_modelo", 0))

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("Explicar predicción", key="explain_prediction_button"):
            st.session_state["show_prediction_explanation"] = True
    with action_col2:
        export_df = prediction_export_frame(prediction, exact_scores, external_context)
        filename_home = prediction.get("home_team", "local").replace(" ", "_")
        filename_away = prediction.get("away_team", "visitante").replace(" ", "_")
        st.download_button(
            "Exportar predicción a CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"prediccion_{filename_home}_vs_{filename_away}.csv",
            mime="text/csv",
            key="download_prediction_csv",
        )

    if st.session_state.get("show_prediction_explanation"):
        section_title("Explicación de la predicción")
        st.markdown('<div class="wc-panel">', unsafe_allow_html=True)
        st.write(prediction.get("explanation", "No hay explicación disponible."))
        context_adjustments = prediction.get("diagnostic", {}).get("context_adjustments", [])
        if context_adjustments:
            st.markdown("**Factores contextuales aplicados**")
            for adjustment in context_adjustments:
                st.write(f"- {adjustment}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_insights(prediction, exact_scores, external_context):
    warnings = data_warnings(prediction, external_context)
    if warnings:
        st.warning("Validaciones de datos: " + "; ".join(warnings))

    section_title("Comparador visual de equipos")
    comparison_df = team_comparison_frame(prediction)
    st.dataframe(comparison_df, use_container_width=True)

    chart_rows = []
    for metric in ["ELO", "Valor plantilla (€m)", "xG esperado"]:
        row = comparison_df[comparison_df["Métrica"] == metric]
        if row.empty:
            continue
        home_value = pd.to_numeric(row[prediction["home_team"]].iloc[0], errors="coerce")
        away_value = pd.to_numeric(row[prediction["away_team"]].iloc[0], errors="coerce")
        if pd.notna(home_value) and pd.notna(away_value):
            chart_rows.append(
                {
                    "Métrica": metric,
                    prediction["home_team"]: home_value,
                    prediction["away_team"]: away_value,
                }
            )
    if chart_rows:
        chart_df = pd.DataFrame(chart_rows).set_index("Métrica")
        st.bar_chart(chart_df)

    section_title("Resumen final del partido")
    st.text_area(
        "Informe",
        value=build_match_report(prediction, exact_scores, external_context),
        height=220,
        disabled=True,
        key="match_report_text_area",
    )


def apply_auto_context_to_session(auto_context):
    st.session_state["auto_context"] = auto_context
    if not auto_context.get("fixture_found"):
        return
    st.session_state["match_context_select"] = auto_context.get("venue_type", "Mundial 2026 neutral")
    st.session_state["city_select"] = auto_context.get("city", "Sin especificar")
    st.session_state["weather_select"] = auto_context.get("weather", "Normal")
    st.session_state["rest_select"] = auto_context.get("rest", "Ambos con descanso normal")
    st.session_state["importance_select"] = auto_context.get("importance", "Fase de grupos")
    st.session_state["fatigue_select"] = auto_context.get("fatigue", "Normal")
    if auto_context.get("host_team"):
        st.session_state["host_team_select"] = auto_context["host_team"]


def complete_context_from_selected_teams():
    team_home_value = st.session_state.get("team_home_select")
    team_away_value = st.session_state.get("team_away_select")
    if team_home_value and team_away_value and team_home_value != team_away_value:
        apply_auto_context_to_session(get_auto_context(team_home_value, team_away_value))


def show_not_found_details(context):
    st.warning("Este partido no existe en el fixture oficial del Mundial 2026. Se usará una predicción general entre selecciones.")
    detail = context.get("not_found_detail")
    if detail:
        st.write(detail)
    related = context.get("related_fixture_labels", [])
    if related:
        st.write("Partidos encontrados:")
        for fixture_label in related:
            st.write(f"- {fixture_label}")


match_context_map = {
    "Mundial 2026 neutral": MATCH_CONTEXT_WORLD_CUP_NEUTRAL,
    "Localía normal": MATCH_CONTEXT_NORMAL_HOME,
    "País anfitrión": MATCH_CONTEXT_HOST_COUNTRY,
    "Predicción general": MATCH_CONTEXT_GENERAL,
}

section_title("Contexto del partido")

section_title("Partidos oficiales disponibles")
if fixture_validation_errors:
    st.error("Errores en el fixture cargado:\n" + "\n".join(f"- {error}" for error in fixture_validation_errors))
fixture_filter_options = ["Todos"]
if not fixtures_df.empty:
    fixture_filter_options += sorted(set(fixtures_df["team_home"].dropna()) | set(fixtures_df["team_away"].dropna()))
selected_fixture_team = st.selectbox("Filtrar partidos por equipo", fixture_filter_options, key="fixture_team_filter")
visible_fixtures = fixtures_df.copy()
if selected_fixture_team != "Todos" and not visible_fixtures.empty:
    visible_fixtures = visible_fixtures[
        (visible_fixtures["team_home"] == selected_fixture_team) |
        (visible_fixtures["team_away"] == selected_fixture_team)
    ]
st.dataframe(visible_fixtures, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    team_home = st.selectbox(
        "Equipo local",
        team_list,
        index=team_list.index("Argentina") if "Argentina" in team_list else 0,
        key="team_home_select",
        on_change=complete_context_from_selected_teams,
    )
with col2:
    default_away = team_list.index("Austria") if "Austria" in team_list else 1
    team_away = st.selectbox(
        "Equipo visitante",
        team_list,
        index=default_away,
        key="team_away_select",
        on_change=complete_context_from_selected_teams,
    )

if "auto_context" not in st.session_state:
    apply_auto_context_to_session(get_auto_context(team_home, team_away))

if "pending_auto_context" in st.session_state:
    apply_auto_context_to_session(st.session_state.pop("pending_auto_context"))

st.session_state.setdefault("match_context_select", st.session_state.get("auto_context", {}).get("venue_type", "Mundial 2026 neutral"))
st.session_state.setdefault("city_select", st.session_state.get("auto_context", {}).get("city", "Sin especificar"))
st.session_state.setdefault("weather_select", st.session_state.get("auto_context", {}).get("weather", "Normal"))
st.session_state.setdefault("rest_select", st.session_state.get("auto_context", {}).get("rest", "Ambos con descanso normal"))
st.session_state.setdefault("importance_select", st.session_state.get("auto_context", {}).get("importance", "Fase de grupos"))
st.session_state.setdefault("fatigue_select", st.session_state.get("auto_context", {}).get("fatigue", "Normal"))

use_custom_context = st.checkbox("Usar contexto personalizado", value=False, key="use_custom_context")
match_context = st.selectbox("Tipo de sede", VENUE_TYPE_OPTIONS, key="match_context_select", disabled=not use_custom_context)
host_team = None
if match_context_map[match_context] == MATCH_CONTEXT_HOST_COUNTRY:
    host_options = sorted(HOST_TEAMS)
    if "host_team_select" not in st.session_state:
        st.session_state["host_team_select"] = host_options[0]
    host_team = st.selectbox("Anfitrión", host_options, key="host_team_select", disabled=not use_custom_context)

city = st.selectbox("Ciudad / sede", CITY_OPTIONS, key="city_select", disabled=not use_custom_context)
weather = st.selectbox("Clima", WEATHER_OPTIONS, key="weather_select", disabled=not use_custom_context)
rest = st.selectbox("Descanso", REST_OPTIONS, key="rest_select", disabled=not use_custom_context)
importance = st.selectbox("Importancia del partido", IMPORTANCE_OPTIONS, key="importance_select", disabled=not use_custom_context)
fatigue = st.selectbox("Fatiga / viaje", FATIGUE_OPTIONS, key="fatigue_select", disabled=not use_custom_context)

auto_submitted = st.button("Completar contexto automáticamente")
submitted = st.button("Predecir partido")

if auto_submitted:
    auto_context = get_auto_context(team_home, team_away)
    st.session_state["pending_auto_context"] = auto_context
    if not auto_context.get("fixture_found"):
        show_not_found_details(auto_context)
    st.rerun()

detected_context = st.session_state.get("auto_context", {})
if detected_context:
    section_title("Contexto detectado")
    if not detected_context.get("fixture_found"):
        show_not_found_details(detected_context)
    detected_df = pd.DataFrame(
        {
                "Valor": [
                detected_context.get("fixture_found", "N/A"),
                "Sí" if detected_context.get("official_world_cup_match") else "No",
                detected_context.get("date", "N/A"),
                detected_context.get("venue", "N/A"),
                detected_context.get("city", "N/A"),
                detected_context.get("country", "N/A"),
                detected_context.get("matchday", "N/A"),
                detected_context.get("neutral_site", "N/A"),
                detected_context.get("host_advantage_team", "N/A"),
                detected_context.get("weather", "N/A"),
                detected_context.get("weather_source_label", "N/A"),
                detected_context.get("altitude_meters", "N/A"),
                detected_context.get("temperature_c", "N/A"),
                detected_context.get("humidity_pct", "N/A"),
                detected_context.get("wind_kmh", "N/A"),
                detected_context.get("rest_days_home", "N/A"),
                detected_context.get("rest_days_away", "N/A"),
                detected_context.get("home_squad_value_eur_m", "N/A"),
                detected_context.get("away_squad_value_eur_m", "N/A"),
                detected_context.get("home_average_age", "N/A"),
                detected_context.get("away_average_age", "N/A"),
                detected_context.get("phase", "N/A"),
                detected_context.get("data_source", detected_context.get("fixture_source", "N/A")),
            ]
        },
        index=[
            "fixture_found",
            "Partido oficial del Mundial",
            "Fecha",
            "Sede",
            "Ciudad",
            "País",
            "Fecha de grupo",
            "Neutral",
            "Equipo anfitrión",
            "clima_detectado",
            "fuente_clima",
            "Altitud",
            "Temperatura",
            "Humedad",
            "Viento",
            "Descanso local",
            "Descanso visitante",
            "Valor plantilla local (€m)",
            "Valor plantilla visitante (€m)",
            "Edad promedio local",
            "Edad promedio visitante",
            "Fase",
            "fuente_datos",
        ],
    )
    st.table(detected_df)

with st.expander("Historial de predicciones", expanded=False):
    history_df = load_prediction_history()
    history_col1, history_col2 = st.columns([3, 1])
    with history_col1:
        if history_df.empty:
            st.info("Todavía no hay predicciones guardadas.")
        else:
            st.dataframe(history_df.tail(50), use_container_width=True)
    with history_col2:
        if st.button("Limpiar historial", key="clear_prediction_history"):
            if PREDICTION_HISTORY_PATH.exists():
                PREDICTION_HISTORY_PATH.unlink()
            st.success("Historial limpiado.")
            st.rerun()

with st.expander("Simulador de grupo completo", expanded=False):
    group_options = sorted(fixtures_df["group"].dropna().unique()) if not fixtures_df.empty else []
    if not group_options:
        st.warning("No hay grupos cargados para simular.")
    else:
        selected_group = st.selectbox("Grupo", group_options, key="group_simulator_select")
        if st.button("Simular grupo completo", key="simulate_group_button"):
            with st.spinner("Simulando grupo..."):
                base_context = dict(st.session_state.get("auto_context", {}))
                st.session_state["last_group_simulation"] = simulate_group(selected_group, base_context)
                st.session_state["last_group_simulation_group"] = selected_group
        if st.session_state.get("last_group_simulation") is not None:
            st.markdown(f"**Tabla esperada: Grupo {st.session_state.get('last_group_simulation_group')}**")
            st.dataframe(st.session_state["last_group_simulation"], use_container_width=True)

if submitted:
    if team_home == team_away:
        st.warning("Selecciona dos equipos distintos para predecir el partido.")
    else:
        with st.spinner("Generando predicción..."):
            external_context = dict(st.session_state.get("auto_context", {}))
            if use_custom_context:
                external_context.update({
                    "venue_type": match_context,
                    "city": city,
                    "weather": weather,
                    "rest": rest,
                    "importance": importance,
                    "fatigue": fatigue,
                })
            elif not external_context:
                external_context = get_auto_context(team_home, team_away)

            if not external_context.get("fixture_found"):
                show_not_found_details(external_context)

            effective_match_context = external_context.get("match_context", match_context_map[match_context])
            if use_custom_context:
                effective_match_context = match_context_map[match_context]
            effective_host_team = external_context.get("host_team") if not use_custom_context else host_team
            prediction = predict_match(
                team_home,
                team_away,
                match_context=effective_match_context,
                host_team=effective_host_team,
                external_context=external_context,
            )
            exact_scores = predict_exact_scores(
                team_home,
                team_away,
                max_goals=5,
                match_context=effective_match_context,
                host_team=effective_host_team,
                external_context=external_context,
            )
            append_prediction_history(prediction, exact_scores, external_context)
            st.session_state["last_prediction"] = prediction
            st.session_state["last_exact_scores"] = exact_scores
            st.session_state["last_external_context"] = external_context
            st.session_state["show_prediction_explanation"] = False

        st.subheader(f"Predicción: {team_home} vs {team_away}")
        st.info(f"Partido oficial del Mundial: {'Sí' if external_context.get('official_world_cup_match') else 'No'}")
        render_result_cards(prediction, exact_scores)
        section_title("Probabilidad 1X2")
        render_probability_bar(prediction)
        render_prediction_actions(prediction, exact_scores, external_context)
        render_prediction_insights(prediction, exact_scores, external_context)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probabilidad victoria local", f"{prediction['probabilidad_victoria_local'] * 100:.1f}%")
            st.metric("Probabilidad empate", f"{prediction['probabilidad_empate'] * 100:.1f}%")
            st.metric("Probabilidad victoria visitante", f"{prediction['probabilidad_victoria_visitante'] * 100:.1f}%")
        with col2:
            st.markdown(f"**Resultado más probable:** {prediction['resultado_mas_probable']}")
            st.markdown(f"**Confianza del modelo:** {prediction['confianza_modelo'] * 100:.1f}%")
            st.markdown(
                f"**Marcador esperado:** {prediction['marcador_esperado']['home']:.2f} - {prediction['marcador_esperado']['away']:.2f}"
            )

        base_probabilities = prediction.get("base_probabilities", {})
        adjusted_probabilities = prediction.get("adjusted_probabilities", {})
        base_expected_goals = prediction.get("base_expected_goals", {})
        adjusted_expected_goals = prediction.get("marcador_esperado", {})
        comparison_df = pd.DataFrame(
            {
                "Base": [
                    base_probabilities.get("home", 0) * 100,
                    base_probabilities.get("draw", 0) * 100,
                    base_probabilities.get("away", 0) * 100,
                    base_expected_goals.get("home", 0),
                    base_expected_goals.get("away", 0),
                ],
                "Ajustado": [
                    adjusted_probabilities.get("home", 0) * 100,
                    adjusted_probabilities.get("draw", 0) * 100,
                    adjusted_probabilities.get("away", 0) * 100,
                    adjusted_expected_goals.get("home", 0),
                    adjusted_expected_goals.get("away", 0),
                ],
            },
            index=[
                "Prob. local (%)",
                "Prob. empate (%)",
                "Prob. visitante (%)",
                "xG local",
                "xG visitante",
            ],
        )
        section_title("Base vs contexto")
        st.dataframe(comparison_df)

        prob_df = pd.DataFrame(
            {
                "Situación": ["Victoria local", "Empate", "Victoria visitante"],
                "Probabilidad": [
                    prediction["probabilidad_victoria_local"],
                    prediction["probabilidad_empate"],
                    prediction["probabilidad_victoria_visitante"],
                ],
            }
        )
        prob_df["Probabilidad %"] = prob_df["Probabilidad"] * 100

        section_title("Probabilidades 1X2")
        st.dataframe(prob_df[["Situación", "Probabilidad %"]].rename(columns={"Situación": "Resultado", "Probabilidad %": "Probabilidad (%)"}))
        st.bar_chart(prob_df.set_index("Situación")["Probabilidad"])

        section_title("Top 10 marcadores exactos")
        st.dataframe(exact_scores_table(exact_scores, limit=10), use_container_width=True)

        section_title("Marcadores exactos 0-0 a 5-5")
        full_df = pd.DataFrame(exact_scores)
        full_df["Probabilidad %"] = full_df["probabilidad"] * 100
        full_df = full_df[["marcador", "Probabilidad %", "resultado"]]
        full_df.columns = ["Marcador", "Probabilidad (%)", "Resultado"]
        full_df = full_df.sort_values("Probabilidad (%)", ascending=False).reset_index(drop=True)
        st.dataframe(full_df.style.format({"Probabilidad (%)": "{:.2f}"}), use_container_width=True)

        section_title("Diagnóstico del partido")
        diagnostic = prediction.get("diagnostic", {})
        diag_df = pd.DataFrame(
            {
                "Valor": [
                    diagnostic.get("home_team", team_home),
                    diagnostic.get("away_team", team_away),
                    diagnostic.get("match_context", "N/A"),
                    diagnostic.get("neutral_site", "N/A"),
                    diagnostic.get("home_advantage_applied", "N/A"),
                    diagnostic.get("host_advantage_team", "N/A"),
                    diagnostic.get("home_advantage_value", "N/A"),
                    diagnostic.get("context_city", "N/A"),
                    diagnostic.get("context_weather", "N/A"),
                    diagnostic.get("context_rest", "N/A"),
                    diagnostic.get("context_importance", "N/A"),
                    diagnostic.get("context_fatigue", "N/A"),
                    diagnostic.get("home_elo", "N/A"),
                    diagnostic.get("away_elo", "N/A"),
                    diagnostic.get("elo_diff", "N/A"),
                    diagnostic.get("home_rank", "N/A"),
                    diagnostic.get("away_rank", "N/A"),
                    diagnostic.get("rank_diff", "N/A"),
                    diagnostic.get("home_squad_value_eur_m", "N/A"),
                    diagnostic.get("away_squad_value_eur_m", "N/A"),
                    diagnostic.get("squad_value_diff_eur_m", "N/A"),
                    diagnostic.get("home_average_age", "N/A"),
                    diagnostic.get("away_average_age", "N/A"),
                    diagnostic.get("average_age_diff", "N/A"),
                    diagnostic.get("team_metrics_quality_home", "N/A"),
                    diagnostic.get("team_metrics_updated_at", "N/A"),
                    diagnostic.get("base_expected_goals_home", "N/A"),
                    diagnostic.get("base_expected_goals_away", "N/A"),
                    diagnostic.get("expected_goals_home", "N/A"),
                    diagnostic.get("expected_goals_away", "N/A"),
                    diagnostic.get("expected_goals_diff", "N/A"),
                    
                    diagnostic.get("home_win_rate", "N/A"),
                    diagnostic.get("away_win_rate", "N/A"),
                    diagnostic.get("home_goals_avg", "N/A"),
                    diagnostic.get("away_goals_avg", "N/A"),
                    diagnostic.get("home_conceded_avg", "N/A"),
                    diagnostic.get("away_conceded_avg", "N/A"),
                    diagnostic.get("form10_diff", "N/A"),
                    diagnostic.get("home_match_count", "N/A"),
                    diagnostic.get("away_match_count", "N/A"),
                ]
            },
            index=[
                "Equipo local",
                "Equipo visitante",
                "Contexto",
                "Sede neutral",
                "Ventaja aplicada",
                "Equipo con ventaja",
                "Valor ventaja",
                "Ciudad / sede",
                "Clima",
                "Descanso",
                "Importancia",
                "Fatiga / viaje",
                "ELO local",
                "ELO visitante",
                "Diferencia ELO",
                "Ranking local",
                "Ranking visitante",
                "Diferencia ranking",
                "Valor plantilla local (€m)",
                "Valor plantilla visitante (€m)",
                "Diferencia valor (€m)",
                "Edad promedio local",
                "Edad promedio visitante",
                "Diferencia edad",
                "Calidad métricas",
                "Métricas actualizadas",
                "xG base local",
                "xG base visitante",
                "xG local",
                "xG visitante",
                "Diferencia xG",
                "Prob. victoria local",
                "Prob. victoria visitante",
                "Goles avg local",
                "Goles avg visitante",
                "Concedidos avg local",
                "Concedidos avg visitante",
                "Diferencia form10",
                "Partidos local",
                "Partidos visitante",
            ],
        )
        st.table(diag_df)

        warnings = diagnostic.get("warnings", [])
        if warnings:
            st.warning("; ".join(warnings))

        context_adjustments = diagnostic.get("context_adjustments", [])
        if context_adjustments:
            st.markdown("**Explicación de factores aplicados:**")
            for adjustment in context_adjustments:
                st.write(f"- {adjustment}")

        section_title("Explicación de la predicción")
        st.write(prediction.get("explanation", ""))

        st.success("Predicción generada correctamente.")

elif st.session_state.get("last_prediction"):
    section_title("Última predicción")
    render_result_cards(
        st.session_state["last_prediction"],
        st.session_state.get("last_exact_scores", []),
    )
    render_probability_bar(st.session_state["last_prediction"])
    render_prediction_actions(
        st.session_state["last_prediction"],
        st.session_state.get("last_exact_scores", []),
        st.session_state.get("last_external_context", {}),
    )
    render_prediction_insights(
        st.session_state["last_prediction"],
        st.session_state.get("last_exact_scores", []),
        st.session_state.get("last_external_context", {}),
    )

st.markdown(
    '<div class="wc-footer">Proyecto de predicción Mundial 2026 — Python + Machine Learning</div>',
    unsafe_allow_html=True,
)
