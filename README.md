# World Cup Predictor 2026

Dashboard en Streamlit para predecir partidos del Mundial 2026 usando Machine Learning, ELO, ranking FIFA, modelo Poisson/xG y contexto del partido.

La app permite comparar selecciones, detectar contexto automático del fixture, ajustar factores externos, revisar diagnóstico del modelo, simular grupos y exportar predicciones.

## Características

- Predicción 1X2: victoria local, empate y victoria visitante.
- Marcadores exactos con modelo Poisson/xG.
- Modelo híbrido con Machine Learning, ELO/ranking y ajustes contextuales.
- Contexto automático para partidos del Mundial 2026.
- Soporte para sede neutral, anfitrión, clima, altitud, descanso, fatiga y fase.
- Comparador visual de equipos.
- Simulador de grupo completo.
- Historial local de predicciones en `reports/prediction_history.csv`.
- Exportación de predicciones a CSV.
- Interfaz compatible con tema claro y oscuro de Streamlit.

## Estructura del proyecto

```text
.
├── app.py                         # Punto de entrada de Streamlit
├── predict_match.py               # Lógica principal de predicción
├── auto_context.py                # Detección automática de fixture/contexto
├── context_adjustments.py         # Ajustes por clima, sede, descanso y fase
├── groups_2026.py                 # Grupos del Mundial 2026
├── data/
│   ├── raw/                       # Rankings, métricas, fixture y sedes
│   └── processed/                 # Dataset maestro procesado
├── models/
│   └── best_model.pkl             # Modelo principal usado por la app
├── reports/                       # Reportes generados localmente
├── scripts/                       # Scripts de construcción/entrenamiento
├── src/                           # Módulos auxiliares de data/modeling
├── .streamlit/config.toml         # Configuración visual de Streamlit
├── requirements.txt               # Dependencias de ejecución
└── README.md
```

## Requisitos

- Python 3.10 o superior.
- `pip`.
- Conexión a internet opcional para consultar clima/servicios externos cuando estén disponibles.

No es necesario configurar API keys para usar la predicción general con archivos locales.

## Instalación local

1. Clonar o descargar el proyecto.

2. Crear un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

En macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar la app:

```bash
streamlit run app.py
```

La app se abrirá normalmente en `http://localhost:8501`.

## Archivos de datos requeridos

Para que la app funcione correctamente deben existir estos archivos:

- `models/best_model.pkl`
- `data/raw/world_cup_2026_fixtures.csv`
- `data/raw/world_cup_2026_venues.csv`
- `data/raw/team_metrics_2026.csv`
- `data/raw/fifa_rankings_2026.csv`
- `data/processed/matches_master.csv`

## Variables de entorno y API keys

El proyecto no debe subir archivos `.env`, `key.env` ni secretos al repositorio.

Si en el futuro se usa una API privada, configura los secretos en Streamlit Community Cloud desde:

```text
App settings -> Secrets
```

Ejemplo:

```toml
FOOTBALL_DATA_API_KEY = "tu_api_key"
```

## Despliegue en Streamlit Community Cloud

1. Subir el proyecto a GitHub.
2. Verificar que el repositorio incluya:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `models/best_model.pkl`
   - carpetas `data/`, `src/` y archivos auxiliares.
3. Entrar a [Streamlit Community Cloud](https://streamlit.io/cloud).
4. Crear una nueva app desde el repositorio.
5. Configurar:
   - Main file path: `app.py`
   - Python version: compatible con el proyecto.
6. Agregar secretos solo si se usan APIs privadas.
7. Desplegar.

## Comandos de verificación

```bash
pip install -r requirements.txt
streamlit run app.py
```

Verificación rápida de sintaxis:

```bash
python -m py_compile app.py auto_context.py context_adjustments.py predict_match.py
```

## Notas

- Los archivos generados localmente como historial, logs, cachés, entornos virtuales y `.env` están excluidos por `.gitignore`.
- XGBoost, LightGBM y CatBoost son opcionales para experimentos de entrenamiento, pero no son necesarios para ejecutar la app principal.
- Las predicciones son estimaciones estadísticas y no garantizan resultados reales.
