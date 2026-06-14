import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.base import clone
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
import joblib


FORBIDDEN_FEATURES = [
    "goals_home",
    "goals_away",
    "result",
    "winner",
    "match_outcome",
]


def _optional_model(name: str):
    try:
        if name == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                use_label_encoder=False,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                num_class=3,
            )
        if name == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMClassifier(objective="multiclass", random_state=42, num_class=3)
        if name == "catboost":
            import catboost as cb
            return cb.CatBoostClassifier(verbose=0, random_state=42)
    except ImportError:
        print(f"[WARN] Biblioteca opcional {name} no instalada; se omitirá este modelo.")
        return None
    return None


class ModelTrainer:
    """Entrena y guarda modelos de clasificación para predicción de resultados."""

    def __init__(self, config: Dict[str, object]) -> None:
        self.config = config
        self.model_dir = config["paths"]["models"]
        os.makedirs(self.model_dir, exist_ok=True)
        self.random_state = config["model"]["random_state"]

    def _prepare_data(self, data: pd.DataFrame) -> Dict[str, object]:
        target = self.config["model"]["target_column"]

        # Recompute/validate target from goals if present to ensure mapping 0=draw,1=home,2=away
        if {"goals_home", "goals_away"}.issubset(set(data.columns)):
            recomputed = []
            for h, a in zip(data["goals_home"].fillna(0), data["goals_away"].fillna(0)):
                if h > a:
                    recomputed.append(1)
                elif h < a:
                    recomputed.append(2)
                else:
                    recomputed.append(0)
            recomputed = pd.Series(recomputed, index=data.index)
            if target in data.columns:
                mismatches = (data[target].fillna(-1).astype(int) != recomputed.astype(int)).sum()
                if mismatches > 0:
                    print(f"[WARN] Se encontraron {mismatches} discrepancias entre '{target}' y goles. Sobrescribiendo '{target}' con recomputado.")
            data[target] = recomputed

        if target not in data.columns:
            print(f"[ERROR] La columna objetivo '{target}' no existe en los datos.")
            return {}

        # Build feature set by excluding forbidden and metadata columns
        forbidden = set(FORBIDDEN_FEATURES)
        exclude_prefixes = ["team_", "date", "match_id"]

        candidate_cols = [c for c in data.columns if c not in forbidden]
        feature_cols = []
        for c in candidate_cols:
            low = c.lower()
            if any(low.startswith(p) for p in exclude_prefixes):
                continue
            if c == target:
                continue
            if c in ["goals_home", "goals_away", "home_goals", "away_goals", "score"]:
                continue
            feature_cols.append(c)

        # Narrow to numeric features only
        numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(data[c])]
        X = data[numeric_cols].fillna(0)
        y = data[target].astype(int)

        if X.empty or y.empty:
            print("[ERROR] No hay suficientes datos de entrenamiento.")
            return {}

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return {
            "X": X_scaled,
            "y": y,
            "scaler": scaler,
            "feature_cols": list(X.columns),
        }

    def train_all(self, data: pd.DataFrame) -> Dict[str, object]:
        prepared = self._prepare_data(data)
        if not prepared:
            return {}

        X = prepared["X"]
        y = prepared["y"]

        print(f"[INFO] Columnas utilizadas como features: {prepared['feature_cols']}")
        # Save feature list
        try:
            feat_file = os.path.join(self.config["paths"]["reports"], "plots", "feature_list.txt")
            with open(feat_file, "w", encoding="utf-8") as fh:
                fh.write("\n".join(prepared["feature_cols"]))
            print(f"[INFO] Lista de features guardada en {feat_file}")
        except Exception:
            pass

        models = {
            "logistic_regression": LogisticRegression(random_state=self.random_state, max_iter=1000),
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=self.random_state),
        }

        for optional_name in ["xgboost", "lightgbm", "catboost"]:
            model = _optional_model(optional_name)
            if model is not None:
                models[optional_name] = model

        # Create reports dir for plots
        plots_dir = os.path.join(self.config["paths"]["reports"], "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Correlation matrix
        corr = X.join(y.rename("target")).corr()
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="coolwarm")
            corr_path = os.path.join(plots_dir, "correlation_matrix.png")
            plt.title("Correlation matrix (features vs target)")
            plt.tight_layout()
            plt.savefig(corr_path)
            plt.close()
            print(f"[INFO] Matriz de correlación guardada en {corr_path}")
        else:
            print("[INFO] Matplotlib/Seaborn no disponibles; se omite matriz de correlación.")

        # Class distribution
        class_counts = y.value_counts().to_dict()
        print(f"[INFO] Distribución de clases: {class_counts}")

        # Choose n_splits based on smallest class size to avoid invalid split
        min_class_count = int(y.value_counts().min())
        n_splits = min(5, max(2, min_class_count))
        if n_splits < 2:
            print("[ERROR] No hay suficientes muestras por clase para validación estratificada.")
            return {}
        if n_splits != 5:
            print(f"[WARN] Ajustando StratifiedKFold n_splits a {n_splits} por imbalance de clases.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Save class distribution
        try:
            class_file = os.path.join(self.config["paths"]["reports"], "plots", "class_distribution.txt")
            with open(class_file, "w", encoding="utf-8") as fh:
                for k, v in class_counts.items():
                    fh.write(f"{k}: {v}\n")
            print(f"[INFO] Distribución de clases guardada en {class_file}")
        except Exception:
            pass

        trained_models = {}
        for name, model in models.items():
            print(f"[INFO] Entrenando y validando {name} con StratifiedKFold...")
            y_pred = np.zeros(len(y), dtype=int)
            try:
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train = y.iloc[train_idx]
                    clf = clone(model)
                    clf.fit(X_train, y_train)
                    y_pred[test_idx] = clf.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y, y_pred),
                    "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
                }

                # Fit on full data to extract feature importances where available
                try:
                    model.fit(X, y)
                except Exception as exc_full:
                    print(f"[WARN] No se pudo ajustar {name} en todo el conjunto: {exc_full}")

                feature_importances = None
                if hasattr(model, "feature_importances_"):
                    feature_importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    fi = np.abs(model.coef_)
                    if fi.ndim > 1:
                        fi = fi.sum(axis=0)
                    feature_importances = fi

                # Save feature importance plot
                if feature_importances is not None:
                    fi_series = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
                    if PLOTTING_AVAILABLE:
                        plt.figure(figsize=(10, 6))
                        sns.barplot(x=fi_series.values[:30], y=fi_series.index[:30])
                        plt.title(f"Feature importances - {name}")
                        plt.tight_layout()
                        plot_file = os.path.join(plots_dir, f"feature_importance_{name}.png")
                        plt.savefig(plot_file)
                        plt.close()
                        print(f"[INFO] Importancia de variables guardada en {plot_file}")
                    else:
                        print(f"[INFO] Matplotlib/Seaborn no disponibles; se omite plot de importancia para {name}.")

                # Save model
                model_path = os.path.join(self.model_dir, f"{name}.joblib")
                try:
                    joblib.dump(model, model_path)
                except Exception as exc:
                    print(f"[WARN] No se pudo guardar el modelo {name}: {exc}")

                trained_models[name] = {
                    "model": model,
                    "metrics": metrics,
                    "features": prepared["feature_cols"],
                }

                # Print classification report
                print(f"[RESULTS] {name}")
                print(classification_report(y, y_pred, zero_division=0))

            except Exception as exc:
                print(f"[WARN] Error entrenando/validando {name}: {exc}")
                continue

        # Data leakage simple check: look for very high correlation with target
        corr_target = corr["target"].drop("target") if "target" in corr.columns else pd.Series()
        if not corr_target.empty:
            top_corr = corr_target.abs().sort_values(ascending=False)
            if not top_corr.empty and top_corr.iloc[0] > 0.9:
                print(f"[WARN] Posible data leakage: la columna '{top_corr.index[0]}' está altamente correlacionada con el target (|corr|={top_corr.iloc[0]:.3f})")

        return trained_models
