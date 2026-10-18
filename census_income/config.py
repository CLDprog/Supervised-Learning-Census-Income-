"""Lecture de config.yaml et chemins communs du projet."""
import os

import yaml

from census_income.data import PROJECT_ROOT

CONFIG = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
SEED = CONFIG["seed"]

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
BEST_PARAMS_PATH = REPORTS_DIR / "tuning" / "best_params.json"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def setup_mlflow():
    """Configure MLflow (base locale SQLite à la racine du projet) et renvoie le module."""
    import mlflow

    uri = CONFIG["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + (PROJECT_ROOT / uri.removeprefix("sqlite:///")).as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(CONFIG["mlflow"]["experiment"])
    return mlflow
