"""Partie 5 : entraînement du modèle final (Pipeline unique) et choix de son seuil.

Lit les meilleurs réglages trouvés par Optuna (reports/tuning/best_params.json),
entraîne le Pipeline complet (preprocessing + modèle) sur le train, choisit le
seuil de décision qui maximise le F1 sur la validation, puis enregistre le tout
dans models/final_model.joblib. Le test n'est pas utilisé ici.

Lancement : python -m census_income.train_final
"""
import json

import joblib

from census_income.compare_models import load_data, model_pipeline
from census_income.config import BEST_PARAMS_PATH, FINAL_MODEL_PATH, setup_mlflow
from census_income.evaluation import evaluate_model, find_best_threshold
from census_income.tuning import MODEL_NAMES, make_model


def train_final_model(output_path=FINAL_MODEL_PATH, log_to_mlflow=True):
    """Entraîne, choisit le seuil, enregistre et renvoie le modèle final."""
    best = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    model_key = best["selected_model"]
    params = best["models"][model_key]["params"]

    X_train, y_train, X_val, y_val, _ = load_data()
    pipeline = model_pipeline(make_model(model_key, params)).fit(X_train, y_train)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    threshold = find_best_threshold(y_val, val_proba)

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_key": model_key,
        "model_name": MODEL_NAMES[model_key],
        "params": params,
        "features": list(X_train.columns),
        "validation_metrics": evaluate_model(y_val, val_proba, threshold),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    if log_to_mlflow:
        mlflow = setup_mlflow()
        with mlflow.start_run(run_name="modele-final"):
            mlflow.log_params({"model": model_key, "threshold": threshold, **params})
            mlflow.log_metrics({f"validation_{k}": v for k, v in artifact["validation_metrics"].items()})
            mlflow.log_artifact(str(output_path))
    return artifact


def load_final_model(path=FINAL_MODEL_PATH):
    """Charge le modèle final enregistré (à lancer après train_final)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord python -m census_income.train_final")
    return joblib.load(path)


def main():
    artifact = train_final_model()
    print(f"Modèle final : {artifact['model_name']}")
    print(f"Réglages     : {artifact['params']}")
    print(f"Seuil retenu (max F1 sur la validation) : {artifact['threshold']:.4f}")
    print("Métriques sur la validation :")
    for name, value in artifact["validation_metrics"].items():
        print(f"{name:>18} : {value}")
    print(f"Enregistré dans {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
