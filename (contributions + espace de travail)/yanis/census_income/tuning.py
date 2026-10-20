"""Parties 6 et 7 : réglage des hyperparamètres avec Optuna, suivi des essais avec MLflow.

Pour chacun des 3 modèles, Optuna cherche les réglages qui maximisent l'AP
moyenne en validation croisée groupée (5 découpages, train uniquement).
Chaque essai est enregistré dans MLflow (réglages, métriques, durée), sous une
exécution « parent » par modèle qui garde aussi les meilleurs réglages et les
fichiers produits. La validation sert ensuite à comparer les modèles réglés :
elle n'a jamais été vue pendant la recherche, son score n'est donc pas optimiste.

Lancement : python -m census_income.tuning
Résultats : reports/tuning/ (dont best_params.json, lu par census_income.train_final)
"""
import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from census_income.compare_models import build_models, load_data, make_folds, model_pipeline
from census_income.config import BEST_PARAMS_PATH, CONFIG, REPORTS_DIR, SEED, setup_mlflow
from census_income.evaluation import evaluate_model

OUTPUT_DIR = REPORTS_DIR / "tuning"

# Nom court -> (nom affiché, nom dans la partie 4)
MODEL_NAMES = {
    "logistic_regression": "Régression logistique",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


# Réglages de la partie 4 (notebook de Hugo), testés en premier par Optuna :
# le modèle réglé ne peut donc pas être moins bon que celui de départ en validation croisée.
STARTING_PARAMS = {
    "logistic_regression": {"C": 1.0},
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
    "gradient_boosting": {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 31,
                          "min_samples_leaf": 20, "l2_regularization": 1e-4, "max_features": 1.0},
}


def suggest_params(model_key, trial):
    """Espace de recherche de chaque modèle."""
    if model_key == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if model_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 30]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        }
    if model_key == "gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    raise ValueError(f"Modèle inconnu : {model_key}")


def make_model(model_key, params):
    """Construit le modèle avec ses réglages (toujours sans pondération des classes)."""
    if model_key == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=SEED, **params)
    if model_key == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    if model_key == "gradient_boosting":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params)
    raise ValueError(f"Modèle inconnu : {model_key}")


def cv_scores(model, X, y, folds):
    """AP, ROC-AUC et F1 (seuil 0,5) sur chaque découpage, et durée totale."""
    scores, start = [], time.perf_counter()
    for fit_idx, score_idx in folds:
        pipeline = model_pipeline(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        proba = pipeline.predict_proba(X.iloc[score_idx])[:, 1]
        scores.append(evaluate_model(y.iloc[score_idx], proba))
    table = pd.DataFrame(scores)
    return table, time.perf_counter() - start


def tune_model(mlflow, model_key, X, y, folds, n_trials):
    """Recherche Optuna pour un modèle ; chaque essai est une exécution MLflow."""
    def objective(trial):
        params = suggest_params(model_key, trial)
        with mlflow.start_run(run_name=f"{model_key}-essai-{trial.number:03d}", nested=True):
            table, seconds = cv_scores(make_model(model_key, params), X, y, folds)
            mean_ap, sd_ap = table["average_precision"].mean(), table["average_precision"].std()
            mlflow.log_params({"model": model_key, "trial": trial.number, **params})
            mlflow.log_metrics({
                "cv_average_precision": mean_ap,
                "cv_average_precision_sd": sd_ap,
                "cv_roc_auc": table["roc_auc"].mean(),
                "cv_f1_seuil_0_5": table["f1"].mean(),
                "duree_secondes": seconds,
            })
        trial.set_user_attr("cv_ap_sd", sd_ap)
        trial.set_user_attr("cv_roc_auc", table["roc_auc"].mean())
        return mean_ap

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=model_key)
    study.enqueue_trial(STARTING_PARAMS[model_key])
    with mlflow.start_run(run_name=f"{model_key}-recherche"):
        mlflow.log_params({"model": model_key, "n_trials": n_trials, "cv": "StratifiedGroupKFold(5)",
                           "objectif": "AP moyenne en validation croisée"})
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
        trials_path = OUTPUT_DIR / f"essais_{model_key}.csv"
        trials.to_csv(trials_path, index=False)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_cv_average_precision", study.best_value)
        mlflow.log_artifact(str(trials_path))
    return study


def export_mlflow_runs(mlflow):
    """Exporte toutes les exécutions MLflow dans un CSV versionné (preuve du suivi des essais)."""
    runs = mlflow.search_runs(search_all_experiments=False)
    keep = [c for c in runs.columns if c.startswith(("params.", "metrics.")) or
            c in {"tags.mlflow.runName", "start_time", "end_time", "status"}]
    runs = runs[keep].rename(columns={"tags.mlflow.runName": "run_name"}).sort_values("start_time")
    path = OUTPUT_DIR / "mlflow_runs.csv"
    runs.to_csv(path, index=False)
    return path, len(runs)


def main():
    parser = argparse.ArgumentParser(description="Réglage Optuna des 3 modèles")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="remplace le nombre d'essais de config.yaml (utile pour un test rapide)")
    parser.add_argument("--export-only", action="store_true",
                        help="exporte seulement les exécutions MLflow existantes dans reports/tuning/mlflow_runs.csv")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow = setup_mlflow()
    if args.export_only:
        path, n = export_mlflow_runs(mlflow)
        print(f"{n} exécutions MLflow exportées dans {path}")
        return

    X_train, y_train, X_val, y_val, groups = load_data()
    folds = make_folds(X_train, y_train, groups)
    defaults = build_models()

    rows, best = [], {}
    for model_key, display_name in MODEL_NAMES.items():
        n_trials = args.n_trials or CONFIG["tuning"]["n_trials"][model_key]
        print(f"\n=== {display_name} : {n_trials} essais Optuna ===")
        default_table, _ = cv_scores(defaults[display_name], X_train, y_train, folds)
        study = tune_model(mlflow, model_key, X_train, y_train, folds, n_trials)

        tuned_model = make_model(model_key, study.best_params)
        pipeline = model_pipeline(tuned_model).fit(X_train, y_train)
        val_ap = average_precision_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        default_pipeline = model_pipeline(defaults[display_name]).fit(X_train, y_train)
        default_val_ap = average_precision_score(y_val, default_pipeline.predict_proba(X_val)[:, 1])

        best[model_key] = {"params": study.best_params, "cv_average_precision": study.best_value,
                           "validation_average_precision": val_ap}
        rows.append({
            "model": display_name,
            "CV_AP_avant_reglage": default_table["average_precision"].mean(),
            "CV_AP_apres_reglage": study.best_value,
            "CV_AP_sd_apres": study.best_trial.user_attrs["cv_ap_sd"],
            "validation_AP_avant": default_val_ap,
            "validation_AP_apres": val_ap,
        })
        print(f"  AP validation croisée : {rows[-1]['CV_AP_avant_reglage']:.4f} -> {study.best_value:.4f}")
        print(f"  AP validation         : {default_val_ap:.4f} -> {val_ap:.4f}")
        print(f"  meilleurs réglages    : {study.best_params}")

    summary = pd.DataFrame(rows).set_index("model")
    summary["gain_validation_AP"] = summary["validation_AP_apres"] - summary["validation_AP_avant"]
    summary.to_csv(OUTPUT_DIR / "resume_reglage.csv")

    selected = max(best, key=lambda k: best[k]["cv_average_precision"])
    BEST_PARAMS_PATH.write_text(json.dumps({"selected_model": selected, "models": best}, indent=2,
                                           default=lambda v: v.item()),
                                encoding="utf-8")
    print("\n=== Résumé ===")
    print(summary.round(4).to_string())
    print(f"\nModèle retenu (meilleure AP en validation croisée) : {MODEL_NAMES[selected]}")
    print(f"Réglages enregistrés dans {BEST_PARAMS_PATH}")
    path, n = export_mlflow_runs(mlflow)
    print(f"{n} exécutions MLflow exportées dans {path}")


if __name__ == "__main__":
    main()
