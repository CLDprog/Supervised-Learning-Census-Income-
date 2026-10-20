"""Les trois modèles comparés et leurs espaces de recherche Optuna (exigences 4, 5 et 6)."""

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import construire_preprocessing


def construire_modele(estimateur):
    """Pipeline complet : le même prétraitement pour tous les modèles, suivi de l'estimateur (exigence 5)."""
    return Pipeline([
        ("preprocessing", construire_preprocessing()),
        ("modele", estimateur),
    ])


def estimateurs_par_defaut():
    """Les trois modèles de la progression, avec leurs hyperparamètres par défaut (exigence 4)."""
    return {
        "Régression logistique": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    }


def proposer_estimateur(trial, nom):
    """Crée l'estimateur avec les hyperparamètres proposés par Optuna pour cet essai (exigence 6)."""
    if nom == "Régression logistique":
        return LogisticRegression(C=trial.suggest_float("C", 1e-3, 1e2, log=True), max_iter=2000)
    if nom == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
            max_depth=trial.suggest_int("max_depth", 4, 30),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    if nom == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=RANDOM_STATE, verbose=-1,
        )
    raise ValueError(f"Modèle inconnu : {nom}")
