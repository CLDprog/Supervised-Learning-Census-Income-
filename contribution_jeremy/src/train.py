"""Exigences 4 à 7 : modèles par défaut, réglage Optuna, suivi MLflow et sauvegarde du modèle final.

Lancement : python -m src.train
Test rapide : python -m src.train --essais-max 3
"""

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

from src.config import (BUDGETS_OPTUNA, CHEMIN_METADONNEES, CHEMIN_MODELE_FINAL, DOSSIER_MODELES,
                        MLFLOW_EXPERIENCE, MLFLOW_URI, N_PLIS, RANDOM_STATE)
from src.data import charger_jeux
from src.evaluation import evaluer, seuil_pour_rappel
from src.models import construire_modele, estimateurs_par_defaut, proposer_estimateur
from src.sorties import sauvegarder_figure, sauvegarder_tableau

FICHIERS = {"Régression logistique": "regression_logistique", "Random Forest": "random_forest", "LightGBM": "lightgbm"}


def evaluer_sur_validation(nom, modele, jeux):
    """Entraîne le Pipeline sur le train, puis l'évalue sur la validation au seuil qui donne 80 % de rappel."""
    modele.fit(jeux.X_train, jeux.y_train)
    proba = modele.predict_proba(jeux.X_val)[:, 1]
    return evaluer(nom, jeux.y_val, proba, seuil_pour_rappel(jeux.y_val, proba)), proba


def enregistrer_run(nom_run, parametres, metriques, resultat, proba, y_val, chemin_modele=None):
    """Enregistre une exécution MLflow : paramètres, métriques de validation et artefacts (exigence 7)."""
    with mlflow.start_run(run_name=nom_run):
        mlflow.log_params(parametres)
        mlflow.log_metrics({
            **metriques,
            "pr_auc_val": resultat["PR-AUC"], "roc_auc_val": resultat["ROC-AUC"],
            "precision_val": resultat["précision"], "rappel_val": resultat["rappel"], "f1_val": resultat["F1"],
            "seuil": resultat["seuil"], "contactes_a_tort_val": resultat["contactés à tort"],
        })
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_val, proba, ax=ax, name=nom_run)
        mlflow.log_figure(fig, "courbe_precision_rappel.png")
        plt.close(fig)
        if chemin_modele is not None:
            mlflow.log_artifact(str(chemin_modele))


def regler(nom, n_essais, jeux, validation_croisee):
    """Règle un modèle avec Optuna (PR-AUC en validation croisée sur le train) et enregistre chaque essai."""
    def objectif(trial):
        modele = construire_modele(proposer_estimateur(trial, nom))
        scores = cross_val_score(modele, jeux.X_train, jeux.y_train, cv=validation_croisee,
                                 scoring="average_precision")
        with mlflow.start_run(run_name=f"{nom} - essai {trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"pr_auc_cv": scores.mean(), "pr_auc_cv_ecart_type": scores.std()})
        return scores.mean()

    with mlflow.start_run(run_name=f"Optuna - {nom}"):
        etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
        etude.optimize(objectif, n_trials=n_essais)
        mlflow.log_params(etude.best_params)
        mlflow.log_metric("meilleure_pr_auc_cv", etude.best_value)
    return etude


def main(essais_max=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIENCE)
    DOSSIER_MODELES.mkdir(exist_ok=True)

    jeux = charger_jeux()
    validation_croisee = StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=RANDOM_STATE)

    # Exigence 4 : les trois modèles avec leurs hyperparamètres par défaut
    resultats_defaut = []
    for nom, estimateur in estimateurs_par_defaut().items():
        modele = construire_modele(estimateur)
        scores = cross_validate(modele, jeux.X_train, jeux.y_train, cv=validation_croisee,
                                scoring="average_precision")
        resultat, proba = evaluer_sur_validation(nom, modele, jeux)
        resultat["PR-AUC CV"], resultat["écart-type CV"] = scores["test_score"].mean(), scores["test_score"].std()
        resultat["temps d'entraînement (s)"] = scores["fit_time"].mean()
        resultats_defaut.append(resultat)
        enregistrer_run(f"Défaut - {nom}", estimateur.get_params(),
                        {"pr_auc_cv": resultat["PR-AUC CV"], "pr_auc_cv_ecart_type": resultat["écart-type CV"]},
                        resultat, proba, jeux.y_val)
        print(f"Défaut - {nom} : PR-AUC CV = {resultat['PR-AUC CV']:.3f}, validation = {resultat['PR-AUC']:.3f}")
    sauvegarder_tableau(pd.DataFrame(resultats_defaut).set_index("modèle"), "modeles_defaut")

    # Exigences 6 et 7 : réglage Optuna, chaque essai enregistré dans MLflow
    resultats_regles, modeles_regles = [], {}
    for nom, budget in BUDGETS_OPTUNA.items():
        etude = regler(nom, min(budget, essais_max or budget), jeux, validation_croisee)
        modele = construire_modele(proposer_estimateur(optuna.trial.FixedTrial(etude.best_params), nom))
        resultat, proba = evaluer_sur_validation(f"{nom} (réglé)", modele, jeux)
        resultat["PR-AUC CV"] = etude.best_value
        resultats_regles.append(resultat)
        modeles_regles[nom] = (modele, etude.best_params, resultat)

        chemin = DOSSIER_MODELES / f"{FICHIERS[nom]}_regle.joblib"
        joblib.dump(modele, chemin)
        enregistrer_run(f"Final - {nom}", etude.best_params, {"pr_auc_cv": etude.best_value},
                        resultat, proba, jeux.y_val, chemin)
        print(f"{nom} réglé : meilleure PR-AUC CV = {etude.best_value:.3f}, validation = {resultat['PR-AUC']:.3f}")
        print(f"  meilleurs hyperparamètres : {etude.best_params}")
    sauvegarder_tableau(pd.DataFrame(resultats_regles).set_index("modèle"), "modeles_regles")

    # Modèle final : la meilleure PR-AUC de validation parmi les modèles réglés
    nom_final = max(modeles_regles, key=lambda nom: modeles_regles[nom][2]["PR-AUC"])
    modele_final, parametres, resultat = modeles_regles[nom_final]
    joblib.dump(modele_final, CHEMIN_MODELE_FINAL)
    CHEMIN_METADONNEES.write_text(json.dumps({
        "modele": nom_final,
        "hyperparametres": parametres,
        "seuil": resultat["seuil"],
        "metriques_validation": {cle: resultat[cle] for cle in ["PR-AUC", "ROC-AUC", "précision", "rappel", "F1"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModèle final : {nom_final} (seuil {resultat['seuil']:.3f}), sauvegardé dans {CHEMIN_MODELE_FINAL}")


if __name__ == "__main__":
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("--essais-max", type=int, default=None,
                           help="limite le nombre d'essais Optuna par modèle (pour un test rapide)")
    main(analyseur.parse_args().essais_max)
