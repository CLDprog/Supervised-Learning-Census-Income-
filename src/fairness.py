"""Exigence 10 : audit d'équité selon le sexe, mitigation et comparaison avant / après sur le test.

Lancement : python -m src.fairness   (après python -m src.train)
"""

import json

import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import ATTRIBUT_SENSIBLE, DOSSIER_RESULTATS, RANDOM_STATE
from src.data import charger_jeux
from src.evaluation import charger_modele_final, seuil_pour_rappel
from src.sorties import sauvegarder_tableau

METRIQUES_EQUITE = {
    "part contactée": selection_rate,              # parité démographique
    "rappel": true_positive_rate,                  # égalité des chances (définition retenue)
    "taux de faux positifs": false_positive_rate,  # avec le rappel : égalité des erreurs
    "précision": lambda y, p: precision_score(y, p, zero_division=0),  # parité prédictive
}


def audit_equite(y_vrai, prediction, sexe):
    """Métriques d'équité par sexe, et écart entre les deux groupes."""
    cadre = MetricFrame(metrics=METRIQUES_EQUITE, y_true=y_vrai, y_pred=prediction, sensitive_features=sexe)
    tableau = cadre.by_group.copy()
    tableau.loc["écart (H - F)"] = tableau.loc["Male"] - tableau.loc["Female"]
    return tableau


def main():
    jeux = charger_jeux()
    modele, metadonnees = charger_modele_final()
    seuil = metadonnees["seuil"]
    sexe_val, sexe_test = jeux.X_val[ATTRIBUT_SENSIBLE], jeux.X_test[ATTRIBUT_SENSIBLE]

    # 10.1 Audit sur la validation, avec un seuil unique
    proba_val = modele.predict_proba(jeux.X_val)[:, 1]
    sauvegarder_tableau(audit_equite(jeux.y_val, (proba_val >= seuil).astype(int), sexe_val), "equite_validation")

    # 10.2 Mitigation (post-traitement) : un seuil par sexe, choisi sur la validation pour 80 % de rappel par groupe
    seuils_par_sexe = {
        sexe: seuil_pour_rappel(jeux.y_val[sexe_val == sexe], proba_val[(sexe_val == sexe).to_numpy()])
        for sexe in ["Female", "Male"]
    }
    print(f"\nSeuil unique : {seuil:.3f} ; seuils par sexe : "
          f"{ {sexe: round(valeur, 3) for sexe, valeur in seuils_par_sexe.items()} }")
    DOSSIER_RESULTATS.mkdir(exist_ok=True)
    (DOSSIER_RESULTATS / "seuils_par_sexe.json").write_text(json.dumps(seuils_par_sexe, indent=2), encoding="utf-8")

    # Comparaison : ThresholdOptimizer de Fairlearn, contrainte d'égalité des rappels
    optimiseur = ThresholdOptimizer(estimator=modele, constraints="true_positive_rate_parity",
                                    objective="balanced_accuracy_score", prefit=True,
                                    predict_method="predict_proba")
    optimiseur.fit(jeux.X_val, jeux.y_val, sensitive_features=sexe_val)

    # 10.3 Avant / après, sur le test resté intact
    proba_test = modele.predict_proba(jeux.X_test)[:, 1]
    predictions = {  # nom affiché -> (nom du fichier de résultats, décisions)
        "Avant (seuil unique)": ("avant", (proba_test >= seuil).astype(int)),
        "Après : seuil par sexe": (
            "seuil_par_sexe", (proba_test >= sexe_test.map(seuils_par_sexe).to_numpy(dtype=float)).astype(int)),
        "Après : ThresholdOptimizer": (
            "threshold_optimizer",
            optimiseur.predict(jeux.X_test, sensitive_features=sexe_test, random_state=RANDOM_STATE)),
    }
    lignes = []
    for nom, (fichier, prediction) in predictions.items():
        audit = audit_equite(jeux.y_test, prediction, sexe_test)
        lignes.append({
            "version": nom,
            "précision": precision_score(jeux.y_test, prediction),
            "rappel": recall_score(jeux.y_test, prediction),
            "F1": f1_score(jeux.y_test, prediction),
            "rappel femmes": audit.loc["Female", "rappel"],
            "rappel hommes": audit.loc["Male", "rappel"],
            "écart de rappel": audit.loc["écart (H - F)", "rappel"],
        })
        sauvegarder_tableau(audit, f"equite_test_{fichier}")
    sauvegarder_tableau(pd.DataFrame(lignes).set_index("version"), "equite_avant_apres_test")


if __name__ == "__main__":
    main()
