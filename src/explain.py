"""Exigence 8 : explicabilité SHAP (globale et locale) et diagnostic par segment du modèle final.

Lancement : python -m src.explain   (après python -m src.train)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, RANDOM_STATE
from src.data import charger_jeux
from src.evaluation import charger_modele_final, evaluer_tranches
from src.sorties import sauvegarder_figure, sauvegarder_tableau

TAILLE_ECHANTILLON = 2000


def calculer_shap(modele, X):
    """Valeurs SHAP du modèle final sur X, avec les numériques affichés dans leur unité d'origine."""
    etape_variables = modele.named_steps["preprocessing"].named_steps["nouvelles_variables"]
    etape_colonnes = modele.named_steps["preprocessing"].named_steps["traitement_colonnes"]

    X_transforme = etape_colonnes.transform(etape_variables.transform(X))
    noms = [nom.split("__", 1)[1] for nom in etape_colonnes.get_feature_names_out()]

    X_lisible = pd.DataFrame(X_transforme, columns=noms, index=X.index)
    standardisation = etape_colonnes.named_transformers_["numeriques"].named_steps["mise_a_echelle"]
    X_lisible[COLONNES_NUMERIQUES] = standardisation.inverse_transform(X_lisible[COLONNES_NUMERIQUES])

    explication = shap.TreeExplainer(modele.named_steps["modele"])(X_transforme)
    explication.data = X_lisible.to_numpy()
    explication.feature_names = noms
    return explication


def importance_par_variable(explication):
    """Importance moyenne |SHAP| par variable d'origine (les colonnes one-hot d'une même variable sont additionnées)."""
    noms = explication.feature_names
    variable_origine = {nom: next((c for c in COLONNES_CATEGORIELLES if nom.startswith(c + "_")), nom)
                        for nom in noms}
    contributions = pd.DataFrame(explication.values, columns=noms).T.groupby(variable_origine).sum().T
    return contributions.abs().mean().sort_values(ascending=False)


def main():
    jeux = charger_jeux()
    modele, metadonnees = charger_modele_final()
    seuil = metadonnees["seuil"]

    # 8.1 Importance globale, sur un échantillon de la validation
    echantillon = jeux.X_val.sample(TAILLE_ECHANTILLON, random_state=RANDOM_STATE)
    explication = calculer_shap(modele, echantillon)

    shap.plots.beeswarm(explication, max_display=15, show=False)
    plt.title(f"Importance globale SHAP ({TAILLE_ECHANTILLON} personnes de la validation)")
    sauvegarder_figure("shap_beeswarm")

    importance = importance_par_variable(explication)
    sauvegarder_tableau(importance.to_frame("importance SHAP moyenne"), "shap_importance_par_variable")
    importance.sort_values().plot.barh(figsize=(7, 4.5), title="Importance SHAP moyenne par variable d'origine")
    plt.xlabel("Contribution moyenne absolue (log-odds)")
    sauvegarder_figure("shap_importance_par_variable")

    # 8.2 Explications locales : un vrai positif, un faux négatif et un faux positif
    proba = modele.predict_proba(echantillon)[:, 1]
    y = jeux.y_val.loc[echantillon.index].to_numpy()
    contacte = proba >= seuil
    vrais_positifs = np.where((y == 1) & contacte)[0]
    faux_negatifs = np.where((y == 1) & ~contacte)[0]
    faux_positifs = np.where((y == 0) & contacte)[0]
    cas = {
        "vrai_positif": vrais_positifs[proba[vrais_positifs].argmax()],
        "faux_negatif": faux_negatifs[proba[faux_negatifs].argmin()],
        "faux_positif": faux_positifs[proba[faux_positifs].argmax()],
    }
    colonnes_profil = ["age", "workclass", "education", "marital-status", "occupation",
                       "relationship", "sex", "capital-gain", "hours-per-week"]
    profils = echantillon.iloc[list(cas.values())][colonnes_profil].assign(
        probabilite=proba[list(cas.values())], vraie_classe=y[list(cas.values())]
    )
    profils.index = list(cas)
    sauvegarder_tableau(profils, "shap_cas_locaux")
    for nom, position in cas.items():
        shap.plots.waterfall(explication[position], max_display=10, show=False)
        plt.title(nom.replace("_", " ").capitalize())
        sauvegarder_figure(f"shap_waterfall_{nom}")

    # 8.3 Diagnostic par segment, sur toute la validation
    proba_val = modele.predict_proba(jeux.X_val)[:, 1]
    tranches_age = pd.cut(jeux.X_val["age"], [16, 25, 35, 45, 55, 65, 90])
    for titre, groupes in [("age", tranches_age), ("sexe", jeux.X_val["sex"]), ("race", jeux.X_val["race"])]:
        sauvegarder_tableau(evaluer_tranches(jeux.X_val, jeux.y_val, proba_val, seuil, groupes),
                            f"segments_{titre}")


if __name__ == "__main__":
    main()
