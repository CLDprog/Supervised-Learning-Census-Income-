"""Exigence 3 : comparaison des remèdes au déséquilibre des classes, sur la validation.

Lancement : python -m src.desequilibre
"""

import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineImblearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, brier_score_loss

from src.config import RANDOM_STATE, RAPPEL_CIBLE
from src.data import charger_jeux
from src.evaluation import evaluer, seuil_pour_rappel
from src.features import construire_preprocessing
from src.models import construire_modele
from src.sorties import sauvegarder_figure, sauvegarder_tableau


def construire_modeles():
    """Régression logistique de référence, avec pondération des classes, et avec SMOTE."""
    return {
        "Référence": construire_modele(LogisticRegression(max_iter=1000)),
        "Pondération des classes": construire_modele(LogisticRegression(max_iter=1000, class_weight="balanced")),
        # SMOTE n'est appliqué qu'au fit (train) ; le Pipeline d'imbalanced-learn refuse un Pipeline comme étape
        "SMOTE": PipelineImblearn([
            *construire_preprocessing().steps,
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("modele", LogisticRegression(max_iter=1000)),
        ]),
    }


def main():
    jeux = charger_jeux()
    lignes, probas = [], {}
    for nom, modele in construire_modeles().items():
        modele.fit(jeux.X_train, jeux.y_train)
        proba = modele.predict_proba(jeux.X_val)[:, 1]
        probas[nom] = proba
        # Chaque remède est évalué au seuil de 0,5, puis au seuil qui donne 80 % de rappel (comparaison équitable)
        for libelle, seuil in [(nom, 0.5), (f"{nom}, seuil ajusté", seuil_pour_rappel(jeux.y_val, proba))]:
            resultat = evaluer(libelle, jeux.y_val, proba, seuil)
            resultat["probabilité moyenne"] = proba.mean()
            resultat["Brier"] = brier_score_loss(jeux.y_val, proba)
            lignes.append(resultat)
    sauvegarder_tableau(pd.DataFrame(lignes).set_index("modèle"), "desequilibre_comparaison")
    print(f"\nVraie part de >50K dans la validation : {jeux.y_val.mean():.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for nom, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(jeux.y_val, proba, ax=ax, name=nom)
    ax.axvline(RAPPEL_CIBLE, linestyle=":", color="black", label="Rappel visé (80 %)")
    ax.axhline(jeux.y_val.mean(), linestyle="--", color="grey", label=f"Hasard ({jeux.y_val.mean():.3f})")
    ax.set(title="Courbes précision-rappel des trois remèdes (validation)", xlabel="Rappel", ylabel="Précision")
    ax.legend()
    sauvegarder_figure("desequilibre_courbes_pr", fig)


if __name__ == "__main__":
    main()
