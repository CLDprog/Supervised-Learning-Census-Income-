"""Exigence 9 : diagnostics de calibration du modèle final (courbe de fiabilité, Brier, ECE, recalibration).

Lancement : python -m src.calibration   (après python -m src.train)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import ATTRIBUT_SENSIBLE, N_PLIS, RANDOM_STATE
from src.data import charger_jeux
from src.evaluation import charger_modele_final, ece
from src.sorties import sauvegarder_figure, sauvegarder_tableau


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    jeux = charger_jeux()
    modele, _ = charger_modele_final()
    proba = modele.predict_proba(jeux.X_val)[:, 1]

    # 9.1 Score de Brier et ECE, comparés à une prédiction constante
    print(f"Score de Brier du modèle final : {brier_score_loss(jeux.y_val, proba):.4f}")
    constante = np.full(len(jeux.y_val), jeux.y_train.mean())
    print(f"Score de Brier d'une prédiction constante : {brier_score_loss(jeux.y_val, constante):.4f}")
    print(f"ECE (10 tranches) : {ece(jeux.y_val, proba):.4f}")

    # 9.2 Courbe de fiabilité et répartition des probabilités prédites
    tranches = pd.cut(proba, np.linspace(0, 1, 11), include_lowest=True)
    detail = (pd.DataFrame({"proba": proba, "cible": jeux.y_val.to_numpy()})
              .groupby(tranches, observed=True)
              .agg(effectif=("cible", "size"), probabilite_moyenne=("proba", "mean"),
                   part_observee=("cible", "mean")))
    sauvegarder_tableau(detail, "calibration_par_tranche")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    part_observee, probabilite_moyenne = calibration_curve(jeux.y_val, proba, n_bins=10)
    ax1.plot([0, 1], [0, 1], "--", color="grey", label="Calibration parfaite")
    ax1.plot(probabilite_moyenne, part_observee, "o-", label="Modèle final")
    ax1.set(ylabel="Part observée de >50K", title="Courbe de fiabilité (validation)")
    ax1.legend()
    ax2.hist(proba, bins=10, range=(0, 1), edgecolor="white")
    ax2.set(xlabel="Probabilité prédite", ylabel="Nombre de personnes")
    sauvegarder_figure("calibration_courbe_fiabilite", fig)

    # 9.3 Recalibration (Platt, isotonique), apprise sur des prédictions hors pli du train
    validation_croisee = StratifiedKFold(n_splits=N_PLIS, shuffle=True, random_state=RANDOM_STATE)
    proba_hors_pli = cross_val_predict(clone(modele), jeux.X_train, jeux.y_train,
                                       cv=validation_croisee, method="predict_proba")[:, 1]
    platt = LogisticRegression().fit(logit(proba_hors_pli).reshape(-1, 1), jeux.y_train)
    isotonique = IsotonicRegression(out_of_bounds="clip").fit(proba_hors_pli, jeux.y_train)
    versions = {
        "Sans recalibration": proba,
        "Platt": platt.predict_proba(logit(proba).reshape(-1, 1))[:, 1],
        "Isotonique": isotonique.predict(proba),
    }
    sauvegarder_tableau(pd.DataFrame({
        "score de Brier": [brier_score_loss(jeux.y_val, p) for p in versions.values()],
        "ECE": [ece(jeux.y_val, p) for p in versions.values()],
        "PR-AUC": [average_precision_score(jeux.y_val, p) for p in versions.values()],
    }, index=list(versions)), "calibration_recalibration")

    # 9.4 Calibration par sexe
    sexe = jeux.X_val[ATTRIBUT_SENSIBLE]
    sauvegarder_tableau(pd.DataFrame([
        {"groupe": groupe,
         "probabilité moyenne prédite": proba[(sexe == groupe).to_numpy()].mean(),
         "part observée de >50K": jeux.y_val[sexe == groupe].mean(),
         "ECE": ece(jeux.y_val[sexe == groupe], proba[(sexe == groupe).to_numpy()])}
        for groupe in ["Female", "Male"]
    ]).set_index("groupe"), "calibration_par_sexe")


if __name__ == "__main__":
    main()
