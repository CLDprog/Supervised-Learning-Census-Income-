"""Métriques du cadrage, règle de seuil, calibration et chargement du modèle final."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from src.config import CHEMIN_METADONNEES, CHEMIN_MODELE_FINAL, RAPPEL_CIBLE


def evaluer(nom, y_vrai, proba, seuil=0.5):
    """Calcule les métriques du cadrage pour des probabilités et un seuil donnés."""
    prediction = (proba >= seuil).astype(int)
    return {
        "modèle": nom,
        "seuil": seuil,
        "PR-AUC": average_precision_score(y_vrai, proba),
        "ROC-AUC": roc_auc_score(y_vrai, proba),
        "précision": precision_score(y_vrai, prediction),
        "rappel": recall_score(y_vrai, prediction),
        "F1": f1_score(y_vrai, prediction),
        "contactés à tort": int(((prediction == 1) & (np.asarray(y_vrai) == 0)).sum()),
    }


def seuil_pour_rappel(y_vrai, proba, rappel_min=RAPPEL_CIBLE):
    """Seuil le plus élevé qui garde un rappel d'au moins rappel_min (remède retenu à l'exigence 3)."""
    _, rappels, seuils = precision_recall_curve(y_vrai, proba)
    return float(seuils[rappels[:-1] >= rappel_min].max())


def ece(y_vrai, proba, n_tranches=10):
    """Expected Calibration Error : écart moyen entre probabilité prédite et part observée, pondéré par tranche."""
    y_vrai, proba = np.asarray(y_vrai), np.asarray(proba)
    tranches = np.minimum((proba * n_tranches).astype(int), n_tranches - 1)
    return float(sum((tranches == t).mean() * abs(y_vrai[tranches == t].mean() - proba[tranches == t].mean())
                     for t in range(n_tranches) if (tranches == t).any()))


def evaluer_tranches(X, y, proba, seuil, groupes):
    """Métriques par groupe (diagnostic par segment, exigence 8)."""
    proba = pd.Series(proba, index=X.index)
    lignes = []
    for groupe, index in X.groupby(groupes, observed=True).groups.items():
        y_groupe, proba_groupe = y.loc[index], proba.loc[index]
        contacte = proba_groupe >= seuil
        lignes.append({
            "groupe": groupe,
            "effectif": len(index),
            "dont >50K": int(y_groupe.sum()),
            "part de >50K": y_groupe.mean(),
            "PR-AUC": average_precision_score(y_groupe, proba_groupe),
            "précision": precision_score(y_groupe, contacte, zero_division=0),
            "rappel": recall_score(y_groupe, contacte),
        })
    return pd.DataFrame(lignes).set_index("groupe")


def charger_modele_final():
    """Recharge le Pipeline final et ses métadonnées (nom, hyperparamètres, seuil) sauvegardés par train.py."""
    if not CHEMIN_MODELE_FINAL.exists():
        raise FileNotFoundError("Modèle final introuvable : lancer d'abord `python -m src.train`.")
    modele = joblib.load(CHEMIN_MODELE_FINAL)
    metadonnees = json.loads(CHEMIN_METADONNEES.read_text(encoding="utf-8"))
    return modele, metadonnees
