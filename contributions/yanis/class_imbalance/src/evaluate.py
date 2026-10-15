"""Calcul des métriques communes au projet (contribution de Yanis).

Ce module est partagé avec la partie 4 : tous les modèles doivent être
évalués avec cette même fonction pour que les résultats soient comparables.
"""
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_proba, threshold=0.5):
    """Renvoie toutes les métriques convenues par le groupe.

    y_true    : vraies classes (0 = <=50K, 1 = >50K)
    y_proba   : probabilité prédite que la classe soit 1 (>50K)
    threshold : seuil au-dessus duquel on déclare >50K (0.5 par défaut)
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "average_precision": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


if __name__ == "__main__":
    import numpy as np

    # L'exemple des 5 personnes A, B, C, D, E
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.7, 0.3])

    for name, value in evaluate_model(y_true, y_proba).items():
        print(f"{name:>18} : {value}")