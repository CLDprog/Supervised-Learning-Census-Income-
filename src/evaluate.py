"""Évaluation finale du modèle retenu sur le test (utilisé une seule fois), comparée à la validation.

Lancement : python -m src.evaluate   (après python -m src.train)
"""

import mlflow
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.config import MLFLOW_EXPERIENCE, MLFLOW_URI
from src.data import charger_jeux
from src.evaluation import charger_modele_final, ece, evaluer
from src.sorties import sauvegarder_tableau


def main():
    jeux = charger_jeux()
    modele, metadonnees = charger_modele_final()
    seuil = metadonnees["seuil"]

    resultats = []
    for jeu, X, y in [("validation", jeux.X_val, jeux.y_val), ("test", jeux.X_test, jeux.y_test)]:
        proba = modele.predict_proba(X)[:, 1]
        resultat = evaluer(f"{metadonnees['modele']} ({jeu})", y, proba, seuil)
        resultat["Brier"] = brier_score_loss(y, proba)
        resultat["ECE"] = ece(y, proba)
        resultats.append(resultat)
    sauvegarder_tableau(pd.DataFrame(resultats).set_index("modèle"), "evaluation_finale")

    test = resultats[1]
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIENCE)
    with mlflow.start_run(run_name=f"Test - {metadonnees['modele']}"):
        mlflow.log_params({"modele": metadonnees["modele"], "seuil": seuil})
        mlflow.log_metrics({"pr_auc_test": test["PR-AUC"], "roc_auc_test": test["ROC-AUC"],
                            "precision_test": test["précision"], "rappel_test": test["rappel"],
                            "f1_test": test["F1"], "brier_test": test["Brier"], "ece_test": test["ECE"]})


if __name__ == "__main__":
    main()
