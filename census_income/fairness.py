"""Partie 10 : évaluation finale sur le test et audit d'équité (attribut sensible : sex).

C'est le SEUL script qui ouvre le test, une seule fois, avec un modèle et un
seuil déjà figés (choisis sur le train et la validation).

1. Performance finale du modèle sur le test (métriques + Brier score).
2. Audit d'équité par sexe : taux de sélection, rappel (TPR), taux de faux
   positifs (FPR), précision, et les écarts correspondant à deux définitions :
   - parité démographique : même taux de prédiction >50K pour les deux groupes ;
   - égalité des chances (equal opportunity) : même rappel pour les deux groupes.
3. Atténuation par post-traitement : un seuil par groupe, choisi sur la
   validation pour que le rappel de chaque groupe soit égal au rappel global
   (méthode « equal opportunity » de Hardt et al., 2016). On compare ensuite
   performance et écarts avant/après sur le test intact.

Lancement : python -m census_income.fairness  (après census_income.train_final)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from census_income.config import REPORTS_DIR
from census_income.data import DATA_DIR, load_partition, model_inputs
from census_income.evaluation import evaluate_model
from census_income.train_final import load_final_model

OUTPUT_DIR = REPORTS_DIR / "fairness"


def group_metrics(y_true, y_pred, sensitive):
    """Métriques par groupe (fairlearn MetricFrame)."""
    frame = MetricFrame(
        metrics={
            "effectif": lambda y, p: len(y),
            "taux_reel_>50K": lambda y, p: np.mean(y),
            "taux_selection": selection_rate,
            "rappel_TPR": true_positive_rate,
            "taux_faux_positifs_FPR": false_positive_rate,
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
        },
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive,
    )
    return frame.by_group


def fairness_gaps(y_true, y_pred, sensitive):
    """Écarts entre groupes pour les deux définitions d'équité (0 = parfaitement équitable)."""
    tpr = MetricFrame(metrics=true_positive_rate, y_true=y_true, y_pred=y_pred,
                      sensitive_features=sensitive).by_group
    return {
        "ecart_parite_demographique": demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive),
        "ecart_egalite_des_chances_TPR": float(tpr.max() - tpr.min()),
        "ecart_equalized_odds": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
    }


def overall_performance(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def equal_opportunity_thresholds(y_true, proba, sensitive, target_recall):
    """Un seuil par groupe, tel que le rappel du groupe soit égal à target_recall."""
    y_true, sensitive = np.asarray(y_true), np.asarray(sensitive)
    thresholds = {}
    for group in np.unique(sensitive):
        positive_proba = proba[(sensitive == group) & (y_true == 1)]
        thresholds[group] = float(np.quantile(positive_proba, 1 - target_recall, method="lower"))
    return thresholds


def predict_with_group_thresholds(proba, sensitive, thresholds):
    sensitive = np.asarray(sensitive)
    cut = np.array([thresholds[g] for g in sensitive])
    return (proba >= cut).astype(int)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_final_model()
    pipeline, threshold = artifact["pipeline"], artifact["threshold"]

    X_val, y_val, sex_val = model_inputs(load_partition(DATA_DIR, "validation"))
    X_test, y_test, sex_test = model_inputs(load_partition(DATA_DIR, "test"))  # unique ouverture du test
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- 1. Performance finale sur le test ---
    test_metrics = evaluate_model(y_test, test_proba, threshold)
    test_metrics["brier"] = brier_score_loss(y_test, test_proba)
    pd.Series(test_metrics).to_csv(OUTPUT_DIR / "performance_finale_test.csv", header=["valeur"])

    # --- 2. Audit avant atténuation ---
    pred_before = (test_proba >= threshold).astype(int)
    groups_before = group_metrics(y_test, pred_before, sex_test)
    gaps_before = fairness_gaps(y_test, pred_before, sex_test)

    # --- 3. Atténuation : seuils par groupe choisis sur la validation ---
    target_recall = recall_score(y_val, (val_proba >= threshold).astype(int))
    group_thresholds = equal_opportunity_thresholds(y_val, val_proba, sex_val, target_recall)
    pred_after = predict_with_group_thresholds(test_proba, sex_test, group_thresholds)
    groups_after = group_metrics(y_test, pred_after, sex_test)
    gaps_after = fairness_gaps(y_test, pred_after, sex_test)

    comparison = pd.DataFrame({
        "avant": {**overall_performance(y_test, pred_before), **gaps_before},
        "apres": {**overall_performance(y_test, pred_after), **gaps_after},
    })
    comparison["difference"] = comparison["apres"] - comparison["avant"]
    comparison.to_csv(OUTPUT_DIR / "comparaison_avant_apres.csv")
    by_group = pd.concat({"avant": groups_before, "apres": groups_after}, names=["moment", "sex"])
    by_group.to_csv(OUTPUT_DIR / "metriques_par_groupe.csv")
    pd.Series(group_thresholds, name="seuil").to_csv(OUTPUT_DIR / "seuils_par_groupe.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ["taux_selection", "rappel_TPR", "taux_faux_positifs_FPR"],
                                 ["Taux de sélection (>50K prédit)", "Rappel (TPR)", "Taux de faux positifs (FPR)"]):
        data = pd.DataFrame({"Avant": groups_before[metric], "Après": groups_after[metric]})
        data.plot.bar(ax=ax, color=["#BCC5CA", "#536878"], rot=0)
        ax.set(title=title, xlabel="", ylim=(0, 1))
    fig.suptitle("Audit d'équité par sexe sur le test : avant / après seuils par groupe")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_equite_avant_apres.png", dpi=150)
    plt.close(fig)

    print(f"=== Performance finale sur le test ({artifact['model_name']}, seuil {threshold:.4f}) ===")
    for name, value in test_metrics.items():
        print(f"{name:>18} : {value}")
    print("\n=== Métriques par sexe, AVANT atténuation ===")
    print(groups_before.round(4).to_string())
    print(f"\nSeuils par groupe (choisis sur la validation, rappel cible {target_recall:.4f}) : "
          + ", ".join(f"{g} = {t:.4f}" for g, t in group_thresholds.items()))
    print("\n=== Métriques par sexe, APRÈS atténuation ===")
    print(groups_after.round(4).to_string())
    print("\n=== Comparaison avant / après (test) ===")
    print(comparison.round(4).to_string())
    print(f"\nGraphique et tableaux dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
