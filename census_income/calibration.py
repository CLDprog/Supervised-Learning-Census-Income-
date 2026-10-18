"""Partie 9 : diagnostic de calibration du modèle final.

Question : quand le modèle annonce « 70 % de chances d'être >50K », est-ce que
70 % de ces personnes le sont vraiment ?

- Courbe de fiabilité (10 tranches de probabilité) sur la validation.
- Brier score (erreur quadratique moyenne des probabilités, plus bas = mieux).
- ECE (écart moyen pondéré entre probabilité annoncée et fréquence observée).
- Écart signé : positif = sur-confiant (annonce trop haut), négatif = sous-confiant.
- Comparaison avec une version recalibrée (isotonique, apprise par validation
  croisée sur le train) pour juger si une recalibration serait utile.

Lancement : python -m census_income.calibration  (après census_income.train_final)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from census_income.compare_models import load_data, make_folds, model_pipeline
from census_income.config import REPORTS_DIR
from census_income.train_final import load_final_model
from census_income.tuning import make_model

OUTPUT_DIR = REPORTS_DIR / "calibration"
N_BINS = 10


def reliability_table(y_true, proba, n_bins=N_BINS):
    """Pour chaque tranche de probabilité : effectif, probabilité moyenne annoncée, fréquence observée."""
    y_true = np.asarray(y_true)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({"tranche": f"[{b / n_bins:.1f} ; {(b + 1) / n_bins:.1f}[", "effectif": int(mask.sum()),
                         "proba_annoncee": proba[mask].mean(), "frequence_observee": y_true[mask].mean()})
    table = pd.DataFrame(rows)
    table["ecart"] = table["proba_annoncee"] - table["frequence_observee"]
    return table


def calibration_errors(y_true, proba, n_bins=N_BINS):
    """ECE (écart absolu pondéré) et écart signé pondéré (> 0 : sur-confiance)."""
    table = reliability_table(y_true, proba, n_bins)
    weights = table["effectif"] / table["effectif"].sum()
    return float((weights * table["ecart"].abs()).sum()), float((weights * table["ecart"]).sum())


def summarize(name, y_true, proba):
    ece, signed = calibration_errors(y_true, proba)
    return {"modele": name, "brier": brier_score_loss(y_true, proba), "ece": ece, "ecart_signe": signed}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_final_model()
    X_train, y_train, X_val, y_val, groups = load_data()
    final_proba = artifact["pipeline"].predict_proba(X_val)[:, 1]

    # Version recalibrée : même modèle, calibration isotonique apprise en validation croisée sur le train
    calibrated = CalibratedClassifierCV(
        model_pipeline(make_model(artifact["model_key"], artifact["params"])),
        method="isotonic", cv=make_folds(X_train, y_train, groups),
    ).fit(X_train, y_train)
    calibrated_proba = calibrated.predict_proba(X_val)[:, 1]

    prevalence = y_train.mean()
    summary = pd.DataFrame([
        summarize(f"{artifact['model_name']} (modèle final)", y_val, final_proba),
        summarize(f"{artifact['model_name']} recalibré (isotonique)", y_val, calibrated_proba),
        summarize("Référence : toujours la proportion de >50K", y_val, np.full(len(y_val), prevalence)),
    ]).set_index("modele")
    table = reliability_table(y_val, final_proba)
    summary.to_csv(OUTPUT_DIR / "resume_calibration.csv")
    table.to_csv(OUTPUT_DIR / "courbe_fiabilite_modele_final.csv", index=False)

    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Calibration parfaite")
    for name, proba, color in [("Modèle final", final_proba, "#536878"),
                               ("Recalibré (isotonique)", calibrated_proba, "#B56953")]:
        t = reliability_table(y_val, proba)
        ax.plot(t["proba_annoncee"], t["frequence_observee"], marker="o", color=color,
                label=f"{name} (ECE={calibration_errors(y_val, proba)[0]:.3f})")
    ax.set(title="Courbe de fiabilité (validation)", ylabel="Fréquence observée de >50K", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax_hist.hist(final_proba, bins=20, range=(0, 1), color="#536878")
    ax_hist.axvline(artifact["threshold"], color="black", linestyle="--", label=f"Seuil {artifact['threshold']:.2f}")
    ax_hist.set(xlabel="Probabilité annoncée de >50K", ylabel="Nombre de personnes")
    ax_hist.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_courbe_fiabilite.png", dpi=150)
    plt.close(fig)

    print("Résumé (validation) :")
    print(summary.round(4).to_string())
    print("\nCourbe de fiabilité du modèle final :")
    print(table.round(3).to_string(index=False))
    ece, signed = summary.iloc[0]["ece"], summary.iloc[0]["ecart_signe"]
    if ece < 0.02:
        verdict = "bien calibré : ni sur- ni sous-confiant de façon notable"
    elif signed > 0:
        verdict = "sur-confiant (annonce des probabilités trop hautes)"
    else:
        verdict = "sous-confiant (annonce des probabilités trop basses)"
    print(f"\nECE {ece:.4f}, écart signé moyen {signed:+.4f} -> modèle final {verdict}")
    print(f"Graphique et tableaux dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
