"""Partie 8 : explicabilité du modèle final avec SHAP.

- Global : importance moyenne de chaque variable (en valeur absolue), d'abord par
  colonne encodée, puis regroupée par variable d'origine (les colonnes one-hot
  d'une même variable sont additionnées pour chaque personne).
- Local : explication détaillée de 4 personnes de la validation (un vrai positif,
  un faux négatif, un faux positif, un vrai négatif).

Les valeurs SHAP sont en log-odds : une valeur positive pousse vers >50K.
Calcul sur un échantillon de la validation (le test n'est pas utilisé).

Lancement : python -m census_income.interpret  (après census_income.train_final)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from census_income.compare_models import load_data
from census_income.config import CONFIG, REPORTS_DIR, SEED
from census_income.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from census_income.train_final import load_final_model

OUTPUT_DIR = REPORTS_DIR / "explainability"

# Colonne encodée -> variable d'origine, pour les features créées en partie 2
DERIVED_TO_ORIGINAL = {
    "capital_gain_log": "capital-gain",
    "capital_gain_capped": "capital-gain",
    "capital_loss_log": "capital-loss",
}


CASE_LABELS = {"vrai_positif": "Vrai positif", "faux_negatif": "Faux négatif",
               "faux_positif": "Faux positif", "vrai_negatif": "Vrai négatif"}


def original_variable(encoded_name):
    """Retrouve la variable d'origine d'une colonne encodée (ex. 'workclass_Private' -> 'workclass')."""
    if encoded_name in DERIVED_TO_ORIGINAL:
        return DERIVED_TO_ORIGINAL[encoded_name]
    if encoded_name in NUMERIC_FEATURES:
        return encoded_name
    for variable in CATEGORICAL_FEATURES:
        if encoded_name.startswith(variable + "_"):
            return variable
    return encoded_name


def aggregate_by_variable(shap_values, encoded_names):
    """Importance par variable d'origine : somme signée des colonnes, puis moyenne absolue."""
    frame = pd.DataFrame(shap_values, columns=encoded_names)
    grouped = frame.T.groupby([original_variable(n) for n in encoded_names]).sum().T
    return grouped.abs().mean().sort_values(ascending=False)


def readable_values(encoder, X_encoded):
    """Valeurs lisibles pour les graphiques : les numériques sont remises à leur échelle d'origine."""
    readable = X_encoded.copy()
    scaler = encoder.named_transformers_["numeric"]
    numeric_cols = [list(encoder.get_feature_names_out()).index(c) for c in scaler.feature_names_in_]
    readable[:, numeric_cols] = scaler.inverse_transform(X_encoded[:, numeric_cols])
    return np.round(readable, 2)


def pick_examples(y_true, proba, threshold):
    """Choisit 4 personnes représentatives des 4 cases de la matrice de confusion."""
    y_true, pred = np.asarray(y_true), (proba >= threshold).astype(int)
    cases = {
        "vrai_positif": (y_true == 1) & (pred == 1),
        "faux_negatif": (y_true == 1) & (pred == 0),
        "faux_positif": (y_true == 0) & (pred == 1),
        "vrai_negatif": (y_true == 0) & (pred == 0),
    }
    picks = {}
    for name, mask in cases.items():
        candidates = np.flatnonzero(mask)
        # la personne la plus « typique » de sa case : probabilité la plus proche de la médiane de la case
        median = np.median(proba[candidates])
        picks[name] = candidates[np.argmin(np.abs(proba[candidates] - median))]
    return picks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = load_final_model()
    pipeline, threshold = artifact["pipeline"], artifact["threshold"]
    encoder, model = pipeline.named_steps["encode"], pipeline.named_steps["model"]

    _, _, X_val, y_val, _ = load_data()
    sample = X_val.sample(n=min(CONFIG["shap"]["sample_size"], len(X_val)), random_state=SEED)
    y_sample = y_val.loc[sample.index]
    X_encoded = pipeline[:-1].transform(sample)
    encoded_names = list(encoder.get_feature_names_out())
    proba = pipeline.predict_proba(sample)[:, 1]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(pd.DataFrame(X_encoded, columns=encoded_names))
    explanation.display_data = readable_values(encoder, X_encoded)

    # --- Global : par colonne encodée ---
    by_column = pd.Series(np.abs(explanation.values).mean(axis=0), index=encoded_names)
    by_column.sort_values(ascending=False).to_csv(OUTPUT_DIR / "importance_par_colonne.csv",
                                                  header=["mean_abs_shap"])
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP : effet des 15 colonnes les plus importantes (validation)")
    plt.savefig(OUTPUT_DIR / "01_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # --- Global : par variable d'origine ---
    by_variable = aggregate_by_variable(explanation.values, encoded_names)
    by_variable.to_csv(OUTPUT_DIR / "importance_par_variable.csv", header=["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(8, 5))
    by_variable.sort_values().plot.barh(ax=ax, color="#536878")
    ax.set(title="Importance globale par variable d'origine (moyenne |SHAP|, log-odds)",
           xlabel="Moyenne de |valeur SHAP|")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_importance_par_variable.png", dpi=150)
    plt.close(fig)

    # --- Local : 4 personnes ---
    picks = pick_examples(y_sample, proba, threshold)
    rows = []
    for i, (case, position) in enumerate(picks.items(), start=3):
        plt.figure()
        shap.plots.waterfall(explanation[position], max_display=12, show=False)
        plt.title(f"{CASE_LABELS[case]} : probabilité {proba[position]:.2f} (seuil {threshold:.2f})")
        plt.savefig(OUTPUT_DIR / f"{i:02d}_local_{case}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        rows.append({"cas": case, "row_id": sample.index[position], "vraie_classe": int(y_sample.iloc[position]),
                     "probabilite": round(float(proba[position]), 4), **sample.iloc[position].to_dict()})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "personnes_expliquees.csv", index=False)

    print("Importance globale par variable d'origine (moyenne |SHAP|) :")
    print(by_variable.round(4).to_string())
    print("\nPersonnes expliquées :")
    print(pd.DataFrame(rows)[["cas", "vraie_classe", "probabilite", "age", "marital-status",
                              "education-num", "capital-gain", "hours-per-week"]].to_string(index=False))
    print(f"\nGraphiques et tableaux dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
