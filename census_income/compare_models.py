"""Partie 4 : progression de modélisation (3 modèles comparés).

Conversion en script du notebook de Hugo (notebooks/04_model_comparison.ipynb) :
mêmes modèles, mêmes réglages, mêmes découpages de validation croisée.
Le test n'est jamais chargé ici.

Lancement : python -m census_income.compare_models
Résultats : tableaux et graphiques dans reports/model_comparison/
"""
import time

import matplotlib

matplotlib.use("Agg")  # enregistre les graphiques sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from census_income.data import DATA_DIR, PROJECT_ROOT, ROW_ID, TARGET, load_partition, model_inputs
from census_income.evaluation import evaluate_model, find_best_threshold
from census_income.pipeline import build_pipeline

SEED = 42
N_FOLDS = 5
OUTPUT_DIR = PROJECT_ROOT / "reports" / "model_comparison"
DUMMY_NAME = "Modèle naïf"
TABLE_METRICS = ["average_precision", "roc_auc", "precision", "recall", "f1", "accuracy"]
COLORS = ["#536878", "#B56953", "#6F8261"]


def load_data():
    """Charge train et validation, et calcule les groupes de la validation croisée.

    Deux lignes avec exactement le même profil d'origine forment un groupe :
    elles ne sont jamais séparées entre entraînement et évaluation.
    """
    train_df = load_partition(DATA_DIR, "train")
    val_df = load_partition(DATA_DIR, "validation")
    X_train, y_train, _ = model_inputs(train_df)
    X_val, y_val, _ = model_inputs(val_df)

    profile_columns = [c for c in train_df.columns if c not in {ROW_ID, TARGET}]
    groups = pd.util.hash_pandas_object(train_df[profile_columns], index=False).to_numpy()
    return X_train, y_train, X_val, y_val, groups


def build_models():
    """Les 3 modèles comparés, avec les réglages de Hugo (aucune pondération des classes)."""
    return {
        "Régression logistique": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_features="sqrt", min_samples_leaf=2,
            random_state=SEED, n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
            min_samples_leaf=20, early_stopping=False, random_state=SEED,
        ),
    }


def model_pipeline(estimator):
    """La chaîne de la partie 2 suivie du modèle, en sortie dense.

    Le Gradient Boosting n'accepte pas les tableaux creux (sparse) :
    on demande donc un tableau classique pour tous les modèles.
    """
    pipeline = build_pipeline(clone(estimator))
    pipeline.set_params(encode__sparse_threshold=0.0)
    return pipeline


def make_folds(X, y, groups):
    """Les 5 découpages, identiques pour tous les modèles."""
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(X, y, groups))


def cross_validate_models(candidates, X, y, folds):
    """Validation croisée de chaque modèle sur les mêmes découpages (seuil 0,5)."""
    rows = []
    for name, estimator in candidates.items():
        for fold, (fit_idx, score_idx) in enumerate(folds, 1):
            pipeline = model_pipeline(estimator)
            start = time.perf_counter()
            pipeline.fit(X.iloc[fit_idx], y.iloc[fit_idx])
            seconds = time.perf_counter() - start
            proba = pipeline.predict_proba(X.iloc[score_idx])[:, 1]
            rows.append({"model": name, "fold": fold, "fit_seconds": seconds,
                         **evaluate_model(y.iloc[score_idx], proba)})
        print(f"  validation croisée terminée : {name}")
    return pd.DataFrame(rows)


def fit_on_full_train(candidates, X_train, y_train, X_val):
    """Entraîne chaque modèle sur tout le train.

    Renvoie ses probabilités sur la validation, et son AP sur le train lui-même
    (score optimiste, qui sert seulement à repérer le surapprentissage).
    """
    val_proba, train_ap = {}, {}
    for name, estimator in candidates.items():
        pipeline = model_pipeline(estimator).fit(X_train, y_train)
        val_proba[name] = pipeline.predict_proba(X_val)[:, 1]
        train_ap[name] = average_precision_score(y_train, pipeline.predict_proba(X_train)[:, 1])
    return val_proba, train_ap


def summarize_cv(cv_results, order):
    """Moyenne ± écart-type de chaque métrique sur les 5 découpages."""
    grouped = cv_results.groupby("model")
    mean, sd = grouped[TABLE_METRICS].mean(), grouped[TABLE_METRICS].std()
    table = pd.DataFrame(index=order)
    for metric in TABLE_METRICS:
        table[metric] = [f"{mean.loc[n, metric]:.4f} ± {sd.loc[n, metric]:.4f}" for n in order]
    return table, mean, sd


def validation_tables(val_proba, y_val, learned_names):
    """Métriques de validation au seuil 0,5, puis au meilleur seuil de chaque modèle."""
    default_rows, selected_rows, thresholds = [], [], {}
    for name, proba in val_proba.items():
        default_rows.append({"model": name, "threshold": 0.5, **evaluate_model(y_val, proba)})
        threshold = find_best_threshold(y_val, proba) if name in learned_names else 0.5
        thresholds[name] = threshold
        selected_rows.append({"model": name, "threshold": threshold,
                              **evaluate_model(y_val, proba, threshold)})
    return (pd.DataFrame(default_rows).set_index("model"),
            pd.DataFrame(selected_rows).set_index("model"),
            thresholds)


def progression_table(names, cv_mean, cv_sd, train_ap, selected):
    """Ce que chaque étape apporte par rapport à la précédente."""
    rows, previous = [], None
    for name in names:
        cv_ap = cv_mean.loc[name, "average_precision"]
        rows.append({
            "model": name,
            "train_AP": train_ap[name],
            "CV_AP": cv_ap,
            "CV_AP_sd": cv_sd.loc[name, "average_precision"],
            "gain_CV_AP_vs_etape_precedente": np.nan if previous is None else cv_ap - previous,
            "ecart_train_moins_CV": train_ap[name] - cv_ap,
            "validation_AP": selected.loc[name, "average_precision"],
            "validation_F1_meilleur_seuil": selected.loc[name, "f1"],
        })
        previous = cv_ap
    return pd.DataFrame(rows).set_index("model")


def save_figures(names, cv_mean, cv_sd, train_ap, val_proba, y_val, default, selected, thresholds):
    """Les 4 graphiques du notebook de Hugo, enregistrés pour le rapport."""
    short = ["Logistique", "Random Forest", "Gradient Boosting"]
    x = np.arange(len(names))

    # 1. Progression : AP en validation croisée, et écart train / validation croisée
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, cv_mean.loc[names, "average_precision"], yerr=cv_sd.loc[names, "average_precision"],
                color=COLORS, capsize=4)
    axes[0].scatter(x, selected.loc[names, "average_precision"], marker="D", color="black", label="AP validation")
    axes[0].set(title="AP en validation croisée (± écart-type) et AP validation",
                ylabel="Average Precision", ylim=(0, 1))
    axes[0].set_xticks(x, short)
    axes[0].legend()
    axes[1].bar(x - 0.18, [train_ap[n] for n in names], width=0.36, label="AP sur le train", color="#BCC5CA")
    axes[1].bar(x + 0.18, cv_mean.loc[names, "average_precision"], width=0.36, label="AP validation croisée",
                color=COLORS)
    axes[1].set(title="Surapprentissage : train contre validation croisée",
                ylabel="Average Precision", ylim=(0, 1))
    axes[1].set_xticks(x, short)
    axes[1].legend(loc="upper left")
    _save(fig, "01_progression_modeles.png")

    # 2. Courbes precision-recall et ROC sur la validation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, color in zip(names, COLORS):
        precision, recall, _ = precision_recall_curve(y_val, val_proba[name])
        axes[0].plot(recall, precision, color=color,
                     label=f"{name} : AP={selected.loc[name, 'average_precision']:.3f}")
        fpr, tpr, _ = roc_curve(y_val, val_proba[name])
        axes[1].plot(fpr, tpr, color=color, label=f"{name} : AUC={selected.loc[name, 'roc_auc']:.3f}")
    axes[0].axhline(y_val.mean(), color="grey", linestyle="--", label="Part de >50K")
    axes[0].set(title="Courbes precision-recall (validation)", xlabel="Recall", ylabel="Precision",
                xlim=(0, 1), ylim=(0, 1))
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set(title="Courbes ROC (validation)", xlabel="Taux de faux positifs",
                ylabel="Taux de vrais positifs", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=8)
    _save(fig, "02_courbes_validation.png")

    # 3. Effet du seuil sur precision, recall et F1
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, ax in zip(names, axes):
        precision, recall, grid = precision_recall_curve(y_val, val_proba[name])
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        ax.plot(grid, precision[:-1], label="Precision", color=COLORS[0])
        ax.plot(grid, recall[:-1], label="Recall", color=COLORS[1])
        ax.plot(grid, f1, label="F1", color=COLORS[2])
        ax.axvline(thresholds[name], color="black", linestyle="--", label=f"Seuil retenu : {thresholds[name]:.3f}")
        ax.set(title=name, xlabel="Seuil de décision", ylabel="Score", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.suptitle("Choix du seuil sur la validation")
    _save(fig, "03_choix_du_seuil.png")

    # 4. Matrices de confusion : seuil 0,5 (haut) et meilleur seuil (bas)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, name in enumerate(names):
        for row, results in enumerate([default, selected]):
            r = results.loc[name]
            counts = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=int)
            ax = axes[row, col]
            ax.imshow(counts, cmap="Blues")
            for (i, j), count in np.ndenumerate(counts):
                ax.text(j, i, f"{count:,}", ha="center", va="center",
                        color="white" if count > counts.max() / 2 else "black")
            ax.set_xticks([0, 1], ["<=50K", ">50K"])
            ax.set_yticks([0, 1], ["<=50K", ">50K"])
            ax.set(title=f"{name} | seuil {r.threshold:.3f}", xlabel="Prédit", ylabel="Réel")
    fig.suptitle("Matrices de confusion (validation) : seuil 0,5 en haut, meilleur seuil en bas")
    _save(fig, "04_matrices_confusion.png")


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, groups = load_data()
    print("Train :", X_train.shape, "| Validation :", X_val.shape)

    models = build_models()
    names = list(models)
    candidates = {**models, DUMMY_NAME: DummyClassifier(strategy="most_frequent")}
    order = names + [DUMMY_NAME]

    print("\n=== Validation croisée groupée sur le train (5 découpages, seuil 0,5) ===")
    folds = make_folds(X_train, y_train, groups)
    cv_results = cross_validate_models(candidates, X_train, y_train, folds)
    cv_table, cv_mean, cv_sd = summarize_cv(cv_results, order)
    print(cv_table.to_string())
    best_model = cv_mean.loc[names, "average_precision"].idxmax()
    print(f"\nMeilleur modèle selon l'AP en validation croisée : {best_model}")

    print("\n=== Validation (7 317 personnes) ===")
    val_proba, train_ap = fit_on_full_train(candidates, X_train, y_train, X_val)
    default, selected, thresholds = validation_tables(val_proba, y_val, names)
    print("\nAu seuil 0,5 :")
    print(default.round(4).to_string())
    print("\nAu meilleur seuil de chaque modèle (choisi sur cette même validation, donc optimiste) :")
    print(selected.round(4).to_string())

    print("\n=== Ce que chaque étape apporte ===")
    progression = progression_table(names, cv_mean, cv_sd, train_ap, selected)
    print(progression.round(4).to_string())

    cv_results.to_csv(OUTPUT_DIR / "cv_metriques_par_decoupage.csv", index=False)
    cv_table.to_csv(OUTPUT_DIR / "cv_resume.csv")
    default.to_csv(OUTPUT_DIR / "validation_seuil_0_5.csv")
    selected.to_csv(OUTPUT_DIR / "validation_meilleur_seuil.csv")
    progression.to_csv(OUTPUT_DIR / "progression.csv")
    save_figures(names, cv_mean, cv_sd, train_ap, val_proba, y_val, default, selected, thresholds)
    print(f"\nTableaux et graphiques enregistrés dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
