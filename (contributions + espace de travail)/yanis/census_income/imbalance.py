"""Partie 3 : gestion du déséquilibre des classes (contribution de Yanis)."""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from census_income.data import DATA_DIR, load_partition, model_inputs
from census_income.evaluation import evaluate_model, find_best_threshold
from census_income.pipeline import build_pipeline


def prepare_features():
    """Charge train et validation, et applique la chaîne de la partie 2.

    Tout est appris (fit) sur le train uniquement, puis seulement
    appliqué (transform) sur la validation.
    """
    X_train, y_train, _ = model_inputs(load_partition(DATA_DIR, "train"))
    X_val, y_val, _ = model_inputs(load_partition(DATA_DIR, "validation"))

    preprocessing = build_pipeline("passthrough")
    X_train_encoded = preprocessing.fit_transform(X_train)
    X_val_encoded = preprocessing.transform(X_val)
    return X_train_encoded, y_train, X_val_encoded, y_val


CV_METRICS = ["average_precision", "roc_auc", "precision", "recall", "f1"]


def cross_validate_model(title, model, X, y):
    """Validation croisée en 5 découpages : affiche moyenne ± écart."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(build_pipeline(model), X, y, cv=cv, scoring=CV_METRICS)

    print()
    print(title)
    for metric in CV_METRICS:
        values = scores[f"test_{metric}"]
        print(f"{metric:>18} : {values.mean():.3f} ± {values.std():.3f}")


def print_results(title, y_true, y_proba):
    """Affiche les métriques d'un modèle et les renvoie."""
    results = evaluate_model(y_true, y_proba)
    print()
    print(title)
    for name, value in results.items():
        print(f"{name:>18} : {value}")
    return results


def main():
    X_train, y_train, X_val, y_val = prepare_features()
    print("Train :", X_train.shape, "| Validation :", X_val.shape)

    # 1. Modèle naïf
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print_results(
        "Modèle naïf (répond toujours <=50K) :",
        y_val,
        dummy.predict_proba(X_val)[:, 1],
    )

    # 2. Régression logistique sans remède
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    logreg.fit(X_train, y_train)
    print_results(
        "Régression logistique SANS remède :",
        y_val,
        logreg.predict_proba(X_val)[:, 1],
    )
    
    # 3. Régression logistique AVEC remède
    logreg_balanced = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    logreg_balanced.fit(X_train, y_train)
    print_results(
        "Régression logistique AVEC class_weight='balanced' :",
        y_val,
        logreg_balanced.predict_proba(X_val)[:, 1],
    )
    
        # 4. Validation croisée sur le train, pour confirmer le choix du remède
    print()
    print("=== Validation croisée sur le train (5 découpages) ===")
    X_train_raw, y_train_raw, _ = model_inputs(load_partition(DATA_DIR, "train"))

    cross_validate_model(
        "Logistique SANS remède :",
        LogisticRegression(max_iter=1000, random_state=42),
        X_train_raw,
        y_train_raw,
    )
    cross_validate_model(
        "Logistique AVEC class_weight='balanced' :",
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        X_train_raw,
        y_train_raw,
    )
    
        # 5. Démonstration : baisser le seuil du modèle SANS remède
    print()
    print("=== Logistique SANS remède, avec différents seuils (validation) ===")
    proba_logreg = logreg.predict_proba(X_val)[:, 1]
    for threshold in [0.5, 0.4, 0.3, 0.25, 0.2]:
        r = evaluate_model(y_val, proba_logreg, threshold=threshold)
        print(
            f"seuil {threshold:.2f} -> precision {r['precision']:.3f}"
            f" | recall {r['recall']:.3f} | f1 {r['f1']:.3f}"
        )
        
            # 6. Remède retenu : pas de pondération, seuil qui maximise le F1 sur la validation
    best_threshold = find_best_threshold(y_val, proba_logreg)
    print()
    print(f"=== Remède retenu : sans pondération, seuil = {best_threshold:.3f} ===")
    for name, value in evaluate_model(y_val, proba_logreg, threshold=best_threshold).items():
        print(f"{name:>18} : {value}")

if __name__ == "__main__":
    main()