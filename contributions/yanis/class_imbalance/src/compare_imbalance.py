"""Partie 3 : gestion du déséquilibre des classes (contribution de Yanis)."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from contributions.yanis.class_imbalance.src.evaluate import evaluate_model
from contributions.yanis.feature_engineering.src.build_features import (
    DATA_DIR,
    add_capital_features,
    build_encoder,
    build_missing_value_imputer,
    group_rare_categories,
    load_partition,
    model_inputs,
)


def prepare_features():
    """Charge train et validation, et applique la chaîne de la partie 2.

    Tout est appris (fit) sur le train uniquement, puis seulement
    appliqué (transform) sur la validation.
    """
    X_train, y_train, _ = model_inputs(load_partition(DATA_DIR, "train"))
    X_val, y_val, _ = model_inputs(load_partition(DATA_DIR, "validation"))

    imputer = build_missing_value_imputer()
    encoder = build_encoder()

    # Train : on apprend ET on transforme
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=imputer.get_feature_names_out(),
        index=X_train.index,
    )
    X_train_featured = add_capital_features(group_rare_categories(X_train_imputed))
    X_train_encoded = encoder.fit_transform(X_train_featured)

    # Validation : on transforme seulement
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=imputer.get_feature_names_out(),
        index=X_val.index,
    )
    X_val_featured = add_capital_features(group_rare_categories(X_val_imputed))
    X_val_encoded = encoder.transform(X_val_featured)

    return X_train_encoded, y_train, X_val_encoded, y_val


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

if __name__ == "__main__":
    main()