"""Partie 5 : la chaîne complète (preprocessing de la partie 2 + modèle) en un seul objet."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from census_income.features import (
    add_capital_features,
    build_encoder,
    build_missing_value_imputer,
    group_rare_categories,
)


def build_pipeline(model):
    """Range toute la partie 2 puis le modèle dans un seul objet.

    Indispensable pour la validation croisée : à chaque découpage,
    l'imputation et l'encodage sont réappris sur la partie
    d'entraînement uniquement (pas de fuite de données).
    """
    return Pipeline([
        ("impute", build_missing_value_imputer().set_output(transform="pandas")),
        ("group_rare", FunctionTransformer(group_rare_categories)),
        ("capital", FunctionTransformer(add_capital_features)),
        ("encode", build_encoder()),
        ("model", model),
    ])
