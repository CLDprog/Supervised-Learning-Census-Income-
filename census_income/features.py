"""Partie 2 : feature engineering (imputation, catégories rares, features capital, encodage)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from census_income.data import DATA_DIR, load_partition, model_inputs

def build_missing_value_imputer() -> ColumnTransformer:
    """Impute les 3 colonnes à valeurs manquantes, chacune avec sa règle."""
    return ColumnTransformer(
        transformers=[
            (
                "workclass_occupation",
                SimpleImputer(strategy="constant", fill_value="Unemployed"),
                ["workclass", "occupation"],
            ),
            (
                "native_country",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
                ["native-country"],
            ),
        ],
                remainder="passthrough",
        verbose_feature_names_out=False,
    )

RARE_WORKCLASS = ["Never-worked", "Without-pay"]

def group_rare_categories(X):
    """Regroupe les catégories rares sous une étiquette 'Other'.

    Règle fixe (pas apprise sur les données) : pas de risque de fuite,
    peut donc être appliquée telle quelle sur train, validation et test.
    """
    X = X.copy()

    is_rare_workclass = X["workclass"].isin(RARE_WORKCLASS)
    X["workclass"] = X["workclass"].where(~is_rare_workclass, "Other")

    keep_native_country = X["native-country"].isin(["United-States", "Unknown"])
    X["native-country"] = X["native-country"].where(keep_native_country, "Other")

    return X


CAPITAL_GAIN_CAP = 99_999

def add_capital_features(X):
    """..."""
    X = X.copy()

    capital_gain = pd.to_numeric(X["capital-gain"])
    capital_loss = pd.to_numeric(X["capital-loss"])

    X["capital_gain_log"] = np.log1p(capital_gain)
    X["capital_loss_log"] = np.log1p(capital_loss)
    X["capital_gain_capped"] = (capital_gain >= CAPITAL_GAIN_CAP).astype(int)

    return X


NUMERIC_FEATURES = ["age", "education-num", "hours-per-week", "capital_gain_log", "capital_loss_log"]
PASSTHROUGH_FEATURES = ["capital_gain_capped"]
CATEGORICAL_FEATURES = ["workclass", "occupation", "marital-status", "relationship", "race", "native-country"]


def build_encoder() -> ColumnTransformer:
    """Standardise les numériques, encode les catégorielles, laisse l'indicateur binaire tel quel.

    Tout ce qui n'est pas listé ici (capital-gain, capital-loss bruts) est
    supprimé automatiquement (remainder='drop', comportement par défaut).
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("passthrough", "passthrough", PASSTHROUGH_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def main():
    """Résumé de la partie 2 : la chaîne est apprise sur le train, puis appliquée à la validation."""
    from census_income.pipeline import build_pipeline

    X_train, _, _ = model_inputs(load_partition(DATA_DIR, "train"))
    X_val, _, _ = model_inputs(load_partition(DATA_DIR, "validation"))
    preprocessing = build_pipeline("passthrough")
    X_train_encoded = preprocessing.fit_transform(X_train)
    X_val_encoded = preprocessing.transform(X_val)
    grouped = preprocessing[:3].transform(X_train)

    print("Valeurs manquantes avant :", X_train.isna().sum()[lambda s: s > 0].to_dict())
    print("Valeurs manquantes après :", int(grouped.isna().sum().sum()))
    print("workclass après regroupement :", grouped["workclass"].value_counts().to_dict())
    print("native-country après regroupement :", grouped["native-country"].value_counts().to_dict())
    print("Lignes au plafond de capital-gain :", int(grouped["capital_gain_capped"].sum()))
    print("Colonnes finales :", X_train_encoded.shape[1], "(train)", X_val_encoded.shape[1], "(validation)")


if __name__ == "__main__":
    main()
