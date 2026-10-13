"""Feature engineering - Census Income (contribution de Yanis)."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

CONTRIBUTIONS_DIR = Path(__file__).resolve().parents[3]
HUGO_SRC = CONTRIBUTIONS_DIR / "hugo" / "eda_and_data_splitting"

sys.path.insert(0, str(HUGO_SRC))
from src.prepare_data import load_partition, model_inputs  # noqa: E402

DATA_DIR = HUGO_SRC / "data" / "processed"

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

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

from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    train_df = load_partition(DATA_DIR, "train")
    X_train, y_train, sex_train = model_inputs(train_df)

    print("Forme de X_train :", X_train.shape)
    print("Valeurs manquantes avant imputation :")
    print(X_train.isna().sum())

    imputer = build_missing_value_imputer()
    X_train_imputed = imputer.fit_transform(X_train)

    X_train_imputed = pd.DataFrame(
        X_train_imputed, columns=imputer.get_feature_names_out()
    )
    X_train_imputed.index = X_train.index
    
    X_train_grouped = group_rare_categories(X_train_imputed)

    print()
    print("Modalités de workclass après regroupement :")
    print(X_train_grouped["workclass"].value_counts())
    print()
    print("Modalités de native-country après regroupement :")
    print(X_train_grouped["native-country"].value_counts())

    print()
    print("Valeurs manquantes après imputation :")
    print(X_train_imputed.isna().sum())

    X_train_featured = add_capital_features(X_train_grouped)

    print()
    print("Nouvelles colonnes :")
    print(X_train_featured[["capital-gain", "capital_gain_log", "capital_gain_capped", "capital-loss", "capital_loss_log"]].describe())
    print()
    print("Nombre de lignes au plafond (capital_gain_capped == 1) :")
    print((X_train_featured["capital_gain_capped"] == 1).sum())
    
    encoder = build_encoder()
    X_train_encoded = encoder.fit_transform(X_train_featured)

    print()
    print("Forme de X_train après encodage :", X_train_encoded.shape)
    print("Nombre total de colonnes générées :", len(encoder.get_feature_names_out()))
    print("Premières colonnes :", list(encoder.get_feature_names_out())[:15])
    
    print()
    print("=== Application sur validation ===")

    validation_df = load_partition(DATA_DIR, "validation")
    X_val, y_val, sex_val = model_inputs(validation_df)

    X_val_imputed = imputer.transform(X_val)  # transform, pas fit_transform
    X_val_imputed = pd.DataFrame(X_val_imputed, columns=imputer.get_feature_names_out())
    X_val_imputed.index = X_val.index

    X_val_grouped = group_rare_categories(X_val_imputed)
    X_val_featured = add_capital_features(X_val_grouped)

    X_val_encoded = encoder.transform(X_val_featured)  # transform, pas fit_transform

    print("Forme de X_val après encodage :", X_val_encoded.shape)
    print("Même nombre de colonnes que train ? ->", X_val_encoded.shape[1] == X_train_encoded.shape[1])

if __name__ == "__main__":
    main()

