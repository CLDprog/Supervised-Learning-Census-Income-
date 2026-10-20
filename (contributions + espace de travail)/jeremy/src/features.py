"""Ingénierie des variables et prétraitement (exigence 2)."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, MIN_FREQUENCE_MODALITE


def creer_variables(X):
    """Ajoute les nouvelles variables (2.3). Calcul ligne par ligne : rien n'est appris sur les données.

    - age_carre : capte la courbe en cloche du revenu selon l'âge (équation de Mincer) ;
    - a_des_placements : un gain ou une perte en capital signalent tous deux un patrimoine ;
    - log_capital_gain, log_capital_loss : écrasent les montants extrêmes (jusqu'à 99 999).
    """
    # Toute valeur manquante (y compris None, envoyé par l'API) devient NaN, pour être reconnue par les imputeurs
    X = X.where(X.notna(), np.nan)
    X["age_carre"] = X["age"] ** 2
    X["a_des_placements"] = ((X["capital-gain"] > 0) | (X["capital-loss"] > 0)).astype(int)
    X["log_capital_gain"] = np.log1p(X["capital-gain"])
    X["log_capital_loss"] = np.log1p(X["capital-loss"])
    return X


def construire_preprocessing():
    """Prétraitement complet, neuf à chaque appel : nouvelles variables, puis traitement par type de colonne."""
    categorielles = Pipeline([
        # 2.1 : le manque est informatif (9 % de >50K contre 25 %), il devient une modalité à part entière
        ("imputation", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # 2.2 : one-hot ; les modalités rares et les modalités inconnues sont regroupées
        ("one_hot", OneHotEncoder(min_frequency=MIN_FREQUENCE_MODALITE, handle_unknown="infrequent_if_exist",
                                  sparse_output=False)),
    ])
    numeriques = Pipeline([
        # 2.4 : médiane par sécurité (aucun manquant dans les données), puis standardisation
        ("imputation", SimpleImputer(strategy="median")),
        ("mise_a_echelle", StandardScaler()),
    ])
    traitement_colonnes = ColumnTransformer([
        ("categorielles", categorielles, COLONNES_CATEGORIELLES),
        ("numeriques", numeriques, COLONNES_NUMERIQUES),
    ], remainder="drop")  # fnlwgt, education, sex, race et les montants bruts sont ignorés
    return Pipeline([
        ("nouvelles_variables", FunctionTransformer(creer_variables)),
        ("traitement_colonnes", traitement_colonnes),
    ])
