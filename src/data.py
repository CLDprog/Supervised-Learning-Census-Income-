"""Chargement des données et découpage train / validation / test (exigence 1)."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import CIBLE, CLASSE_POSITIVE, PART_TEST, PART_VALIDATION, RANDOM_STATE


@dataclass
class Jeux:
    """Les trois jeux de données, avec leurs variables explicatives et leur cible."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def charger_donnees():
    """Charge Adult / Census Income et supprime les 52 doublons stricts."""
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=CIBLE)
    y = (df[CIBLE] == CLASSE_POSITIVE).astype(int)
    return X, y


def decouper(X, y):
    """Découpage 70 / 15 / 15 stratifié sur la cible, avec une graine fixe."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=PART_TEST, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=PART_VALIDATION / (1 - PART_TEST), stratify=y_temp, random_state=RANDOM_STATE
    )
    return Jeux(X_train, X_val, X_test, y_train, y_val, y_test)


def charger_jeux():
    """Charge les données et renvoie les trois jeux."""
    return decouper(*charger_donnees())
