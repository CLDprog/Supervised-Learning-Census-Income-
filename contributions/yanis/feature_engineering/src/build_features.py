"""Feature engineering - Census Income (contribution de Yanis)."""

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

def main():
    train_df = load_partition(DATA_DIR, "train")
    X_train, y_train, sex_train = model_inputs(train_df)

    print("Forme de X_train :", X_train.shape)
    print("Valeurs manquantes avant imputation :")
    print(X_train.isna().sum())

    imputer = build_missing_value_imputer()
    X_train_imputed = imputer.fit_transform(X_train)

    import pandas as pd
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

if __name__ == "__main__":
    main()

