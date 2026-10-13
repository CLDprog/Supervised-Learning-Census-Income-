"""Tests pour le feature engineering (contribution de Yanis)."""
from pathlib import Path
import sys
import unittest

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from build_features import (  # noqa: E402
    CAPITAL_GAIN_CAP,
    DATA_DIR,
    add_capital_features,
    build_encoder,
    build_missing_value_imputer,
    group_rare_categories,
    load_partition,
    model_inputs,
)


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_df = load_partition(DATA_DIR, "train")
        cls.X_train, cls.y_train, cls.sex_train = model_inputs(train_df)

        cls.imputer = build_missing_value_imputer()
        X_imputed = cls.imputer.fit_transform(cls.X_train)
        X_imputed = pd.DataFrame(X_imputed, columns=cls.imputer.get_feature_names_out())
        X_imputed.index = cls.X_train.index

        cls.X_grouped = group_rare_categories(X_imputed)
        cls.X_featured = add_capital_features(cls.X_grouped)

        cls.encoder = build_encoder()
        cls.X_encoded = cls.encoder.fit_transform(cls.X_featured)

    def test_no_missing_values_after_imputation(self):
        self.assertEqual(self.X_grouped.isna().sum().sum(), 0)

    def test_rare_workclass_categories_are_grouped(self):
        categories = set(self.X_grouped["workclass"].unique())
        self.assertNotIn("Never-worked", categories)
        self.assertNotIn("Without-pay", categories)

    def test_native_country_has_three_categories(self):
        self.assertEqual(
            set(self.X_grouped["native-country"].unique()),
            {"United-States", "Other", "Unknown"},
        )

    def test_capital_gain_capped_matches_raw_value(self):
        capital_gain = pd.to_numeric(self.X_featured["capital-gain"])
        expected = int((capital_gain >= CAPITAL_GAIN_CAP).sum())
        actual = int(self.X_featured["capital_gain_capped"].sum())
        self.assertEqual(expected, actual)

    def test_validation_produces_same_column_count_as_train(self):
        validation_df = load_partition(DATA_DIR, "validation")
        X_val, _, _ = model_inputs(validation_df)

        X_val_imputed = self.imputer.transform(X_val)
        X_val_imputed = pd.DataFrame(X_val_imputed, columns=self.imputer.get_feature_names_out())
        X_val_imputed.index = X_val.index

        X_val_grouped = group_rare_categories(X_val_imputed)
        X_val_featured = add_capital_features(X_val_grouped)
        X_val_encoded = self.encoder.transform(X_val_featured)

        self.assertEqual(X_val_encoded.shape[1], self.X_encoded.shape[1])


if __name__ == "__main__":
    unittest.main()