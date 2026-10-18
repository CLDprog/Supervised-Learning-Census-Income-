"""Tests de la partie 4 : comparaison des modèles."""
import unittest

import numpy as np

from census_income.compare_models import build_models, load_data, make_folds, model_pipeline


class TestCompareModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_val, cls.y_val, cls.groups = load_data()

    def test_three_models_without_class_weighting(self):
        models = build_models()
        self.assertEqual(len(models), 3)
        for model in models.values():
            self.assertIsNone(model.get_params().get("class_weight"))

    def test_folds_cover_train_once_and_never_split_a_group(self):
        folds = make_folds(self.X_train, self.y_train, self.groups)
        self.assertEqual(len(folds), 5)
        scored = np.concatenate([score_idx for _, score_idx in folds])
        self.assertEqual(sorted(scored.tolist()), list(range(len(self.X_train))))
        for fit_idx, score_idx in folds:
            self.assertTrue(set(self.groups[fit_idx]).isdisjoint(self.groups[score_idx]))

    def test_pipeline_gives_50_dense_columns(self):
        pipeline = model_pipeline(build_models()["Régression logistique"])
        pipeline.fit(self.X_train, self.y_train)
        transformed = pipeline[:-1].transform(self.X_val.iloc[:5])
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed.shape, (5, 50))


if __name__ == "__main__":
    unittest.main()
