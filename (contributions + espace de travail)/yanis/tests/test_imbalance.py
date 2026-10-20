"""Tests de la partie 3 : déséquilibre des classes (contribution de Yanis)."""
import unittest

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from census_income.evaluation import evaluate_model, find_best_threshold
from census_income.imbalance import prepare_features


class TestEvaluateModel(unittest.TestCase):
    """Tests sur de petits exemples calculables à la main."""

    def test_metrics_match_hand_calculation(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.7, 0.3])
        r = evaluate_model(y_true, y_proba)
        self.assertAlmostEqual(r["average_precision"], 0.75)
        self.assertAlmostEqual(r["roc_auc"], 4 / 6)
        self.assertAlmostEqual(r["precision"], 0.5)
        self.assertAlmostEqual(r["recall"], 0.5)
        self.assertAlmostEqual(r["accuracy"], 0.6)
        self.assertEqual((r["tn"], r["fp"], r["fn"], r["tp"]), (2, 1, 1, 1))

    def test_best_threshold_on_perfectly_separated_example(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(find_best_threshold(y_true, y_proba), 0.8)


class TestOnRealData(unittest.TestCase):
    """Tests sur les vraies données (train et validation)."""

    @classmethod
    def setUpClass(cls):
        cls.X_train, cls.y_train, cls.X_val, cls.y_val = prepare_features()

    def test_dummy_model_has_zero_recall_despite_high_accuracy(self):
        dummy = DummyClassifier(strategy="most_frequent").fit(self.X_train, self.y_train)
        r = evaluate_model(self.y_val, dummy.predict_proba(self.X_val)[:, 1])
        self.assertEqual(r["recall"], 0.0)
        self.assertGreater(r["accuracy"], 0.75)

    def test_best_threshold_beats_default_threshold(self):
        model = LogisticRegression(max_iter=1000, random_state=42).fit(self.X_train, self.y_train)
        y_proba = model.predict_proba(self.X_val)[:, 1]
        best = find_best_threshold(self.y_val, y_proba)
        self.assertTrue(0 < best < 1)
        f1_best = evaluate_model(self.y_val, y_proba, threshold=best)["f1"]
        f1_default = evaluate_model(self.y_val, y_proba)["f1"]
        self.assertGreaterEqual(f1_best, f1_default)


if __name__ == "__main__":
    unittest.main()