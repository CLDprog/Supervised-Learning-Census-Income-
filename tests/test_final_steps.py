"""Tests des parties 6 à 10 (réglage, explicabilité, calibration, équité) sur de petits exemples."""
import unittest

import numpy as np
import optuna

from census_income.calibration import calibration_errors, reliability_table
from census_income.fairness import equal_opportunity_thresholds, predict_with_group_thresholds
from census_income.interpret import aggregate_by_variable, original_variable
from census_income.tuning import MODEL_NAMES, STARTING_PARAMS, make_model, suggest_params


class TestTuning(unittest.TestCase):
    def test_search_spaces_build_models_without_class_weighting(self):
        for key in MODEL_NAMES:
            trial = optuna.trial.FixedTrial(STARTING_PARAMS[key])
            model = make_model(key, suggest_params(key, trial))
            self.assertIsNone(model.get_params().get("class_weight"))


class TestInterpret(unittest.TestCase):
    def test_encoded_columns_map_back_to_original_variables(self):
        self.assertEqual(original_variable("workclass_Private"), "workclass")
        self.assertEqual(original_variable("native-country_Unknown"), "native-country")
        self.assertEqual(original_variable("capital_gain_capped"), "capital-gain")
        self.assertEqual(original_variable("age"), "age")

    def test_one_hot_columns_are_summed_before_taking_absolute_values(self):
        values = np.array([[0.5, -0.5, 1.0], [0.2, 0.2, -1.0]])
        names = ["race_White", "race_Black", "age"]
        importance = aggregate_by_variable(values, names)
        self.assertAlmostEqual(importance["race"], (0.0 + 0.4) / 2)
        self.assertAlmostEqual(importance["age"], 1.0)


class TestCalibration(unittest.TestCase):
    def test_perfectly_calibrated_probabilities_have_zero_error(self):
        proba = np.array([0.25] * 4 + [0.75] * 4)
        y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
        ece, signed = calibration_errors(y, proba)
        self.assertAlmostEqual(ece, 0.0)
        self.assertAlmostEqual(signed, 0.0)

    def test_overconfident_model_has_positive_signed_error(self):
        proba = np.array([0.95] * 10)
        y = np.array([1] * 5 + [0] * 5)
        ece, signed = calibration_errors(y, proba)
        self.assertAlmostEqual(ece, 0.45)
        self.assertGreater(signed, 0)
        self.assertEqual(reliability_table(y, proba)["effectif"].sum(), 10)


class TestFairness(unittest.TestCase):
    def test_group_thresholds_equalize_recall(self):
        rng = np.random.default_rng(0)
        sensitive = np.array(["A"] * 200 + ["B"] * 200)
        y = np.tile([1, 0], 200)
        proba = np.where(sensitive == "A", rng.uniform(0.3, 1, 400), rng.uniform(0, 0.7, 400))
        thresholds = equal_opportunity_thresholds(y, proba, sensitive, target_recall=0.8)
        pred = predict_with_group_thresholds(proba, sensitive, thresholds)
        for group in ["A", "B"]:
            positives = (sensitive == group) & (y == 1)
            self.assertAlmostEqual(pred[positives].mean(), 0.8, delta=0.02)


if __name__ == "__main__":
    unittest.main()
