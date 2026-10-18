"""Tests de l'endpoint de prédiction, avec un modèle final entraîné dans un dossier temporaire."""
import os
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from census_income.config import BEST_PARAMS_PATH
from census_income.train_final import train_final_model

PERSON = {
    "age": 45, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
    "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States",
}


@unittest.skipUnless(BEST_PARAMS_PATH.exists(), "lancer d'abord python -m census_income.tuning")
class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        model_path = Path(cls.temp.name) / "model.joblib"
        cls.artifact = train_final_model(output_path=model_path, log_to_mlflow=False)
        os.environ["CENSUS_MODEL_PATH"] = str(model_path)
        from census_income.api import app
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        os.environ.pop("CENSUS_MODEL_PATH", None)
        cls.temp.cleanup()

    def test_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok", "model_loaded": True})

    def test_prediction_is_consistent_with_threshold(self):
        result = self.client.post("/predict", json=PERSON).json()
        self.assertTrue(0 <= result["probability_gt_50k"] <= 1)
        expected = ">50K" if result["probability_gt_50k"] >= result["threshold"] else "<=50K"
        self.assertEqual(result["prediction"], expected)

    def test_missing_categories_are_accepted(self):
        person = {**PERSON, "workclass": None, "occupation": None, "native-country": None}
        self.assertEqual(self.client.post("/predict", json=person).status_code, 200)

    def test_batch_and_invalid_input(self):
        self.assertEqual(len(self.client.post("/predict/batch", json=[PERSON, PERSON]).json()), 2)
        self.assertEqual(self.client.post("/predict", json={**PERSON, "age": 5}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
