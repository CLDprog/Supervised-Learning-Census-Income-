"""Endpoint de prédiction (FastAPI) : sert le modèle final enregistré.

Lancement : uvicorn census_income.api:app --reload   (après census_income.train_final)
Documentation interactive : http://127.0.0.1:8000/docs

Le modèle n'utilise pas la variable sex (exclue des prédicteurs pour l'audit d'équité).
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from census_income.config import FINAL_MODEL_PATH
from census_income.train_final import load_final_model

MODEL = {}


class Person(BaseModel):
    """Les 11 informations utilisées par le modèle (noms identiques au dataset)."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": {
        "age": 45, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
        "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States",
    }})

    age: int = Field(ge=17, le=100)
    workclass: str | None = None
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: str | None = None
    relationship: str
    race: str
    capital_gain: int = Field(0, alias="capital-gain", ge=0)
    capital_loss: int = Field(0, alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: str | None = Field(None, alias="native-country")


class Prediction(BaseModel):
    probability_gt_50k: float
    threshold: float
    prediction: str


def model_path():
    return Path(os.environ.get("CENSUS_MODEL_PATH", FINAL_MODEL_PATH))


@asynccontextmanager
async def lifespan(app):
    try:
        MODEL.update(load_final_model(model_path()))
    except FileNotFoundError:
        MODEL.clear()  # l'API démarre quand même ; /predict renverra une erreur 503 explicite
    yield


app = FastAPI(title="Census Income - prédiction du revenu >50K", lifespan=lifespan)


def to_frame(people):
    """Transforme les personnes reçues en tableau, avec les colonnes attendues par le Pipeline."""
    rows = [p.model_dump(by_alias=True) for p in people]
    frame = pd.DataFrame(rows)[MODEL["features"]]
    return frame.astype(object).where(frame.notna(), np.nan).infer_objects()


def predict(people):
    if not MODEL:
        raise HTTPException(503, "Modèle absent : lancer python -m census_income.train_final")
    proba = MODEL["pipeline"].predict_proba(to_frame(people))[:, 1]
    return [Prediction(probability_gt_50k=round(float(p), 4), threshold=round(MODEL["threshold"], 4),
                       prediction=">50K" if p >= MODEL["threshold"] else "<=50K") for p in proba]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(MODEL)}


@app.get("/model")
def model_info():
    if not MODEL:
        raise HTTPException(503, "Modèle absent : lancer python -m census_income.train_final")
    return {"model": MODEL["model_name"], "params": MODEL["params"], "threshold": MODEL["threshold"],
            "features": MODEL["features"], "validation_metrics": MODEL["validation_metrics"]}


@app.post("/predict", response_model=Prediction)
def predict_one(person: Person):
    return predict([person])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_many(people: list[Person]):
    return predict(people)
