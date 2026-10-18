"""API de prédiction : expose le Pipeline final (prétraitement + modèle) derrière un point d'accès HTTP.

Lancement : uvicorn api.main:app --reload   (après python -m src.train)
Documentation interactive : http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from src.evaluation import charger_modele_final

etat = {}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    """Charge le modèle une seule fois, au démarrage de l'API."""
    etat["modele"], etat["metadonnees"] = charger_modele_final()
    yield
    etat.clear()


app = FastAPI(
    title="Census Income : ciblage d'une offre premium",
    description="Estime la probabilité qu'un client gagne plus de 50 K$ par an et indique s'il faut le contacter.",
    lifespan=cycle_de_vie,
)


class Client(BaseModel):
    """Données brutes d'un client. Les champs inconnus peuvent être omis (ils deviennent « Unknown »)."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": {
        "age": 45, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial", "relationship": "Husband", "capital-gain": 0, "capital-loss": 0,
        "hours-per-week": 50, "native-country": "United-States",
    }})

    age: int = Field(ge=17, le=100)
    workclass: str | None = None
    education_num: int = Field(alias="education-num", ge=1, le=16, description="Niveau d'études, de 1 à 16")
    marital_status: str | None = Field(default=None, alias="marital-status")
    occupation: str | None = None
    relationship: str | None = None
    capital_gain: float = Field(default=0, alias="capital-gain", ge=0)
    capital_loss: float = Field(default=0, alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: str | None = Field(default=None, alias="native-country")


class Prediction(BaseModel):
    probabilite_plus_50k: float
    a_contacter: bool
    seuil: float


@app.get("/health")
def sante():
    """Vérifie que l'API répond et indique le modèle chargé."""
    return {"statut": "ok", "modele": etat["metadonnees"]["modele"], "seuil": etat["metadonnees"]["seuil"]}


@app.post("/predict", response_model=Prediction)
def predire(client: Client):
    """Probabilité que le client gagne plus de 50 K$, et décision de contact au seuil retenu (80 % de rappel)."""
    donnees = pd.DataFrame([client.model_dump(by_alias=True)])
    probabilite = float(etat["modele"].predict_proba(donnees)[0, 1])
    seuil = etat["metadonnees"]["seuil"]
    return Prediction(probabilite_plus_50k=probabilite, a_contacter=probabilite >= seuil, seuil=seuil)
