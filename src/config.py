"""Paramètres communs à tout le projet : chemins, colonnes et choix du cadrage."""

from pathlib import Path

# Chemins (calculés à partir de l'emplacement de ce fichier, quel que soit le dossier de lancement)
RACINE = Path(__file__).resolve().parents[1]
DOSSIER_MODELES = RACINE / "modeles"
DOSSIER_FIGURES = RACINE / "figures"
DOSSIER_RESULTATS = RACINE / "resultats"
CHEMIN_MODELE_FINAL = DOSSIER_MODELES / "modele_final.joblib"
CHEMIN_METADONNEES = DOSSIER_MODELES / "metadonnees.json"

# Suivi des expériences
MLFLOW_URI = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIENCE = "census-income"

# Reproductibilité et découpage 70 / 15 / 15
RANDOM_STATE = 42
PART_TEST = 0.15
PART_VALIDATION = 0.15
N_PLIS = 5

# Cible
CIBLE = "class"
CLASSE_POSITIVE = ">50K"

# Variables utilisées par le modèle (fnlwgt, education, sex et race sont ignorées)
COLONNES_CATEGORIELLES = ["workclass", "marital-status", "occupation", "relationship", "native-country"]
COLONNES_NUMERIQUES = ["age", "age_carre", "education-num", "hours-per-week",
                       "log_capital_gain", "log_capital_loss", "a_des_placements"]

# Colonnes brutes attendues en entrée (par exemple par l'API)
COLONNES_ENTREE = ["age", "workclass", "education-num", "marital-status", "occupation", "relationship",
                   "capital-gain", "capital-loss", "hours-per-week", "native-country"]

# Attribut sensible : réservé à l'audit d'équité, jamais utilisé comme variable du modèle
ATTRIBUT_SENSIBLE = "sex"

# Regroupement des modalités rares dans le one-hot (justification : marge d'environ ±12 points sous 50 lignes)
MIN_FREQUENCE_MODALITE = 50

# Règle de décision : seuil le plus élevé qui garde au moins 80 % de rappel, choisi sur la validation
RAPPEL_CIBLE = 0.80

# Budget Optuna (nombre d'essais par modèle)
BUDGETS_OPTUNA = {"Régression logistique": 30, "Random Forest": 30, "LightGBM": 50}
