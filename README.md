# Supervised Learning - Census Income

**Dataset : Adult / Census Income** (OpenML, version 2) : 48 842 personnes du recensement américain de 1994.
**Question : le revenu annuel d'une personne dépasse-t-il 50 000 $ ?** Classification binaire, classe positive `>50K` (24 % des personnes).

Projet du cours *Supervised Learning & Model Engineering* (MSA-DATB09-04) : une preuve de concept complète, des données brutes jusqu'à un endpoint de prédiction.
Groupe : Jérémy Caillaud, Yanis Rajhi, Hugo van Zuylen.

## Résultats principaux

**Modèle final : Gradient Boosting (`HistGradientBoostingClassifier`) réglé avec Optuna, sans pondération des classes, seuil de décision 0,404** (choisi sur la validation pour maximiser le F1).

| Évaluation finale sur le **test** (7 317 personnes, ouvert une seule fois) | Valeur |
|---|---|
| Average Precision (métrique principale) | **0,833** |
| ROC-AUC | 0,931 |
| Precision / Recall / F1 (classe `>50K`) | 0,720 / 0,749 / **0,734** |
| Brier score (calibration) | 0,087 |

| Modèle (validation croisée groupée en 5 découpages, train) | AP |
|---|---|
| Modèle naïf (toujours `<=50K`) | 0,239 |
| Régression logistique (réglée) | 0,737 |
| Random Forest (réglé) | 0,809 |
| **Gradient Boosting (réglé)** | **0,827** |

- **Déséquilibre** : la pondération des classes ne change pas la qualité du classement (AP et ROC-AUC identiques en validation croisée). On garde un entraînement normal et on abaisse le seuil de décision.
- **Calibration** : ECE de 0,009 sur la validation. Les probabilités sont fiables telles quelles.
- **Explicabilité (SHAP)** : statut marital, âge, plus-values, niveau d'études et heures travaillées dominent.
- **Équité (sexe)** : l'écart de rappel entre femmes et hommes passe de 12,1 à 0,8 point avec un seuil par groupe, sans perte de F1 (0,734 → 0,736).

## Installation

Python 3.13.

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate        # Windows  (macOS / Linux : source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```

Toutes les commandes se lancent **depuis la racine du dépôt**.

## Reproduire le projet, étape par étape

| # | Étape de l'énoncé | Commande | Résultats |
|---|---|---|---|
| 1 | Découpage train / validation / test (70 / 15 / 15) | `python -m census_income.data` | `data/processed/` |
| 1 | EDA sur le train uniquement | `python -m census_income.explore` | `reports/eda/` |
| 2 | Feature engineering | `python -m census_income.features` | console |
| 3 | Déséquilibre des classes | `python -m census_income.imbalance` | console |
| 4 | Comparaison de 3 modèles | `python -m census_income.compare_models` | `reports/model_comparison/` |
| 6, 7 | Réglage Optuna + suivi MLflow (~30 min) | `python -m census_income.tuning` | `reports/tuning/`, `mlflow.db` |
| 5 | Modèle final (Pipeline + seuil) | `python -m census_income.train_final` | `models/final_model.joblib` |
| 8 | Explicabilité SHAP | `python -m census_income.interpret` | `reports/explainability/` |
| 9 | Calibration | `python -m census_income.calibration` | `reports/calibration/` |
| 10 | Évaluation finale sur le test + audit d'équité | `python -m census_income.fairness` | `reports/fairness/` |
| — | Endpoint de prédiction | `uvicorn census_income.api:app --reload` | http://127.0.0.1:8000/docs |
| — | Tests automatisés | `python -m unittest discover -s tests -v` | console |

Les réglages Optuna retenus sont versionnés dans `reports/tuning/best_params.json` : on peut sauter l'étape de réglage (la plus longue) et lancer directement `train_final`.

Les 89 exécutions MLflow (85 essais Optuna, 3 recherches, le modèle final) sont exportées dans `reports/tuning/mlflow_runs.csv`. Pour les consulter dans l'interface MLflow :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Utiliser l'endpoint

Après `train_final`, lancer `uvicorn census_income.api:app --reload`, puis :

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\": 45, \"workclass\": \"Private\", \"education-num\": 13, \"marital-status\": \"Married-civ-spouse\", \"occupation\": \"Exec-managerial\", \"relationship\": \"Husband\", \"race\": \"White\", \"capital-gain\": 0, \"capital-loss\": 0, \"hours-per-week\": 50, \"native-country\": \"United-States\"}"
```

Réponse : la probabilité de `>50K`, le seuil de décision et la prédiction. La page `/docs` permet aussi de tester l'API depuis le navigateur. Le modèle n'utilise pas la variable `sex`.

## Structure du dépôt

```
census_income/        le code, un fichier par étape
  data.py             chargement, nettoyage, découpage (empreintes SHA-256 des partitions)
  explore.py          EDA sur le train uniquement
  features.py         partie 2 : imputation, catégories rares, features capital, encodage
  pipeline.py         partie 5 : preprocessing + modèle en un seul Pipeline
  evaluation.py       métriques communes et choix du seuil
  imbalance.py        partie 3
  compare_models.py   partie 4
  tuning.py           parties 6 et 7
  train_final.py      partie 5 : modèle final et seuil
  interpret.py        partie 8
  calibration.py      partie 9
  fairness.py         partie 10 (seul script qui ouvre le test)
  api.py              endpoint FastAPI
  config.py           lecture de config.yaml
config.yaml           réglages communs (graine, essais Optuna, MLflow...)
tests/                tests automatisés (unittest)
reports/              tableaux et graphiques utilisés dans le rapport
notebooks/            supports de démonstration (l'implémentation est dans census_income/)
data/raw/             données brutes (census_income.xlsx)
docs/                 cadrage, protocole de validation, guide de soutenance, historique des contributions, guide Git
report/               rapport LaTeX
```

## Protocole, en bref

- **Découpage** 70 / 15 / 15 stratifié (graine 42), après suppression de 52 doublons exacts. Les profils identiques restent dans la même partition.
- **Train** : apprentissage, validation croisée groupée en 5 découpages, réglage Optuna.
- **Validation** : choix du seuil de décision (maximum de F1) et des réglages d'atténuation.
- **Test** : ouvert **une seule fois**, par `census_income/fairness.py`, avec le modèle et le seuil figés.
- Le preprocessing est toujours appris à l'intérieur du `Pipeline`, donc sur la seule partie d'entraînement : pas de fuite de données.

Détails : [docs/problem_framing.md](docs/problem_framing.md) et [docs/validation_protocol_addendum.md](docs/validation_protocol_addendum.md).

## Usage d'assistants IA

Le développement a été assisté par Claude (Anthropic) : explications, écriture et relecture de code, réorganisation du dépôt, rédaction de la documentation. Les choix méthodologiques ont été discutés et validés par le groupe ; le détail figure dans le rapport.
