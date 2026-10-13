# Supervised Learning - Census Income - Part 2 Feature Engineering

**Contribution de Yanis.** Construit sur le split train/validation/test de
[Hugo](../hugo/eda_and_data_splitting/) et sur les constats de l'EDA initiale
([EDA.ipynb](../../EDA.ipynb)). Ce dossier ne modifie aucun fichier existant du
dépôt ; il ajoute uniquement le travail de la partie 2 (requirement technique
n°2 : encoder les catégorielles, gérer les valeurs manquantes, créer des
features justifiées).

## Provenance

| Repris de l'existant | Ajouté ici |
|---|---|
| Split train/validation/test (`load_partition`, `model_inputs` de Hugo) | Imputation des valeurs manquantes |
| Constats de l'EDA (colonnes manquantes, plafond `capital-gain`, catégories rares) | Regroupement des catégories rares |
| 11 prédicteurs bruts (`sex`, `fnlwgt`, `education` déjà exclus par Hugo) | 3 nouvelles features dérivées de `capital-gain`/`capital-loss` |
| | Encodage final (standardisation + one-hot) |
| | Tests automatisés (`unittest`) |

## Ce qui a été fait, étape par étape

### 1. Gestion des valeurs manquantes

Trois colonnes avaient des valeurs manquantes (voir EDA) : `workclass` et
`occupation` (manquent presque toujours ensemble, ~5,7 % chacune) et
`native-country` (~1,8 %).

- `workclass` et `occupation` manquants → remplacés par `"Unemployed"`
- `native-country` manquant → remplacé par `"Unknown"`

Fait avec `sklearn.impute.SimpleImputer(strategy="constant", ...)`, intégré
dans un `ColumnTransformer` pour pouvoir être appris sur le train uniquement
et simplement appliqué (`.transform()`) sur validation/test, sans fuite de
données.

### 2. Regroupement des catégories rares

- `workclass` : les modalités `"Never-worked"` et `"Without-pay"` (quelques
  dizaines de lignes au total) sont fusionnées dans une catégorie `"Other"`.
- `native-country` : sur 41 modalités d'origine dont 91 % de
  `"United-States"`, on garde 3 catégories : `"United-States"`, `"Unknown"`
  (valeur manquante, gardée distincte — décision volontaire pour ne pas
  confondre "pays inconnu" et "pays étranger connu") et `"Other"` (tous les
  autres pays).

Règle fixe, non apprise sur les données : peut être appliquée à l'identique
sur train, validation et test sans risque de fuite.

### 3. Nouvelles features (capital-gain / capital-loss)

L'EDA avait montré que `capital-gain` a 91,7 % de zéros et un plafond de
saisie à 99 999 (244 lignes, toutes `>50K`). Trois features créées :

- `capital_gain_log` = `log1p(capital-gain)`
- `capital_loss_log` = `log1p(capital-loss)`
- `capital_gain_capped` = indicateur binaire (1 si `capital-gain >= 99999`)

Les colonnes brutes `capital-gain`/`capital-loss` sont ensuite supprimées du
jeu de features final, pour éviter la redondance avec leurs versions
transformées.

### 4. Encodage final

- Variables numériques (`age`, `education-num`, `hours-per-week`,
  `capital_gain_log`, `capital_loss_log`) → `StandardScaler`
- Indicateur binaire (`capital_gain_capped`) → laissé tel quel
- Variables catégorielles (`workclass`, `occupation`, `marital-status`,
  `relationship`, `race`, `native-country`) → `OneHotEncoder(handle_unknown="ignore")`

Résultat : **50 colonnes** au total (5 numériques + 1 binaire + 44 colonnes
one-hot), vérifiées à la main (8 + 15 + 7 + 6 + 5 + 3 modalités).

## Décisions prises (à mentionner/défendre à l'oral)

| Décision | Justification |
|---|---|
| `"Unemployed"` plutôt que `"Unknown"` pour `workclass`/`occupation` manquants | Hypothèse métier : l'absence conjointe de ces deux colonnes dans l'immense majorité des cas suggère une non-réponse liée à l'absence d'emploi. Hypothèse explicite, pas neutre. |
| `"Unknown"` gardé distinct de `"Other"` pour `native-country` | Permet de distinguer "non renseigné" de "étranger connu" dans l'explicabilité (SHAP) plus tard. |
| Suppression de `capital-gain`/`capital-loss` bruts | Fortement redondants avec leurs versions log-transformées ; évite la colinéarité pour les modèles linéaires. |
| `handle_unknown="ignore"` dans l'encodeur | Évite un crash si une catégorie rare du train est absente d'un sous-ensemble (ou l'inverse) ; comportement conservateur plutôt que bloquant. |

## Comment exécuter

Depuis la racine du dépôt (environnement avec `numpy`, `pandas`,
`scikit-learn` installés) :

```bash
python -m contributions.yanis.feature_engineering.src.build_features