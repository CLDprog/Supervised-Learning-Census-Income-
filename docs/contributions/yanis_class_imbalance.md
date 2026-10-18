# Supervised Learning - Census Income - Partie 3 : Déséquilibre des classes

> **Document historique de contribution.** Il décrit le travail tel qu'il a été livré, avec les chemins et commandes de l'époque (dossier `contributions/`). Le code se trouve désormais dans `census_income/` ; voir le README racine pour la structure et les commandes actuelles.

**Contribution de Yanis.** Requirement technique n°3 : rapporter des
métriques pertinentes, justifier le remède au déséquilibre et montrer comment
il est appliqué dans le code. Construit sur le split de
[Hugo](hugo_eda_and_data_splitting.md) et sur le feature engineering de la
[partie 2](yanis_feature_engineering.md).

## Provenance

| Repris de l'existant | Ajouté ici |
|---|---|
| Split train/validation/test (Hugo) | `evaluate_model()` : métriques communes à tous les modèles du projet |
| Chaîne de features de la partie 2 (50 colonnes) | `find_best_threshold()` : choix du seuil qui maximise le F1 |
| Métriques retenues dans `docs/problem_framing.md` | Comparaison modèle naïf / logistique avec et sans pondération |
| | Validation croisée en 5 découpages dans un `Pipeline` sans fuite |
| | Démonstration de l'effet du seuil, puis choix du remède |
| | 4 tests automatisés |

## Le problème

76 % des personnes sont `<=50K` et 24 % `>50K`. Un modèle qui répond toujours
`<=50K` obtient 76 % d'accuracy sans trouver une seule personne `>50K`.
L'accuracy n'est donc pas utilisée pour comparer les modèles. Métrique
principale : **Average Precision** (AP), accompagnée de ROC-AUC, precision,
recall, F1 et de la matrice de confusion, calculées sur la classe `>50K`.

## Résultats sur la validation (7 317 personnes, dont 1 752 `>50K`)

| Modèle | AP | ROC-AUC | Precision | Recall | F1 | Accuracy | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| Naïf (toujours `<=50K`) | 0,239 | 0,500 | 0 | 0 | 0 | 0,761 | 0 | 1 752 | 0 |
| Logistique, seuil 0,50 | 0,742 | 0,900 | 0,719 | 0,586 | 0,646 | 0,846 | 1 026 | 726 | 400 |
| Logistique `class_weight="balanced"`, seuil 0,50 | 0,738 | 0,900 | 0,557 | 0,853 | 0,674 | 0,802 | 1 494 | 258 | 1 190 |
| **Logistique, seuil 0,282 (retenu)** | **0,742** | **0,900** | **0,590** | **0,812** | **0,683** | **0,820** | **1 422** | **330** | **989** |

## Validation croisée sur le train (5 découpages, moyenne ± écart-type)

| Modèle | AP | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistique sans pondération | 0,737 ± 0,005 | 0,898 ± 0,003 | 0,713 ± 0,010 | 0,582 ± 0,012 | 0,641 ± 0,005 |
| Logistique `balanced` | 0,734 ± 0,005 | 0,898 ± 0,003 | 0,557 ± 0,008 | 0,842 ± 0,010 | 0,670 ± 0,007 |

Toute la chaîne (imputation, regroupement, features, encodage, modèle) est
rangée dans un `Pipeline` : à chaque découpage, elle est réapprise sur la
partie d'entraînement uniquement, sans fuite de données.

**Lecture** : la différence d'AP (0,003) est plus petite que l'écart-type
(0,005) et le ROC-AUC est identique. La pondération ne rend donc pas le modèle
meilleur pour distinguer les deux classes. En revanche, les écarts de
precision et de recall sont 15 à 20 fois plus grands que l'écart-type : le
compromis qu'elle introduit est réel.

## L'effet du seuil

Logistique sans pondération, validation :

| Seuil | Precision | Recall | F1 |
|---|---|---|---|
| 0,50 | 0,719 | 0,586 | 0,646 |
| 0,40 | 0,662 | 0,691 | 0,676 |
| 0,30 | 0,598 | 0,787 | 0,679 |
| 0,25 | 0,563 | 0,843 | 0,675 |
| 0,20 | 0,524 | 0,890 | 0,659 |

Au seuil 0,25, on retrouve presque exactement le modèle `balanced`
(0,557 / 0,853 / 0,674). La pondération des classes revient donc
essentiellement à abaisser le seuil de décision : elle gonfle toutes les
probabilités sans changer leur ordre.

## Décision

**Pas de pondération des classes. Seuil de décision fixé à 0,282**, choisi sur
la validation pour maximiser le F1 de la classe `>50K` (`find_best_threshold`).

| Justification | |
|---|---|
| Classement inchangé | AP et ROC-AUC identiques à ceux du modèle pondéré |
| Meilleur compromis | F1 de 0,683, contre 0,674 pour `balanced` |
| Recall | De 59 % à 81 % : 396 personnes `>50K` trouvées en plus |
| Probabilités fidèles | Contrairement à `balanced`, qui les gonfle (utile pour la partie 9, calibration) |
| Conforme au cadrage | `docs/problem_framing.md` prévoit de choisir le seuil sur la validation par maximum de F1 |

**Contrepartie assumée** : la precision passe de 72 % à 59 % (989 fausses
alertes au lieu de 400). Aucun coût monétaire n'étant attribué aux erreurs
dans le cadrage, le F1 sert d'arbitre.

## Comment exécuter

Depuis la racine du dépôt :

```bash
python -m contributions.yanis.class_imbalance.src.compare_imbalance
```

## Tests

```bash
python -m unittest discover -s contributions/yanis/class_imbalance/tests -v
```

4 tests, tous passants :
- `evaluate_model` retrouve des métriques calculées à la main ;
- `find_best_threshold` trouve le bon seuil sur un exemple évident ;
- le modèle naïf a plus de 75 % d'accuracy mais un recall nul ;
- le seuil choisi donne un F1 au moins égal à celui du seuil 0,5.

## Pour la partie 4

Tous les modèles doivent être évalués avec les mêmes outils :

```python
from contributions.yanis.class_imbalance.src.evaluate import evaluate_model, find_best_threshold

y_proba = model.predict_proba(X_val)[:, 1]
threshold = find_best_threshold(y_val, y_proba)      # sur la validation uniquement
results = evaluate_model(y_val, y_proba, threshold)
```

## Limites

- Le seuil est choisi et mesuré sur la même validation : les scores au seuil
  0,282 sont légèrement optimistes. Le score définitif sera mesuré une seule
  fois sur le test, avec ce seuil figé.
- La chaîne de features est écrite deux fois dans `compare_imbalance.py`
  (`prepare_features()` et `build_pipeline()`). Seul le `Pipeline` sera
  conservé lors de la réorganisation du dépôt (partie 5).
- Les résultats sont affichés dans la console ; les tableaux et graphiques
  devront être exportés pour le rapport.
- Préparation assistée par IA (Claude), à mentionner dans le rapport.