# Guide de soutenance

Format : **7 minutes de présentation + 3 minutes de questions**. Chaque membre parle, et chacun doit pouvoir défendre **n'importe quelle partie** : 10 % de la note est individuelle.

## 1. Les chiffres à connaître par cœur

| Quoi | Valeur |
|---|---|
| Données | 48 842 personnes, 52 doublons retirés, il en reste 48 790 |
| Classe `>50K` | 23,9 % : un modèle qui répond toujours `<=50K` a 76 % d'accuracy et un rappel nul |
| Découpage | 70 / 15 / 15, soit 34 156 / 7 317 / 7 317, stratifié, graine 42 |
| Features | 11 prédicteurs bruts, puis 50 colonnes après encodage |
| Régression logistique | AP en validation croisée 0,737 |
| Random Forest | AP 0,804, réglé 0,809 |
| Gradient Boosting (retenu) | AP 0,824, réglé **0,827** |
| Seuil final | **0,404**, choisi sur la validation pour le meilleur F1 |
| **Test** (ouvert une seule fois) | **AP 0,833, ROC-AUC 0,931, F1 0,734**, précision 0,720, rappel 0,749 |
| Calibration | ECE 0,009, Brier 0,087 (contre 0,182 pour une référence constante) |
| Équité | Écart de rappel femmes/hommes : **12,1 points → 0,8 point**, F1 0,734 → 0,736 |
| SHAP (top 5) | statut marital, âge, plus-values, niveau d'études, heures travaillées |

## 2. Plan de la présentation (7 minutes)

| # | Diapositive | Durée | Message clé |
|---|---|---|---|
| 1 | Problème et cas d'usage | 0:45 | Prédire `>50K` (24 %) pour un analyste. L'accuracy est un piège : 76 % sans rien faire. Métrique principale : AP |
| 2 | Protocole | 0:45 | 70/15/15. Train pour apprendre et régler, validation pour le seuil, test **une seule fois**. Tout dans un Pipeline : pas de fuite |
| 3 | Données et features | 1:00 | Valeurs manquantes → `Unemployed` / `Unknown` (justifié : 9 % de `>50K`). Catégories rares regroupées. Trois features sur les plus-values. 50 colonnes |
| 4 | Déséquilibre | 1:00 | `class_weight` = même AP et même ROC-AUC, il ne fait que déplacer le seuil. On garde un entraînement normal et un seuil choisi sur la validation |
| 5 | Modèles et Optuna | 1:15 | Logistique 0,737 → Random Forest 0,804 → Gradient Boosting 0,824. Les arbres captent les interactions. Optuna en validation croisée groupée, 85 essais suivis dans MLflow |
| 6 | SHAP et calibration | 1:00 | Statut marital, âge, plus-values : cohérent avec l'intuition. Proxy du sexe via `relationship`. Probabilités fiables (ECE 0,009) |
| 7 | Équité et résultat final | 1:15 | Test : AP 0,833, F1 0,734. Un seuil par sexe : écart de rappel 12 → 0,8 point sans perte. Limite : utilise le sexe à la décision. Démo de l'API |

Figures à montrer : `reports/model_comparison/01_progression_modeles.png`, `reports/explainability/02_importance_par_variable.png`, `reports/calibration/01_courbe_fiabilite.png`, `reports/fairness/01_equite_avant_apres.png`, et la page `/docs` de l'API.

## 3. Questions probables et réponses

**Pourquoi l'Average Precision plutôt que l'accuracy ?**
Avec 76 % de `<=50K`, un modèle inutile a déjà 76 % d'accuracy. L'AP ne regarde que la classe qu'on cherche : elle résume précision et rappel sur tous les seuils, et vaut 0,24 pour le modèle naïf.

**Pourquoi pas `class_weight="balanced"` ?**
En validation croisée, il donne la même AP (0,734 contre 0,737, écart plus petit que l'écart-type) et le même ROC-AUC. Il ne change pas l'ordre des personnes, il gonfle seulement les probabilités, ce qui revient à baisser le seuil. On obtient le même effet avec un seuil plus bas, sans abîmer les probabilités.

**Choisir le seuil sur la validation, ce n'est pas de la triche ?**
C'est prévu pour ça : la validation sert aux choix, le test au score final. Le score de validation au seuil choisi est légèrement optimiste, donc on ne le présente pas comme résultat final. Sur le test, on retrouve le même niveau (F1 0,734 contre 0,736) : le choix n'a pas créé d'optimisme.

**Comment évitez-vous la fuite de données ?**
- Toute la préparation (imputation, encodage, standardisation) est dans un `Pipeline` : elle est réapprise sur la seule partie d'entraînement de chaque découpage.
- Le test n'est ouvert que par `fairness.py`, à la fin.
- Les doublons sont retirés avant le découpage, et les profils identiques restent ensemble.

**Pourquoi une validation croisée « groupée » ?**
Cinq paires de personnes ont exactement le même profil. Si l'une est dans l'entraînement et l'autre dans l'évaluation, le modèle « reconnaît » la réponse. Le groupement les garde dans le même découpage.

**Pourquoi le Gradient Boosting ?**
C'est la meilleure AP en validation croisée (0,827), avec deux fois moins de surapprentissage que le Random Forest (écart train/validation croisée de 0,049 contre 0,107). Ses arbres successifs corrigent les erreurs des précédents.

**Votre score Optuna n'est-il pas optimiste ?**
Optuna choisit les réglages sur la validation croisée du train. On mesure ensuite les modèles réglés sur la validation, qu'Optuna n'a jamais vue : 0,833. Le premier essai reprend les réglages de départ, donc le réglage ne peut pas faire pire en validation croisée.

**Qu'est-ce qu'une valeur SHAP ?**
C'est la contribution de chaque variable à la prédiction d'une personne, par rapport à la prédiction moyenne, en log-odds. La somme des contributions et de la valeur de base donne la prédiction du modèle. Une valeur positive pousse vers `>50K`.

**Les variables importantes ont-elles un sens ?**
Oui : âge, diplôme, profession et heures travaillées suivent l'intuition. Les petites plus-values font **baisser** la probabilité, comme dans l'EDA (17,9 % de `>50K`). Point de vigilance : `relationship` (`Husband` / `Wife`) est un proxy du sexe.

**Le modèle est-il sur-confiant ?**
Non. L'ECE est de 0,009 (moins d'un point d'écart moyen), et les écarts par tranche restent sous 5 points et changent de signe. Une recalibration isotonique n'apporte rien. On peut utiliser les probabilités telles quelles, sur une population comparable.

**Pourquoi la référence constante a-t-elle un ECE de 0 ?**
Prédire 24 % pour tout le monde est parfaitement calibré en moyenne, mais ne distingue personne. C'est pour ça qu'on regarde aussi le Brier score : 0,182 pour la référence, contre 0,087 pour notre modèle.

**Quelle définition d'équité, et pourquoi ?**
L'égalité des chances : une personne réellement `>50K` doit avoir la même chance d'être reconnue, qu'elle soit une femme ou un homme. La parité démographique, elle, exigerait autant de prédictions `>50K` chez les femmes, alors que leur taux réel est de 10,7 % contre 30,8 % : ce serait contraire aux données.

**Retirer `sex` des variables ne suffit pas ?**
Non. Avant correction, l'écart de rappel est de 12 points, car des proxys comme `relationship` portent l'information.

**Limite de votre correction ?**
Elle utilise le sexe au moment de la décision (deux seuils), ce qui peut être juridiquement interdit selon le contexte. L'alternative est une méthode appliquée pendant l'entraînement (`ExponentiatedGradient` de fairlearn). La précision chez les femmes baisse (0,728 → 0,643) : plus de fausses alertes. L'origine (`race`) n'a pas été traitée.

**Pourquoi garder les plus-values si elles sont presque toujours à 0 ?**
L'information est dans les rares valeurs non nulles : au-delà de 5 000 $, plus de 80 % des personnes sont `>50K`. Limite assumée : le revenu cible inclut probablement les plus-values.

**Pourquoi `Unemployed` pour les valeurs manquantes ?**
`workclass` et `occupation` manquent ensemble, et ces personnes n'ont que 9 % de `>50K`, contre 24 % en moyenne. C'est une hypothèse métier, explicitement assumée.

**Pourquoi exclure `fnlwgt` et `education` ?**
`fnlwgt` est un poids de sondage, pas une caractéristique de la personne. `education` est redondante avec `education-num` (un numéro par diplôme) ; on garde la version ordinale.

**Comment reproduire vos résultats ?**
`pip install -r requirements.txt` (versions fixées), puis les commandes du README, dans l'ordre. Graine 42 partout, et les partitions sont vérifiées par empreinte SHA-256. 30 tests automatisés.

**Comment marche l'API ?**
`uvicorn census_income.api:app`, puis `POST /predict` avec les 11 informations d'une personne. Elle renvoie la probabilité, le seuil et la prédiction. Les entrées sont validées, et les valeurs manquantes sont acceptées.

**Avez-vous utilisé une IA ?**
Oui, Claude (Anthropic), pour les explications, l'écriture et la relecture du code et de la documentation. C'est déclaré dans le rapport et le README. Les choix méthodologiques ont été discutés et validés par le groupe.

## 4. Démo de l'API (30 secondes)

1. `python -m census_income.train_final` (si `models/` est vide), puis `uvicorn census_income.api:app`.
2. Ouvrir http://127.0.0.1:8000/docs, puis `POST /predict`, puis « Try it out » avec l'exemple déjà rempli.
3. Montrer la réponse : probabilité, seuil 0,404, prédiction.
