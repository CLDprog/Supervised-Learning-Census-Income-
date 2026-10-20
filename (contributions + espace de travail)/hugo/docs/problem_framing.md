# Problem framing - Adult / Census Income

**Status:** protocol written before modelling, now implemented in `census_income/`. Counts below come from the [EDA](../notebooks/01_eda.ipynb). Decisions taken during the project are recorded in section 7.

## 1. Objective and use

Build a supervised-learning proof of concept for a socio-economic data analyst: estimate whether a person's observed annual income exceeds $50,000. Compare at least three models for predictive performance, probability reliability and fairness. This is classification of an observed income category, not forecasting future income or estimating a salary. Suitability for individual decisions is not established.

Encode `1 = >50K` and `0 = <=50K`. A false negative misses a higher-income person; a false positive incorrectly assigns that category. No monetary error costs are assumed; report the precision/recall trade-off.

## 2. Data and existing decisions

Use `fetch_openml(name="adult", version=2, as_frame=True)`. Each row describes one person. The raw table contains **48,842 rows**, **14 descriptive variables** and target `class`.

The EDA removes **52 exact repeated rows**, leaving **48,790**: **11,681 positives (23.94%)** and **37,109 negatives (76.06%)**. Retain this rule before splitting, acknowledging that identical records without identifiers do not prove repeated individuals. Rebuild `X` and `y` from cleaned `df`; their initial definitions precede deduplication.

Handle missing categories in `workclass`, `occupation` and `native-country` inside preprocessing. Follow the EDA proposal to exclude survey weight `fnlwgt` from predictors and keep `education-num` instead of the redundant `education`. Results describe the unweighted sample, not population-weighted estimates. Evaluate capital transformations, rare-category handling and at least two to three justified engineered features using training data.

## 3. Metrics and imbalance

Use **Average Precision (AP)** as the primary selection metric, computed from class-1 scores with `average_precision_score` / `scoring="average_precision"`. It summarizes precision/recall across thresholds and differs from trapezoidal PR-AUC.

Report AP, ROC-AUC, precision, recall, F1 and confusion matrices consistently for all three models. Always predicting `<=50K` gives **76.06% accuracy on the cleaned full sample**, with zero positive recall; compute the dummy baseline's actual subset scores.

Compare unweighted training with class weighting where supported, selecting the remedy through training cross-validation. Justify and implement the selected approach; never rebalance validation or test data. *Outcome: no class weighting; a decision threshold per model, chosen on validation (see section 7).*

## 4. Validation protocol

Use a **70% training / 15% validation / 15% test** split, stratified on the target, with `random_state=42`. Random splitting assumes observations can reasonably be treated as independent; absent dates or identifiers do not prove this. Check overlap and class proportions.

- **Training:** compare models and tune with Optuna and stratified cross-validation (implemented as grouped stratified 5-fold CV, so that identical profiles never cross a fold boundary). Fit all learned preprocessing and any resampling inside each fold's training pipeline.
- **Validation:** select each model's threshold by maximum positive-class F1 and show the precision/recall trade-off. Fit calibration if needed and choose mitigation settings without accessing test results.
- **Test:** freeze candidate versions and settings, then evaluate final comparisons, including fairness before/after mitigation. Do not revise choices using test scores.

Disclose that the EDA already examined the full table and target associations. The holdout is reserved from subsequent model fitting and selection, not claimed to have been unseen during exploration.

## 5. Fairness, calibration and limitations

Use `sex` for the primary fairness audit: preserve it alongside each split but exclude it from baseline predictors. Variables such as `relationship` can remain proxies.

Report group sizes, precision, recall and selection rates. Measure absolute recall gaps (**equal opportunity**) and selection-rate gaps (**demographic parity**); these criteria can conflict when group label prevalences differ. Choose one mitigation using training/validation, then compare frozen versions on the same test set for performance and fairness. Attribute exclusion alone does not establish fairness.

For the final model, report a reliability curve and Brier score; use global and local SHAP explanations. Historical sample results may not generalize to other populations or periods, and predictive associations do not establish causality.

## 6. Delivery

Implement the protocol in Python scripts and a pipeline, keeping notebooks supplementary. Track runs, parameters, metrics and artifacts. Integrate explanations and results into the LaTeX report with the required contribution statement and AI-use disclosure.

## 7. Decisions taken during the project

| Topic | Decision | Where |
|---|---|---|
| Imbalance remedy | Class weighting (`balanced`) leaves AP and ROC-AUC unchanged in CV and only shifts the operating point while inflating probabilities. **No class weighting**; each model gets its own threshold, chosen on validation by maximum positive-class F1. | `census_income/imbalance.py` |
| Cross-validation | Grouped stratified 5-fold CV on the training set (`StratifiedGroupKFold`), grouping identical original profiles. | `census_income/compare_models.py` |
| Model selection | Highest mean CV AP after Optuna tuning. | `census_income/tuning.py` |
| Fairness mitigation | Post-processing: one threshold per sex, chosen on validation so that both groups reach the overall recall (equal opportunity). | `census_income/fairness.py` |
| Test set | Opened once, by `census_income/fairness.py`, with the frozen final model and threshold. | `census_income/fairness.py` |

**Basis:** course *Project Instructions* and existing EDA. Drafted with AI assistance after reviewing both.

