# Validation protocol addendum - Hugo's contribution

This complements the [existing framing](../../../../docs/problem_framing.md)
and [Jérémy's EDA](../../../../EDA.ipynb); neither file is edited or replaced.
Dataset, target and the exact-deduplication rule are inherited from that work.

## Implemented addition

After removing the existing 52 exact repeats, 48,790 rows remain. A new check
finds five pairs sharing all 14 original predictors but opposite income labels.
They remain in the data and each pair is kept inside one partition. Profiles do
not prove person identity; grouping prevents cross-partition copies of the
observed full profile without claiming independence of all people.

Use 20 stratified grouped folds with seed 42: first 14 training, next 3 validation,
last 3 test. Expected counts under the recorded versions are 34,156 / 7,317 /
7,317, close to 70/15/15, with about 23.94% positives in each subset. Fold choice
is fixed before any model scores are available. Verify disjoint row IDs and
original profiles, complete coverage, class balance and aligned audit attributes.
Save file hashes and prevent silent replacement of an established holdout.

Rebuild predictors and labels after cleaning. The initial implementation excludes
row identifiers, the target, fnlwgt, education and sex from predictors, retaining
education-num and sex separately for the audit. This follows the framing's
proposals; the group can discuss later model variants without editing its
existing work. Evaluation is unweighted, not a population-weighted estimate.

## Relationship to the planned project

Average Precision is the proposed primary selection metric, with precision,
recall, F1, ROC-AUC and confusion matrices. Use training-only CV for Optuna and
learned preprocessing; use validation for thresholds/calibration/mitigation,
then freeze the candidates before final test comparisons. Audit sex-group recall
and selection-rate gaps and discuss their fairness definitions and trade-offs.
These downstream choices remain a plan, not completed model experiments.

The original EDA explored the full table before splitting; disclose this.
The new EDA opens the training CSV only. Missing categories, education redundancy,
capital amounts and sensitive-group label prevalence are revisited in training.
Recorded label prevalence is not a model fairness audit; predictive associations
are not causal. The original exploratory findings are credited in the
contribution README rather than claimed as new work.

Implementation and documentation were prepared with AI assistance. Include
this accurately in the final contribution statement and AI-use disclosure.
