# Hugo's contribution - EDA and reproducible data splitting

This directory is an **additive contribution** to the group project. It builds
on [Jérémy's original EDA notebook](../../../EDA.ipynb) and the existing
[problem framing](../../../docs/problem_framing.md). All files already present
in the group repository remain unchanged; this README belongs only to Hugo's
contribution folder and does not replace the project's README.

## Provenance: reused and added work

| Reused from the existing work | Added in this contribution |
|---|---|
| Adult / Census Income snapshot and target | Reproducible training/validation/test partitions |
| EDA findings and 52-exact-repeat cleaning rule | Grouping of five contradictory-label profile pairs |
| Existing ideas for variables, metrics and audit | Split integrity checks, fingerprints and aligned audit inputs |
| Original descriptive exploration | An adaptation into executable Python scripts and training-only charts |

The training-only EDA script revisits existing descriptive questions in a
reusable format; it is not presented as an entirely original exploration.
Preparation and implementation were assisted by AI and should be reviewed by
the group. Model fitting and requirements 2-10 remain future work.

## Run from the group repository

```bash
cd contributions/hugo/eda_and_data_splitting
python -m pip install -r requirements-eda.txt
python -m src.prepare_data
python -m src.explore_training_data
python -m unittest discover -s tests -v
```

The default input is the group's existing `census_income.xlsx` at the repository
root. It is read only. The `.py` files are the implementation; the accompanying
Colab notebook only launches them. No root-level README, notebook, dependency
file or framing document is modified.

## Files and outputs

- `src/prepare_data.py`: normalization, exact deduplication, grouped stratified
  splitting and aligned raw predictors/labels/sex.
- `src/explore_training_data.py`: training-only tables and seven charts.
- `tests/test_data_protocol.py`: eight protocol checks, including no holdout reads
  during EDA and protection against changed partitions.
- [Validation protocol addendum](docs/validation_protocol_addendum.md): new
  implementation choices, recorded separately from the group's framing.
- [Executed EDA/split results](reports/eda/eda_and_split_results.md): report-ready
  findings, with provenance and limitations.

Generated individual-level splits are stored only in this contribution's
`data/processed/`, ignored by the local `.gitignore`. They are not included in
the upload package. Runtime versions and file hashes are recorded.

## Reuse in future modelling

When working from this directory:

```python
from src.prepare_data import load_partition, model_inputs
train = load_partition("data/processed", "train")
X_train, y_train, sex_train = model_inputs(train)
```

This supplies 11 raw predictors. Fit imputation, encoding and other learned
transformations in training/CV pipelines. Any move into shared project modules
can be agreed with the group later; the current submission only adds this folder.
