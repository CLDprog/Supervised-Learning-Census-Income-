"""Training-only EDA tables, seven figures and a report-ready English summary.

Run after src.prepare_data: python -m src.explore_training_data
Validation and test CSVs are never opened by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prepare_data import CATEGORICAL, NUMERIC, TARGET, ROW_ID, load_partition

COLORS = ["#536878", "#B56953"]


def group_rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    values = frame[column].fillna("(missing)")
    return frame.assign(group=values, positive=frame[TARGET].eq(">50K")).groupby("group", dropna=False)["positive"].agg(n="size", positives="sum", observed_positive_rate="mean")


def save_figure(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_eda(data_dir: str | Path, output_dir: str | Path) -> dict:
    data_dir, out = Path(data_dir), Path(output_dir)
    frame = load_partition(data_dir, "train")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    positive = frame[TARGET].eq(">50K")
    missing = pd.DataFrame({"missing_count": frame.drop(columns=ROW_ID).isna().sum(), "missing_rate": frame.drop(columns=ROW_ID).isna().mean()})
    missing.to_csv(out / "missing_values.csv")
    numeric = frame[NUMERIC].agg(["min", "median", "mean", "max", "std"]).T
    numeric["zero_rate"] = frame[NUMERIC].eq(0).mean()
    numeric.to_csv(out / "numeric_summary.csv")
    frame.groupby(TARGET, observed=True)[NUMERIC].median().to_csv(out / "numeric_medians_by_class.csv")
    rate_tables = {column: group_rates(frame, column) for column in CATEGORICAL}
    pd.concat(rate_tables, names=["feature", "category"]).to_csv(out / "categorical_target_rates.csv")
    correlation = frame[NUMERIC].corr()
    correlation.to_csv(out / "numeric_correlations.csv")
    education_mapping = frame.groupby("education", observed=True)["education-num"].nunique()
    repeated = frame.duplicated(subset=[c for c in frame if c not in [ROW_ID, TARGET]], keep=False)
    summary = {
        "scope": "Training-only EDA. Holdout CSVs are not read. Observed rates are not model fairness metrics.",
        "n_rows": len(frame), "n_original_features": 14,
        "class_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        "positive_rate": float(positive.mean()),
        "missing": missing[missing.missing_count > 0].reset_index(names="feature").to_dict("records"),
        "education_mapping_is_one_to_one": bool(education_mapping.eq(1).all()),
        "capital_gain_zero_rate": float(frame["capital-gain"].eq(0).mean()),
        "capital_loss_zero_rate": float(frame["capital-loss"].eq(0).mean()),
        "capital_gain_99999_count": int(frame["capital-gain"].eq(99999).sum()),
        "repeated_original_profile_rows": int(repeated.sum()),
        "sex_observed_rates": rate_tables["sex"].reset_index().to_dict("records"),
        "race_observed_rates": rate_tables["race"].reset_index().to_dict("records"),
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "split_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    counts = frame[TARGET].value_counts().reindex(["<=50K", ">50K"])
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=COLORS)
    ax.bar_label(bars, labels=[f"{v:,} ({v/len(frame):.1%})" for v in counts], padding=5)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set(title="Income classes - training only", ylabel="People")
    save_figure(fig, out, "01_target_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    shown = missing[missing.missing_count > 0]
    ax.bar(shown.index, shown.missing_rate * 100, color=COLORS[0])
    ax.set(title="Missing feature values - training only", ylabel="Missing values (%)")
    save_figure(fig, out, "02_missing_values.png")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for column, ax in zip(NUMERIC, axes.ravel()):
        values = frame[column].dropna()
        if column.startswith("capital-"):
            values = np.log1p(values)
            ax.set_xlabel(f"log(1 + {column})")
        else:
            ax.set_xlabel(column)
        ax.hist(values, bins=40, color=COLORS[0], edgecolor="white", linewidth=0.3)
        ax.set_title(column)
        ax.set_ylabel("People")
    fig.suptitle("Numeric distributions - training only; capital amounts shown on log scale", fontsize=12)
    save_figure(fig, out, "03_numeric_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for column, ax in zip(["age", "education-num", "hours-per-week"], axes):
        groups = [frame.loc[frame[TARGET].eq(label), column].dropna() for label in ["<=50K", ">50K"]]
        ax.boxplot(groups, tick_labels=["<=50K", ">50K"], showfliers=False)
        ax.set_title(column)
    fig.suptitle("Numeric variables by income class - training only; outliers hidden in display")
    save_figure(fig, out, "04_numeric_by_class.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 11))
    for column, ax in zip(["workclass", "education", "marital-status", "occupation", "relationship", "native-country"], axes.ravel()):
        shown = rate_tables[column].sort_values("n", ascending=False).head(8).iloc[::-1]
        ax.barh([f"{k} (n={n:,})" for k, n in zip(shown.index, shown.n)], shown.observed_positive_rate, color=COLORS[0])
        ax.set(title=column, xlabel="Observed >50K proportion", xlim=(0, 1))
    fig.suptitle("Income rates for eight most frequent categories per variable - training only")
    save_figure(fig, out, "05_categorical_target_rates.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for column, ax in zip(["sex", "race"], axes):
        shown = rate_tables[column].sort_values("n", ascending=False).iloc[::-1]
        ax.barh([f"{k} (n={n:,})" for k, n in zip(shown.index, shown.n)], shown.observed_positive_rate, color=COLORS[1])
        ax.set(title=column, xlabel="Observed >50K proportion", xlim=(0, 1))
    fig.suptitle("Sensitive-group label prevalence - training only; not a model fairness audit")
    save_figure(fig, out, "06_sensitive_group_prevalence.png")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=45, ha="right")
    ax.set_yticks(range(len(NUMERIC)), NUMERIC)
    for i in range(len(NUMERIC)):
        for j in range(len(NUMERIC)):
            ax.text(j, i, f"{correlation.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(plot, ax=ax, label="Pearson correlation")
    ax.set_title("Numeric associations - training only; not causal evidence")
    save_figure(fig, out, "07_numeric_correlations.png")

    lines = ["# Hugo's contribution - training EDA and data splitting", "", "## Execution and scope", "",
             "Executed from Python scripts using the repository's Excel snapshot. The framing is in [validation_protocol_addendum.md](../../docs/validation_protocol_addendum.md). This contribution adapts Jérémy's original EDA and adds splitting and integrity checks. The group's original files are preserved.", "",
             f"Raw rows: **{manifest['raw_rows']:,}**. Exact repeats removed: **{manifest['duplicates_removed']}**. Clean rows: **{manifest['clean_rows']:,}**.", "",
             f"After exact deduplication, {manifest['profile_audit']['rows_in_repeated_profiles']} rows in {manifest['profile_audit']['repeated_profile_groups']} groups share identical original predictors; {manifest['profile_audit']['contradictory_label_groups']} groups have contradictory labels. They are retained and each group is kept within one partition. We therefore use 20 stratified grouped folds: 14 training, 3 validation, 3 test. This gives approximately 70/15/15; folds are fixed by seed 42, not model performance.", "",
             "| Partition | Rows | >50K count | >50K rate | Raw model predictors |", "|---|---:|---:|---:|---:|"]
    for name, stats in manifest["splits"].items():
        lines.append(f"| {name} | {stats['n_rows']:,} | {stats['positive_count']:,} | {stats['positive_rate']:.4%} | {stats['n_predictors']} |")
    lines += ["", "Checks passed: disjoint source rows and original predictor profiles, full coverage, aligned labels/sex and small stratification deviations. CSV SHA-256 hashes and runtime versions are recorded in the manifest; changed splits are not silently overwritten.", "",
              "## Training-only findings", "", f"Training contains **{len(frame):,}** people; **{positive.mean():.2%}** are in >50K. Accuracy alone would reward majority prediction.", "",
              "| Feature | Missing count | Missing rate |", "|---|---:|---:|"]
    for row in summary["missing"]:
        lines.append(f"| {row['feature']} | {row['missing_count']:,} | {row['missing_rate']:.2%} |")
    lines += ["", f"Capital gain is zero for **{summary['capital_gain_zero_rate']:.2%}** of training observations; capital loss is zero for **{summary['capital_loss_zero_rate']:.2%}**. Evaluate justified log/indicator features within the training pipeline; plotting log values does not modify saved inputs.", "",
              ("Education labels map one-to-one to education-num in training." if summary["education_mapping_is_one_to_one"] else "Education mapping is not one-to-one in this input; investigate before dropping a variable.") + " The initial protocol retains education-num. Exclude fnlwgt as a predictor under the chosen unweighted-sample protocol; keep sex separately for the planned fairness audit. This leaves 11 raw predictors, including race, before feature engineering.", "",
              "| Sex group | Training rows | Observed >50K rate |", "|---|---:|---:|"]
    for row in summary["sex_observed_rates"]:
        lines.append(f"| {row['group']} | {row['n']:,} | {row['observed_positive_rate']:.2%} |")
    lines += ["", "These are observed label prevalences, not model selection rates, recall gaps or evidence that a fitted model is fair. Other inputs can proxy sex. Rare categories have small sample sizes; preserve counts beside rates and fit any grouping on training only.", "",
              "## Limitations and next steps", "",
              "The original EDA inspected the full table before splitting; disclose this. These new charts and feature associations use training only. The held-out labels are used only for structural split checks at this stage. No fitted preprocessing, model comparison, Optuna tuning or model-based fairness result is claimed.", "",
              "Values such as capital-gain=99999 may reflect coding conventions; their exact interpretation requires source verification. Identical profiles do not establish person identity, and survey-derived historical results need not generalize to other populations or periods. Predictive associations are not causal explanations.", "",
              "Next implement a fitted preprocessing pipeline and 2-3 justified features; then compare models, address imbalance and tune on training CV. Calibrate/select thresholds and mitigation using validation, then evaluate frozen candidates on test. Transfer this section into the final LaTeX report with the contribution and AI-use disclosures.", "",
              "## Figures", ""]
    for name in ["01_target_distribution.png", "02_missing_values.png", "03_numeric_distributions.png", "04_numeric_by_class.png", "05_categorical_target_rates.png", "06_sensitive_group_prevalence.png", "07_numeric_correlations.png"]:
        lines.extend([f"![{name.replace('_', ' ')}]({name})", ""])
    (out / "eda_and_split_results.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    summary = run_eda(args.data_dir, args.output_dir)
    print(f"Training-only EDA completed: {summary['n_rows']:,} rows; seven figures and tables in {args.output_dir}")


if __name__ == "__main__":
    main()
