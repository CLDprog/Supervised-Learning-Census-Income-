"""Load Adult, deduplicate, split and preserve aligned fairness attributes.

Run: python -m src.prepare_data (reads the group's root Excel snapshot)
No imputation, encoding, scaling or resampling is fitted here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedGroupKFold

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
TARGET = "class"
ROW_ID = "row_id"
EXCLUDED_PREDICTORS = [ROW_ID, TARGET, "fnlwgt", "education", "sex"]
LABELS = {"<=50K": 0, ">50K": 1}


def load_raw(input_path: str | Path | None = None) -> pd.DataFrame:
    """Read the repository snapshot, or explicitly retrieve OpenML adult v2."""
    if input_path is None:
        return fetch_openml(name="adult", version=2, as_frame=True).frame.copy()
    path = Path(input_path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx or .csv; omit it to use OpenML.")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize deterministic source conventions, without estimating parameters."""
    if set(raw.columns) != set(COLUMNS):
        raise ValueError(f"Unexpected columns: expected {COLUMNS}, received {list(raw.columns)}")
    frame = raw[COLUMNS].copy().reset_index(drop=True)
    for column in CATEGORICAL + [TARGET]:
        values = frame[column].astype("string").str.strip()
        values = values.mask(values.isin(["?", ""]))
        if column == TARGET:
            values = values.str.rstrip(".")
        frame[column] = values.astype(object).where(values.notna(), np.nan)
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != set(LABELS):
        raise ValueError("Target must contain both <=50K and >50K, with no missing labels.")
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        present = frame[column].dropna()
        if not np.isfinite(present).all():
            raise ValueError(f"Non-finite values in {column}")
        if present.eq(present.round()).all():
            frame[column] = frame[column].astype("Int64" if frame[column].isna().any() else "int64")
    if frame["sex"].isna().any() or not set(frame["sex"]).issubset({"Female", "Male"}):
        raise ValueError("Expected non-missing Female/Male groups for the planned audit.")
    return frame


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep the first exact record; row_id refers to its original positional row."""
    frame = normalize(raw)
    clean = frame.loc[~frame.duplicated(keep="first")].copy()
    clean.insert(0, ROW_ID, clean.index.to_numpy())
    return clean.reset_index(drop=True)


def model_inputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return raw X, binary y and sex with the same source-row index.

    Missing features are deliberately retained for the future fitted pipeline.
    """
    aligned = frame.set_index(ROW_ID, verify_integrity=True)
    X = aligned.drop(columns=[c for c in EXCLUDED_PREDICTORS if c != ROW_ID]).copy()
    y = aligned[TARGET].map(LABELS).astype("int64").rename("income_gt_50k")
    sensitive = aligned["sex"].copy().rename("sex")
    if not (X.index.equals(y.index) and y.index.equals(sensitive.index)):
        raise ValueError("Predictors, target and sensitive attribute are not aligned.")
    return X, y, sensitive


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def items(self):
        return (("train", self.train), ("validation", self.validation), ("test", self.test))


def validate_splits(clean: pd.DataFrame, splits: DatasetSplits) -> dict:
    """Fail on overlapping IDs or exact predictor profiles and report balance.

    Predictor fingerprints ignore the label: identical profiles must not occur
    in different subsets even if their labels disagree. This conservative check
    complements deduplication, without inferring identities from profiles.
    """
    ids, fingerprints, summaries = {}, {}, {}
    all_ids = set(clean[ROW_ID])
    for name, frame in splits.items():
        ids[name] = set(frame[ROW_ID])
        if len(ids[name]) != len(frame) or not ids[name].issubset(all_ids):
            raise ValueError(f"Invalid row identifiers in {name}")
        X, y, sensitive = model_inputs(frame)
        # Check the full original profile, including the survey weight.
        profile = frame.drop(columns=[ROW_ID, TARGET])
        fingerprints[name] = set(pd.util.hash_pandas_object(profile, index=False).tolist())
        if set(y.unique()) != {0, 1}:
            raise ValueError(f"Both classes must be represented in {name}")
        summaries[name] = {
            "n_rows": len(frame), "positive_count": int(y.sum()),
            "negative_count": int((y == 0).sum()), "positive_rate": float(y.mean()),
            "sex_counts": {str(k): int(v) for k, v in sensitive.value_counts().items()},
            "n_predictors": X.shape[1],
        }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if ids[left] & ids[right]:
            raise ValueError(f"Overlapping row IDs: {left}/{right}")
        if fingerprints[left] & fingerprints[right]:
            raise ValueError(f"Identical original predictor profiles across {left}/{right}; use a grouped split.")
    if set.union(*ids.values()) != all_ids or sum(len(v) for v in ids.values()) != len(clean):
        raise ValueError("Splits do not cover exactly the cleaned data.")
    # Group constraints permit small deviations from ideal class proportions.
    overall_rate = clean[TARGET].eq(">50K").mean()
    for name, stats in summaries.items():
        if abs(stats["positive_rate"] - overall_rate) > 0.005:
            raise ValueError(f"Unexpected class-balance drift in {name}")
        expected_fraction = 0.70 if name == "train" else 0.15
        if abs(stats["n_rows"] / len(clean) - expected_fraction) > 0.005:
            raise ValueError(f"Unexpected split-size drift in {name}")
    return summaries


def make_splits(clean: pd.DataFrame, random_state: int = 42) -> DatasetSplits:
    # Twenty grouped, stratified folds give nominal 14/3/3 = 70/15/15.
    # This retains contradictory-label profiles together instead of discarding
    # them or relying on random row splitting. No model scores select the folds.
    profiles = clean.drop(columns=[ROW_ID, TARGET])
    groups = pd.util.hash_pandas_object(profiles, index=False).to_numpy()
    labels = clean[TARGET].map(LABELS).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=random_state)
    folds = [indices for _, indices in splitter.split(clean, labels, groups)]
    train = clean.iloc[np.concatenate(folds[:14])]
    validation = clean.iloc[np.concatenate(folds[14:17])]
    test = clean.iloc[np.concatenate(folds[17:])]
    # Stable source order makes exported tables easy to compare and audit.
    splits = DatasetSplits(*(part.sort_values(ROW_ID).reset_index(drop=True) for part in (train, validation, test)))
    validate_splits(clean, splits)
    return splits


def dataset_digest(clean: pd.DataFrame) -> str:
    return hashlib.sha256(clean.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def load_partition(data_dir: str | Path, name: str) -> pd.DataFrame:
    """Read one partition only, verifying the established file fingerprint."""
    if name not in {"train", "validation", "test"}:
        raise ValueError("Unknown partition")
    directory = Path(data_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{name}.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["file_sha256"][path.name]:
        raise ValueError(f"Partition fingerprint mismatch: {path}")
    frame = pd.read_csv(path)
    if len(frame) != manifest["splits"][name]["n_rows"]:
        raise ValueError(f"Unexpected row count in {name}")
    model_inputs(frame)
    return frame


def export_splits(raw: pd.DataFrame, output_dir: str | Path, source: str, random_state: int = 42) -> dict:
    """Persist deterministic splits plus hashes; never overwrite a changed split."""
    clean = clean_data(raw)
    splits = make_splits(clean, random_state)
    summary = validate_splits(clean, splits)
    profile_hashes = pd.util.hash_pandas_object(clean.drop(columns=[ROW_ID, TARGET]), index=False)
    audit = pd.DataFrame({"profile": profile_hashes.to_numpy(), "label": clean[TARGET].to_numpy()}).groupby("profile")["label"].agg(size="size", label_count="nunique")
    profile_audit = {
        "repeated_profile_groups": int(audit["size"].gt(1).sum()),
        "rows_in_repeated_profiles": int(audit.loc[audit["size"].gt(1), "size"].sum()),
        "contradictory_label_groups": int(audit["label_count"].gt(1).sum()),
    }
    out = Path(output_dir)
    tables = {name: frame.to_csv(index=False, lineterminator="\n").encode() for name, frame in splits.items()}
    manifest = {
        "dataset": "OpenML adult version 2", "source": source,
        "raw_rows": len(raw), "clean_rows": len(clean), "duplicates_removed": len(raw) - len(clean),
        "clean_dataset_sha256": dataset_digest(clean), "random_state": random_state,
        "profile_audit": profile_audit,
        "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "split_method": "StratifiedGroupKFold(20, shuffle=True): first 14 folds train, next 3 validation, last 3 test",
        "group_definition": "Identical original 14-variable profiles, including fnlwgt; target excluded",
        "balance_tolerance": "At most 0.5 percentage point deviation in size fraction or target prevalence",
        "target_mapping": LABELS, "excluded_predictors": EXCLUDED_PREDICTORS,
        "splits": summary,
        "file_sha256": {f"{name}.csv": hashlib.sha256(data).hexdigest() for name, data in tables.items()},
        "checks": {"disjoint_source_rows": True, "disjoint_original_predictor_profiles": True,
                   "complete_coverage": True, "stratified_balance": True, "aligned_sensitive_attribute": True},
        "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "scope": "No learned preprocessing or modelling; test reserved for final frozen comparisons.",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("clean_dataset_sha256") != manifest["clean_dataset_sha256"] or previous.get("file_sha256") != manifest["file_sha256"]:
            raise ValueError("Existing split differs. Use a new directory; do not replace the established test set.")
    # Preflight every file before any writes, including incomplete output directories.
    for name, data in tables.items():
        path = out / f"{name}.csv"
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite a changed split: {path}")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        path = out / f"{name}.csv"
        if not path.exists():
            path.write_bytes(data)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif "profile_audit" not in previous:
        # Metadata-only addition for an existing byte-identical partition set.
        previous["profile_audit"] = profile_audit
        manifest_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    return json.loads(manifest_path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--input", type=Path, help="Repository .xlsx/.csv snapshot")
    sources.add_argument("--openml", action="store_true", help="Download OpenML adult v2 explicitly")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    path = None if args.openml else (args.input or Path(__file__).resolve().parents[4] / "census_income.xlsx")
    source = "fetch_openml(name='adult', version=2)" if path is None else str(path)
    result = export_splits(load_raw(path), args.output_dir, source, args.random_state)
    print(f"Raw: {result['raw_rows']:,}; removed: {result['duplicates_removed']}; clean: {result['clean_rows']:,}")
    for name, stats in result["splits"].items():
        print(f"{name}: {stats['n_rows']:,} rows, {stats['positive_rate']:.4%} positive, {stats['n_predictors']} raw predictors")
    print(f"Checks passed; splits and manifest in {args.output_dir}")


if __name__ == "__main__":
    main()
