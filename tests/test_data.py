"""Protocol tests: grouping, alignment, reproducibility and frozen holdouts."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from census_income.data import (COLUMNS, CATEGORICAL, NUMERIC, ROW_ID, clean_data,
                                export_splits, load_partition, make_splits, model_inputs)


def fixture():
    # Enough observations per fold to meet the Adult protocol's 0.5 pp bound.
    n = 2000
    frame = pd.DataFrame({column: ["known"] * n for column in CATEGORICAL})
    for column in NUMERIC:
        frame[column] = np.arange(n) + 1
    frame["sex"] = np.where(np.arange(n) % 2, "Male", "Female")
    frame["class"] = np.where(np.arange(n) % 4 == 0, ">50K", "<=50K")
    frame.loc[0, "occupation"] = np.nan
    return frame[COLUMNS]


class ProtocolTests(unittest.TestCase):
    def test_source_normalization_and_exact_deduplication(self):
        raw = fixture()
        raw.loc[1, "class"] = " <=50K. "
        raw.loc[1, "workclass"] = " ? "
        raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
        clean = clean_data(raw)
        self.assertEqual(len(clean), 2000)
        self.assertEqual(clean.loc[1, "class"], "<=50K")
        self.assertTrue(pd.isna(clean.loc[1, "workclass"]))
        self.assertEqual(clean[ROW_ID].nunique(), 2000)

    def test_contradictory_label_profiles_stay_together(self):
        raw = fixture()
        conflicting = raw.iloc[[0]].copy()
        conflicting["class"] = "<=50K"
        raw = pd.concat([raw, conflicting], ignore_index=True)
        clean = clean_data(raw)
        self.assertEqual(len(clean), 2001)  # Keep the contradictory observation.
        splits = make_splits(clean)
        locations = {row_id: name for name, frame in splits.items() for row_id in frame[ROW_ID]}
        self.assertEqual(locations[0], locations[2000])
        self.assertEqual(len(locations), 2001)

    def test_model_inputs_exclude_sensitive_weight_and_identifiers(self):
        clean = clean_data(fixture())
        X, y, sex = model_inputs(clean)
        self.assertEqual(X.shape[1], 11)
        self.assertFalse(set([ROW_ID, "class", "sex", "education", "fnlwgt"]) & set(X.columns))
        self.assertTrue(X.index.equals(y.index) and y.index.equals(sex.index))
        self.assertTrue(pd.isna(X.loc[0, "occupation"]))  # No early imputation.
        self.assertEqual(y.loc[0], 1)
        self.assertEqual(sex.loc[0], "Female")

    def test_seed_reproducibility_and_disjoint_coverage(self):
        clean = clean_data(fixture())
        first, second = make_splits(clean), make_splits(clean)
        ids = []
        for (name, frame), (_, repeated) in zip(first.items(), second.items()):
            pd.testing.assert_frame_equal(frame, repeated)
            ids.extend(frame[ROW_ID])
            self.assertEqual(frame["class"].nunique(), 2)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(clean[ROW_ID]))

    def test_export_hashes_and_refusal_to_change_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            first = export_splits(fixture(), temp, "synthetic fixture")
            repeated = export_splits(fixture(), temp, "synthetic fixture")
            self.assertEqual(first["file_sha256"], repeated["file_sha256"])
            self.assertEqual(len(load_partition(temp, "train")), first["splits"]["train"]["n_rows"])
            changed = fixture()
            changed.loc[0, "age"] = 999
            with self.assertRaisesRegex(ValueError, "Existing split differs"):
                export_splits(changed, temp, "changed input")

    def test_corrupted_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            export_splits(fixture(), temp, "synthetic fixture")
            path = Path(temp) / "test.csv"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_partition(temp, "test")

    def test_unknown_target_is_rejected(self):
        raw = fixture()
        raw.loc[0, "class"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Target must"):
            clean_data(raw)

    def test_eda_runs_with_only_training_csv_available(self):
        from census_income.explore import run_eda
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp) / "data"
            manifest = export_splits(fixture(), data_dir, "synthetic fixture")
            (data_dir / "validation.csv").unlink()
            (data_dir / "test.csv").unlink()
            summary = run_eda(data_dir, Path(temp) / "report")
            self.assertEqual(summary["n_rows"], manifest["splits"]["train"]["n_rows"])
            self.assertEqual(len(list((Path(temp) / "report").glob("*.png"))), 7)


if __name__ == "__main__":
    unittest.main()
