"""`upsert_csv`: the guard that exists because the same file was destroyed twice.

`results/e5_depth.csv` lost rows to a narrow re-run in an earlier session and
again on 2026-08-31 (63 of 66 rows). Both times the mechanism was identical: a
sweep that writes its whole table at once, run over a subset, calling
`to_csv`. These tests pin the behaviour every such sweep now shares.
"""
import pandas as pd
import pytest

from src.results import upsert_csv

KEYS = ["dataset", "method", "seed"]


def _frame(rows):
    return pd.DataFrame([dict(zip(KEYS + ["value"], r)) for r in rows])


def test_narrow_rerun_keeps_rows_it_did_not_touch(tmp_path):
    path = tmp_path / "sweep.csv"
    upsert_csv(path, _frame([("a", "map", 0, 1.0), ("a", "map", 1, 2.0),
                             ("b", "mcd", 0, 3.0)]), KEYS)
    upsert_csv(path, _frame([("b", "mcd", 0, 30.0)]), KEYS)

    out = pd.read_csv(path)
    assert len(out) == 3, "a narrow re-run must not shrink the file"
    assert out[(out.dataset == "b")].value.iloc[0] == 30.0, "its own key must be replaced"
    assert sorted(out[out.dataset == "a"].value) == [1.0, 2.0], "other keys must survive"


def test_rerunning_the_same_key_replaces_rather_than_duplicates(tmp_path):
    path = tmp_path / "sweep.csv"
    rows = _frame([("a", "map", 0, 1.0)])
    upsert_csv(path, rows, KEYS)
    upsert_csv(path, _frame([("a", "map", 0, 9.0)]), KEYS)

    out = pd.read_csv(path)
    assert len(out) == 1 and out.value.iloc[0] == 9.0


def test_missing_key_column_is_an_error_not_a_silent_overwrite(tmp_path):
    path = tmp_path / "sweep.csv"
    upsert_csv(path, _frame([("a", "map", 0, 1.0)]), KEYS)
    with pytest.raises(ValueError):
        upsert_csv(path, pd.DataFrame([{"dataset": "a", "value": 2.0}]), KEYS)
    assert len(pd.read_csv(path)) == 1, "the existing file must be untouched on error"
