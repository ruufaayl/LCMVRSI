import json

from lcmvrsi.viz.dashboard_data import (
    load_runs,
    load_summary,
    pivot_recall,
    summary_to_rows,
)


def test_load_summary_missing_returns_none(tmp_path):
    assert load_summary(tmp_path / "does_not_exist.json") is None


def test_load_summary_pivot_and_rows(tmp_path):
    summary = {
        "setup": {"steps": 10},
        "points": [
            {"model": "transformer", "num_pairs": 1, "recall_accuracy": 0.9},
            {"model": "transformer", "num_pairs": 2, "recall_accuracy": 0.8},
            {"model": "linear_attention", "num_pairs": 1, "recall_accuracy": 0.5},
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    loaded = load_summary(path)
    assert loaded is not None
    assert loaded["points"][0]["model"] == "transformer"

    piv = pivot_recall(loaded)
    assert piv["transformer"][1] == 0.9
    assert piv["transformer"][2] == 0.8
    assert piv["linear_attention"][1] == 0.5

    rows = summary_to_rows(loaded)
    assert len(rows) == 3


def test_load_runs_reads_all_json(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "a.json").write_text(json.dumps({"model": {"name": "transformer"}}), encoding="utf-8")
    (runs / "b.json").write_text(
        json.dumps({"model": {"name": "linear_attention"}}), encoding="utf-8"
    )
    loaded = load_runs(runs)
    assert len(loaded) == 2
    names = {r["model"]["name"] for r in loaded}
    assert names == {"transformer", "linear_attention"}


def test_load_runs_missing_dir_returns_empty(tmp_path):
    assert load_runs(tmp_path / "nope") == []
