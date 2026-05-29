import pytest

from lcmvrsi.train.sweep import run_sweep


def test_run_sweep_structure_and_state_contrast(tmp_path):
    summary = run_sweep(
        ["transformer", "linear_attention"],
        [1, 2],
        steps=2,
        seq_len=16,
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        eval_n=16,
        out_dir=tmp_path,
    )
    assert set(summary) == {"setup", "points"}
    assert len(summary["points"]) == 4  # 2 models x 2 num_pairs

    for pt in summary["points"]:
        assert pt["model"] in {"transformer", "linear_attention"}
        assert 0.0 <= pt["recall_accuracy"] <= 1.0
        assert "state_size_bytes" in pt
        assert "final_loss" in pt

    tf = [p for p in summary["points"] if p["model"] == "transformer"]
    la = [p for p in summary["points"] if p["model"] == "linear_attention"]
    # the comparison axis: transformer holds no fixed state, linear attention does
    assert all(p["state_size_bytes"] == 0 for p in tf)
    assert all(p["state_size_bytes"] > 0 for p in la)

    # raw per-run JSONs are written under runs/ (gitignored) for reproducibility
    runs = list((tmp_path / "runs").glob("*.json"))
    assert len(runs) == 4


def test_plot_recall_vs_pairs_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from lcmvrsi.viz.recall_plot import plot_recall_vs_pairs

    def pt(model, n, acc, state):
        return {
            "model": model,
            "num_pairs": n,
            "recall_accuracy": acc,
            "state_size_bytes": state,
        }

    summary = {
        "setup": {"steps": 1},
        "points": [
            pt("transformer", 1, 0.95, 0),
            pt("transformer", 8, 0.90, 0),
            pt("linear_attention", 1, 0.6, 256),
            pt("linear_attention", 8, 0.2, 256),
        ],
    }
    out = plot_recall_vs_pairs(summary, tmp_path / "figures" / "recall.png")
    assert out.exists()
    assert out.stat().st_size > 0
