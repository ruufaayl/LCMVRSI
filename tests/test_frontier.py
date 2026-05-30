import pytest

from lcmvrsi.train.frontier import run_frontier


def test_run_frontier_structure_and_state_contrast(tmp_path):
    summary = run_frontier(
        ["transformer", "ssm"],
        "mqar",
        benchmark_params={"vocab_size": 32, "num_pairs": 3, "seq_len": 16},
        d_model=16,
        n_layers=1,
        n_heads=2,
        steps=2,
        batch_size=8,
        eval_n=16,
        out_dir=tmp_path,
    )
    assert set(summary) == {"setup", "points"}
    assert len(summary["points"]) == 2
    for pt in summary["points"]:
        assert pt["model"] in {"transformer", "ssm"}
        assert 0.0 <= pt["accuracy"] <= 1.0
        assert "state_size_bytes" in pt
        assert "tokens_per_sec" in pt
        assert "param_count" in pt

    by = {p["model"]: p for p in summary["points"]}
    assert by["transformer"]["state_size_bytes"] == 0
    assert by["ssm"]["state_size_bytes"] > 0  # fixed-state model on the frontier
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2


def test_plot_frontier_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from lcmvrsi.viz.frontier_plot import plot_frontier

    def pt(model, acc, state, tps):
        return {
            "model": model,
            "accuracy": acc,
            "state_size_bytes": state,
            "tokens_per_sec": tps,
        }

    summary = {
        "setup": {"benchmark": "mqar"},
        "points": [
            pt("transformer", 1.0, 0, 1000.0),
            pt("linear_attention", 0.2, 16896, 500.0),
            pt("ssm", 0.3, 8192, 300.0),
        ],
    }
    out = plot_frontier(summary, tmp_path / "figures" / "frontier.png")
    assert out.exists()
    assert out.stat().st_size > 0
