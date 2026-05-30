import pytest


def test_plot_h2_comparison_writes_figure(tmp_path):
    pytest.importorskip("matplotlib")
    from lcmvrsi.viz.h2_plot import plot_h2_comparison

    def pt(label, acc, worst, store, wf):
        return {
            "label": label,
            "accuracy": acc,
            "state_size_bytes_worstcase": worst,
            "realized_store_bytes": store,
            "write_fraction": wf,
        }

    summary = {
        "setup": {"benchmark": "structured_recall"},
        "points": [
            pt("transformer", 1.0, 0, None, None),
            pt("linear_attention", 0.3, 16896, None, None),
            pt("sgsm", 0.9, 49152, 6144, 0.25),
            pt("sgsm_no_gate", 0.95, 49152, 49152, 1.0),
        ],
    }
    out = plot_h2_comparison(summary, tmp_path / "figures" / "h2.png")
    assert out.exists()
    assert out.stat().st_size > 0
