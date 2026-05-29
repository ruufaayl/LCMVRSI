from lcmvrsi.analysis import complexity_table, render_markdown_table


def test_complexity_table_reflects_live_models():
    rows = complexity_table(["transformer", "linear_attention"], seq_len=128)
    by = {r["model"]: r for r in rows}

    # quadratic attention vs subquadratic linear attention (read straight from the models)
    assert "T^2" in by["transformer"]["time"]
    assert "T^2" not in by["linear_attention"]["time"]

    # the decisive column: transformer keeps no fixed state (growing KV cache),
    # linear attention keeps a fixed recurrent state -- the rank-limited bottleneck
    assert by["transformer"]["state_size_bytes"] == 0
    assert by["linear_attention"]["state_size_bytes"] > 0

    assert by["transformer"]["param_count"] > 0
    assert by["linear_attention"]["param_count"] > 0


def test_complexity_table_defaults_to_all_registered_models():
    rows = complexity_table(seq_len=64)
    names = {r["model"] for r in rows}
    assert {"transformer", "linear_attention"} <= names


def test_render_markdown_table_has_header_and_rows():
    rows = complexity_table(["transformer", "linear_attention"], seq_len=128)
    md = render_markdown_table(rows)
    lines = md.splitlines()
    assert lines[0].startswith("| Model |")
    assert set("|-") >= set(lines[1].replace(" ", ""))  # separator row is only | and -
    assert any("transformer" in line for line in lines)
    assert any("linear_attention" in line for line in lines)
