import torch

from lcmvrsi.models.registry import get_model, list_models
from lcmvrsi.models.sgsm import SGSM


def test_sgsm_registered():
    assert "sgsm" in list_models()
    assert get_model("sgsm") is SGSM


def test_output_shape():
    model = SGSM(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=64)
    logits = model(torch.zeros(3, 12, dtype=torch.long))
    assert logits.shape == (3, 12, 32)


def test_state_size_includes_growing_store():
    model = SGSM(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=64)
    backbone = 2 * 2 * (8 * 8 + 8) * 4  # linear-attention backbone state
    assert model.state_size > backbone  # the external store adds (worst-case O(T)) capacity
    c = model.complexity(128)
    assert {"time", "memory"} <= set(c)


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = SGSM(vocab_size=20, d_model=16, n_layers=2, n_heads=2, max_seq_len=32)
    model.eval()
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        logits_a = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 20
        logits_b = model(x2)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


def test_forward_backward_flows_through_gate_and_memory():
    model = SGSM(vocab_size=16, d_model=16, n_layers=1, n_heads=2, max_seq_len=32)
    x = torch.randint(0, 16, (2, 8))
    model(x).log_softmax(-1).mean().backward()
    assert model.head.weight.grad is not None
    assert model.memory.qkv.weight.grad is not None  # gradient reaches the store projections


def test_gate_is_exposed_and_in_unit_interval():
    model = SGSM(vocab_size=32, d_model=16, n_layers=1, n_heads=2, max_seq_len=64)
    model(torch.randint(0, 32, (4, 12)))
    assert model.last_gate is not None
    assert model.last_gate.shape == (4, 12)
    assert model.last_gate.min() >= 0.0
    assert model.last_gate.max() <= 1.0


def test_threshold_controls_write_rate():
    torch.manual_seed(0)
    x = torch.randint(0, 32, (4, 16))
    writes_few = SGSM(vocab_size=32, d_model=16, n_layers=1, n_heads=2, init_threshold=1e6)
    writes_many = SGSM(vocab_size=32, d_model=16, n_layers=1, n_heads=2, init_threshold=-1e6)
    writes_few(x)
    writes_many(x)
    assert writes_few.last_gate.mean() < 0.1  # high threshold => almost nothing written
    assert writes_many.last_gate.mean() > 0.9  # low threshold => almost everything written
