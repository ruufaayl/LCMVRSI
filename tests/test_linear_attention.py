import torch

from lcmvrsi.models.linear_attention import LinearAttention
from lcmvrsi.models.registry import get_model, list_models


def test_linear_attention_registered():
    assert "linear_attention" in list_models()
    assert get_model("linear_attention") is LinearAttention


def test_output_shape():
    model = LinearAttention(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=64)
    logits = model(torch.zeros(3, 12, dtype=torch.long))
    assert logits.shape == (3, 12, 32)


def test_fixed_state_size_independent_of_seq_len():
    model = LinearAttention(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=64)
    # fixed recurrent state: n_layers * n_heads * (d_head^2 + d_head) floats * 4 bytes
    d_head = 16 // 2
    expected = 2 * 2 * (d_head * d_head + d_head) * 4
    assert model.state_size == expected
    assert model.state_size > 0  # unlike the transformer, the state is a real bottleneck
    c = model.complexity(128)
    assert {"time", "memory"} <= set(c)
    assert "T^2" not in c["time"]  # subquadratic in sequence length


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = LinearAttention(vocab_size=20, d_model=16, n_layers=2, n_heads=2, max_seq_len=32)
    model.eval()
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        logits_a = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 20
        logits_b = model(x2)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


def test_forward_backward_runs():
    model = LinearAttention(vocab_size=16, d_model=16, n_layers=1, n_heads=2, max_seq_len=32)
    x = torch.randint(0, 16, (2, 8))
    loss = model(x).log_softmax(-1).mean()
    loss.backward()
    assert model.head.weight.grad is not None
