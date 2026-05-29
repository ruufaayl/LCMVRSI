import torch

from lcmvrsi.models.registry import get_model, list_models
from lcmvrsi.models.transformer import Transformer


def test_transformer_registered():
    assert "transformer" in list_models()
    assert get_model("transformer") is Transformer


def test_output_shape():
    model = Transformer(vocab_size=32, d_model=16, n_layers=2, n_heads=2, max_seq_len=64)
    logits = model(torch.zeros(3, 12, dtype=torch.long))
    assert logits.shape == (3, 12, 32)


def test_state_size_zero_and_complexity():
    model = Transformer(vocab_size=32, d_model=16, n_layers=1, n_heads=2, max_seq_len=64)
    assert model.state_size == 0  # KV cache grows with T; no fixed recurrent state
    c = model.complexity(128)
    assert {"time", "memory"} <= set(c)
    assert "T^2" in c["time"]


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = Transformer(vocab_size=20, d_model=16, n_layers=2, n_heads=2, max_seq_len=32)
    model.eval()
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        logits_a = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 20  # change ONLY the last token
        logits_b = model(x2)
    # causal => positions before the last cannot see the changed future token
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    # sanity: the change did affect the last position's logits
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-5)


def test_forward_backward_runs():
    model = Transformer(vocab_size=16, d_model=16, n_layers=1, n_heads=2, max_seq_len=32)
    x = torch.randint(0, 16, (2, 8))
    loss = model(x).log_softmax(-1).mean()
    loss.backward()
    assert model.head.weight.grad is not None


def test_seq_len_over_max_raises():
    model = Transformer(vocab_size=16, d_model=16, n_layers=1, n_heads=2, max_seq_len=8)
    try:
        model(torch.zeros(1, 16, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("expected ValueError when seq_len exceeds max_seq_len")
