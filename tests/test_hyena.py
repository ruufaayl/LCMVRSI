import torch

from lcmvrsi.models.hyena import Hyena
from lcmvrsi.models.registry import get_model, list_models


def test_hyena_registered():
    assert "hyena" in list_models()
    assert get_model("hyena") is Hyena


def test_output_shape():
    model = Hyena(vocab_size=32, d_model=16, n_layers=2, max_seq_len=64)
    logits = model(torch.zeros(3, 12, dtype=torch.long))
    assert logits.shape == (3, 12, 32)


def test_no_fixed_state_and_subquadratic_complexity():
    model = Hyena(vocab_size=32, d_model=16, n_layers=2, max_seq_len=64)
    # a global convolution keeps no fixed recurrent state (like attention); history is O(T)
    assert model.state_size == 0
    c = model.complexity(128)
    assert {"time", "memory"} <= set(c)
    assert "T^2" not in c["time"]  # subquadratic: FFT long conv is O(T log T)
    assert "log" in c["time"]


def test_causal_no_future_leak():
    torch.manual_seed(0)
    model = Hyena(vocab_size=20, d_model=16, n_layers=2, max_seq_len=32)
    model.eval()
    x = torch.randint(0, 20, (2, 10))
    with torch.no_grad():
        logits_a = model(x)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 20
        logits_b = model(x2)
    # FFT convolution couples everything at round-off level; a real leak would be O(1).
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-4)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1], atol=1e-4)


def test_forward_backward_runs():
    model = Hyena(vocab_size=16, d_model=16, n_layers=1, max_seq_len=32)
    x = torch.randint(0, 16, (2, 8))
    loss = model(x).log_softmax(-1).mean()
    loss.backward()
    assert model.head.weight.grad is not None
