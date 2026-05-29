import torch

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.models.registry import get_model, list_models, register_model


def test_abstract_cannot_instantiate():
    import pytest

    with pytest.raises(TypeError):
        SequenceModel()  # abstract methods not implemented


def test_concrete_model_registers_and_runs():
    @register_model("dummy")
    class Dummy(SequenceModel):
        def __init__(self, vocab_size: int = 10, d_model: int = 4) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(vocab_size, d_model)
            self.head = torch.nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.emb(input_ids))

        @property
        def state_size(self) -> int:
            return 0

        def complexity(self, seq_len: int) -> dict[str, str]:
            return {"time": "O(T)", "memory": "O(T)"}

    assert "dummy" in list_models()
    model = get_model("dummy")(vocab_size=10, d_model=4)
    logits = model(torch.zeros(2, 5, dtype=torch.long))
    assert logits.shape == (2, 5, 10)
    assert model.state_size == 0
    assert model.complexity(5)["time"] == "O(T)"
