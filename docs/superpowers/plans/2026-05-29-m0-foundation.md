# M0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the LCMVRSI repository skeleton — an installable, linted, tested Python package with the core `SequenceModel`/`Benchmark` interfaces, reproducibility/config/profiling utilities, CI, a LaTeX→PDF paper pipeline, and the docs structure — then push it to the remote.

**Architecture:** A `src/`-layout package `lcmvrsi` managed by `uv`. Small, single-responsibility modules behind abstract interfaces and a generic registry, each covered by fast CPU-only pytest tests. CI runs ruff + pytest; a separate workflow builds the paper PDF via `xu-cheng/latex-action`.

**Tech Stack:** Python 3.12, uv, PyTorch (CPU for tests), pydantic, PyYAML, einops, pytest, ruff, LaTeX (texlive via GitHub Actions), GitHub Actions.

---

## File Structure (created by this plan)

```
LCMVRSI/
├── pyproject.toml                      # project metadata, deps, ruff/pytest config
├── .gitignore
├── LICENSE                             # MIT
├── README.md                           # honest framing + quickstart
├── .github/workflows/ci.yml            # ruff + pytest
├── .github/workflows/paper.yml         # LaTeX -> PDF
├── configs/base.yaml                   # example experiment config
├── src/lcmvrsi/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py                     # set_seed()
│   │   ├── config.py                   # pydantic schema + load_config()
│   │   ├── profiling.py                # count_parameters()
│   │   └── registry.py                 # generic Registry[T]
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                     # SequenceModel ABC
│   │   └── registry.py                 # register_model/get_model/list_models
│   └── benchmarks/
│       ├── __init__.py
│       ├── base.py                     # Benchmark ABC
│       └── registry.py                 # register_benchmark/get_benchmark/list_benchmarks
├── tests/
│   ├── test_seed.py
│   ├── test_config.py
│   ├── test_profiling.py
│   ├── test_registry.py
│   ├── test_models.py
│   └── test_benchmarks.py
├── paper/
│   ├── main.tex
│   ├── sections/01-introduction.tex
│   ├── sections/02-problem.tex
│   ├── refs.bib
│   └── Makefile
└── docs/
    ├── knowledge-map/README.md
    ├── architecture-review/README.md
    └── problem/README.md
```

---

## Task 1: Project scaffolding + git init

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`, `src/lcmvrsi/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "lcmvrsi"
version = "0.0.1"
description = "Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence: a research scaffold."
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Rufayl Waseem" }]
license = { text = "MIT" }
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "einops>=0.7",
    "pyyaml>=6.0",
    "pydantic>=2.6",
]

[project.optional-dependencies]
viz = [
    "streamlit>=1.33",
    "matplotlib>=3.8",
    "pandas>=2.2",
    "tensorboard>=2.16",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lcmvrsi"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
*.egg-info/
.pytest_cache/
.ruff_cache/

# Run logs (figures are committed; raw runs are not)
results/runs/
runs/
*.log

# LaTeX build artifacts
paper/*.pdf
paper/*.aux
paper/*.bbl
paper/*.blg
paper/*.fls
paper/*.fdb_latexmk
paper/*.out
paper/*.toc
paper/_build/

# OS / editor
.DS_Store
Thumbs.db
.idea/
.vscode/
```

- [ ] **Step 3: Write `LICENSE` (MIT)**

```text
MIT License

Copyright (c) 2026 Rufayl Waseem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Write `README.md`**

````markdown
# LCMVRSI

**Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence** — a research
scaffold for studying the **recall–memory tradeoff** in subquadratic sequence models.

> **Honesty discipline.** Every non-trivial claim in this repo and the paper is tagged
> **[PROVEN]**, **[EMPIRICAL]**, or **[CONJECTURE]**. No fabricated theorems, citations,
> or results. References enter `paper/refs.bib` only after verification.

## What this is
A runnable, tested comparison of sequence-model architectures (transformer, linear
attention, SSM/Mamba, RWKV, …) on synthetic associative-recall tasks (MQAR), plus an
honest attempt at a novel result: a lower bound linking recall capacity to recurrent
state size, and an entropy-gated sparse-memory mechanism. See
[`docs/superpowers/specs/2026-05-29-lcmvrsi-design.md`](docs/superpowers/specs/2026-05-29-lcmvrsi-design.md).

## Quickstart
```bash
# install uv: https://docs.astral.sh/uv/
uv sync --group dev      # create .venv and install deps + dev tools
uv run pytest            # run the test suite
uv run ruff check .      # lint
```
GPU note: `uv sync` installs the CPU build of PyTorch by default. For your CUDA GPU,
install the matching CUDA wheel per https://pytorch.org/get-started/locally/.

## Layout
- `src/lcmvrsi/` — package: `models/`, `benchmarks/`, `train/`, `utils/`
- `configs/` — YAML experiment configs (tiny-by-default; scale-up provided later)
- `experiments/`, `dashboard/` — runner CLI and Streamlit dashboard (later milestones)
- `paper/` — LaTeX sources; PDF built by CI
- `docs/` — knowledge map, architecture review, problem formalization

## Status
Milestone **M0 (foundation)**. Roadmap M0→M5 in the design spec.
````

- [ ] **Step 5: Write `src/lcmvrsi/__init__.py`**

```python
"""LCMVRSI: Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence."""

__version__ = "0.0.1"
```

- [ ] **Step 6: Initialize git and install deps**

Run:
```bash
git init
uv sync --group dev
```
Expected: `.git/` created; uv creates `.venv/` and `uv.lock`, installs torch/pydantic/etc.
(First run downloads PyTorch — may take a few minutes.)

- [ ] **Step 7: Verify the package imports**

Run: `uv run python -c "import lcmvrsi; print(lcmvrsi.__version__)"`
Expected: prints `0.0.1`

- [ ] **Step 8: Commit** (includes the already-written design spec)

```bash
git add pyproject.toml uv.lock .gitignore LICENSE README.md src/lcmvrsi/__init__.py docs/ claude.md
git commit -m "chore: scaffold lcmvrsi package, tooling, and design spec (M0)"
```

---

## Task 2: Reproducibility — `set_seed`

**Files:**
- Create: `src/lcmvrsi/utils/__init__.py`, `src/lcmvrsi/utils/seed.py`
- Test: `tests/test_seed.py`

- [ ] **Step 1: Write the failing test**

`tests/test_seed.py`:
```python
import torch

from lcmvrsi.utils.seed import set_seed


def test_same_seed_reproduces_torch_draw():
    set_seed(123)
    a = torch.rand(8)
    set_seed(123)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.rand(8)
    set_seed(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.utils.seed'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/utils/__init__.py`:
```python
```
(empty file)

`src/lcmvrsi/utils/seed.py`:
```python
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/utils/__init__.py src/lcmvrsi/utils/seed.py tests/test_seed.py
git commit -m "feat(utils): add set_seed for reproducibility"
```

---

## Task 3: Config system — pydantic schema + YAML loader

**Files:**
- Create: `src/lcmvrsi/utils/config.py`, `configs/base.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path

import pydantic
import pytest

from lcmvrsi.utils.config import ExperimentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_committed_base_config():
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.name
    assert cfg.benchmark.name
    assert cfg.train.device in {"cpu", "cuda"}


def test_missing_required_fields_raise():
    with pytest.raises(pydantic.ValidationError):
        ExperimentConfig.model_validate({"seed": 0})  # no model/benchmark


def test_load_from_dict_roundtrip(tmp_path):
    import yaml

    data = {
        "seed": 7,
        "model": {"name": "transformer", "params": {"d_model": 32}},
        "benchmark": {"name": "mqar", "params": {"seq_len": 64}},
        "train": {"steps": 10, "batch_size": 4, "lr": 0.01, "device": "cpu"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 7
    assert cfg.model.params["d_model"] == 32
    assert cfg.train.steps == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.utils.config'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/utils/config.py`:
```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    steps: int = 1000
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"


class ExperimentConfig(BaseModel):
    seed: int = 0
    model: ModelConfig
    benchmark: BenchmarkConfig
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
```

`configs/base.yaml`:
```yaml
seed: 0
model:
  name: transformer
  params:
    d_model: 64
    n_layers: 2
    n_heads: 2
benchmark:
  name: mqar
  params:
    vocab_size: 64
    seq_len: 128
    num_pairs: 8
train:
  steps: 1000
  batch_size: 32
  lr: 0.001
  device: cpu
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/utils/config.py configs/base.yaml tests/test_config.py
git commit -m "feat(utils): add pydantic experiment config + YAML loader"
```

---

## Task 4: Profiling — `count_parameters`

**Files:**
- Create: `src/lcmvrsi/utils/profiling.py`
- Test: `tests/test_profiling.py`

- [ ] **Step 1: Write the failing test**

`tests/test_profiling.py`:
```python
import torch.nn as nn

from lcmvrsi.utils.profiling import count_parameters


def test_counts_linear_params():
    # Linear(10, 10): weight 10*10 + bias 10 = 110
    assert count_parameters(nn.Linear(10, 10)) == 110


def test_trainable_only_excludes_frozen():
    layer = nn.Linear(10, 10)
    for p in layer.parameters():
        p.requires_grad_(False)
    assert count_parameters(layer, trainable_only=True) == 0
    assert count_parameters(layer, trainable_only=False) == 110
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_profiling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.utils.profiling'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/utils/profiling.py`:
```python
from __future__ import annotations

import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters in a module."""
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_profiling.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/utils/profiling.py tests/test_profiling.py
git commit -m "feat(utils): add count_parameters profiling helper"
```

---

## Task 5: Generic registry

**Files:**
- Create: `src/lcmvrsi/utils/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

`tests/test_registry.py`:
```python
import pytest

from lcmvrsi.utils.registry import Registry


def test_register_get_and_names():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A:
        pass

    assert reg.get("a") is A
    assert reg.names() == ["a"]


def test_duplicate_registration_raises():
    reg: Registry[object] = Registry("widget")

    @reg.register("a")
    class A:
        pass

    with pytest.raises(ValueError):

        @reg.register("a")
        class B:
            pass


def test_unknown_name_raises_keyerror():
    reg: Registry[object] = Registry("widget")
    with pytest.raises(KeyError):
        reg.get("missing")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.utils.registry'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/utils/registry.py`:
```python
from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> class registry used for models and benchmarks."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(f"Unknown {self._kind} '{name}'. Registered: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/utils/registry.py tests/test_registry.py
git commit -m "feat(utils): add generic Registry"
```

---

## Task 6: Model interface — `SequenceModel` ABC + model registry

**Files:**
- Create: `src/lcmvrsi/models/__init__.py`, `src/lcmvrsi/models/base.py`, `src/lcmvrsi/models/registry.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

`tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.models.base'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/models/__init__.py`:
```python
```
(empty file)

`src/lcmvrsi/models/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SequenceModel(nn.Module, ABC):
    """Common interface for sequence models compared in LCMVRSI.

    Subclasses must implement `forward` and self-report their memory/compute story via
    `state_size` and `complexity`, so models are comparable on the recall-memory frontier.
    """

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids (B, T) to logits (B, T, vocab_size)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Bytes of recurrent state per sequence at inference (0 if O(T) cache / not applicable)."""
        raise NotImplementedError

    @abstractmethod
    def complexity(self, seq_len: int) -> dict[str, str]:
        """Big-O annotations, e.g. {'time': 'O(T^2)', 'memory': 'O(T)'}."""
        raise NotImplementedError
```

`src/lcmvrsi/models/registry.py`:
```python
from __future__ import annotations

from lcmvrsi.models.base import SequenceModel
from lcmvrsi.utils.registry import Registry

MODELS: Registry[SequenceModel] = Registry("model")
register_model = MODELS.register
get_model = MODELS.get
list_models = MODELS.names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/models/__init__.py src/lcmvrsi/models/base.py src/lcmvrsi/models/registry.py tests/test_models.py
git commit -m "feat(models): add SequenceModel interface and model registry"
```

---

## Task 7: Benchmark interface — `Benchmark` ABC + benchmark registry

**Files:**
- Create: `src/lcmvrsi/benchmarks/__init__.py`, `src/lcmvrsi/benchmarks/base.py`, `src/lcmvrsi/benchmarks/registry.py`
- Test: `tests/test_benchmarks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_benchmarks.py`:
```python
import torch

from lcmvrsi.benchmarks.base import Benchmark
from lcmvrsi.benchmarks.registry import get_benchmark, list_benchmarks, register_benchmark
from lcmvrsi.models.base import SequenceModel


def test_abstract_cannot_instantiate():
    import pytest

    with pytest.raises(TypeError):
        Benchmark()


def test_concrete_benchmark_registers_generates_evaluates():
    @register_benchmark("echo")
    class Echo(Benchmark):
        def generate(self, n: int, seq_len: int, seed: int):
            g = torch.Generator().manual_seed(seed)
            x = torch.randint(0, 5, (n, seq_len), generator=g)
            return x, x.clone()

        def evaluate(self, model: SequenceModel, n: int, seq_len: int, seed: int):
            x, y = self.generate(n, seq_len, seed)
            logits = model(x)
            acc = (logits.argmax(-1) == y).float().mean().item()
            return {"accuracy": acc}

    assert "echo" in list_benchmarks()
    bench = get_benchmark("echo")()
    x, y = bench.generate(n=3, seq_len=4, seed=0)
    assert x.shape == (3, 4)
    assert torch.equal(x, y)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lcmvrsi.benchmarks.base'`

- [ ] **Step 3: Write minimal implementation**

`src/lcmvrsi/benchmarks/__init__.py`:
```python
```
(empty file)

`src/lcmvrsi/benchmarks/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lcmvrsi.models.base import SequenceModel


class Benchmark(ABC):
    """Common interface for synthetic sequence benchmarks."""

    @abstractmethod
    def generate(self, n: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs (n, T) long, targets (n, T) long) for a split."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, model: SequenceModel, n: int, seq_len: int, seed: int
    ) -> dict[str, float]:
        """Return a metrics dict, e.g. {'accuracy': 0.9}."""
        raise NotImplementedError
```

`src/lcmvrsi/benchmarks/registry.py`:
```python
from __future__ import annotations

from lcmvrsi.benchmarks.base import Benchmark
from lcmvrsi.utils.registry import Registry

BENCHMARKS: Registry[Benchmark] = Registry("benchmark")
register_benchmark = BENCHMARKS.register
get_benchmark = BENCHMARKS.get
list_benchmarks = BENCHMARKS.names
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmarks.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/lcmvrsi/benchmarks/__init__.py src/lcmvrsi/benchmarks/base.py src/lcmvrsi/benchmarks/registry.py tests/test_benchmarks.py
git commit -m "feat(benchmarks): add Benchmark interface and benchmark registry"
```

---

## Task 8: CI workflow (ruff + pytest)

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Run the full suite + lint locally first**

Run:
```bash
uv run ruff check .
uv run pytest
```
Expected: ruff reports "All checks passed!"; pytest shows all tests passing (13 passed).
If ruff flags issues, fix them, then re-run until clean before continuing.

- [ ] **Step 2: Write the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - name: Sync dependencies
        run: uv sync --group dev
      - name: Lint (ruff)
        run: uv run ruff check .
      - name: Tests (pytest)
        run: uv run pytest
```

- [ ] **Step 3: Validate the YAML parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint + test on push/PR via uv"
```

---

## Task 9: Paper skeleton + PDF build workflow

**Files:**
- Create: `paper/main.tex`, `paper/sections/01-introduction.tex`, `paper/sections/02-problem.tex`, `paper/refs.bib`, `paper/Makefile`, `.github/workflows/paper.yml`

- [ ] **Step 1: Write `paper/refs.bib`** (verified reference only)

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is All you Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
```

- [ ] **Step 2: Write `paper/sections/01-introduction.tex`**

```latex
\section{Introduction}
This is a working-draft skeleton, built automatically by CI. The transformer
architecture~\citep{vaswani2017attention} established attention as a core sequence-modeling
primitive at the cost of quadratic time and memory in sequence length. This project studies
the \emph{recall--memory tradeoff} exhibited by subquadratic alternatives. Full content is
populated in later milestones.
```

- [ ] **Step 3: Write `paper/sections/02-problem.tex`**

```latex
\section{Problem Statement (Beachhead)}
\textbf{[CONJECTURE]} We target a lower bound linking associative-recall capacity to the size
of a model's recurrent state, alongside an entropy-gated sparse-memory mechanism intended to
push the empirical recall--memory Pareto frontier. The precise theorem statement and its
prior-art positioning are developed in milestone M4.
```

- [ ] **Step 4: Write `paper/main.tex`**

```latex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage[numbers]{natbib}
\usepackage{hyperref}

\newtheorem{theorem}{Theorem}
\newtheorem{conjecture}{Conjecture}

\title{Long-Context Memory, Verifiable Reasoning, and Scalable Intelligence:\\
A Research Scaffold (Working Draft)}
\author{Rufayl Waseem}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
Working draft. Claims are tagged PROVEN / EMPIRICAL / CONJECTURE. This abstract is a
placeholder, populated in milestone M5.
\end{abstract}

\input{sections/01-introduction}
\input{sections/02-problem}

\bibliographystyle{plainnat}
\bibliography{refs}

\end{document}
```

- [ ] **Step 5: Write `paper/Makefile`** (optional local build)

```makefile
main.pdf: main.tex sections/*.tex refs.bib
	latexmk -pdf main.tex

clean:
	latexmk -C
```

- [ ] **Step 6: Write `.github/workflows/paper.yml`**

```yaml
name: Paper

on:
  push:
    paths:
      - "paper/**"
      - ".github/workflows/paper.yml"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build PDF
        uses: xu-cheng/latex-action@v3
        with:
          working_directory: paper
          root_file: main.tex
      - name: Upload PDF artifact
        uses: actions/upload-artifact@v4
        with:
          name: lcmvrsi-paper
          path: paper/main.pdf
```

- [ ] **Step 7: Validate both YAML and LaTeX structure parse**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/paper.yml'))"`
Expected: no output, exit 0.
Note: the PDF itself is built/verified by CI (no local LaTeX assumed). If `tectonic` or
`latexmk` is available locally, `cd paper && latexmk -pdf main.tex` should also succeed.

- [ ] **Step 8: Commit**

```bash
git add paper/ .github/workflows/paper.yml
git commit -m "docs(paper): add LaTeX skeleton and CI PDF build"
```

---

## Task 10: Docs structure

**Files:**
- Create: `docs/knowledge-map/README.md`, `docs/architecture-review/README.md`, `docs/problem/README.md`

- [ ] **Step 1: Write `docs/knowledge-map/README.md`**

```markdown
# Knowledge Map

Prerequisite knowledge scoped to the beachhead (recall vs. memory). Populated in **M1**:
linear algebra (low-rank, rank of attention maps), information theory (entropy, mutual
information, communication complexity), and associative-memory background. Each claim tagged
[PROVEN] / [EMPIRICAL] / [CONJECTURE].
```

- [ ] **Step 2: Write `docs/architecture-review/README.md`**

```markdown
# Architecture Review

Mechanism + complexity + recurrent-state derivations for transformer, linear attention,
SSM/Mamba, and RWKV (the architectures relevant to the recall-memory tradeoff). Populated in
**M1**, with correct big-O time/memory and state-size analysis.
```

- [ ] **Step 3: Write `docs/problem/README.md`**

```markdown
# Problem Formalization

The precise MQAR setup, the recall-state tradeoff, the exact open question, our hypothesis,
the theorem statement we attempt, and its falsification plan. Includes a citation-verified
literature review. Populated in **M1**.
```

- [ ] **Step 4: Commit**

```bash
git add docs/knowledge-map/README.md docs/architecture-review/README.md docs/problem/README.md
git commit -m "docs: add knowledge-map, architecture-review, and problem stubs"
```

---

## Task 11: Push to remote (safe)

**Files:** none (git operations)

- [ ] **Step 1: Add remote and inspect its state**

Run:
```bash
git remote add origin https://github.com/ruufaayl/LCMVRSI.git
git ls-remote origin
```
Expected: either empty output (fresh repo) or a list of refs (repo already has commits).
If `ls-remote` fails with auth error, check `gh auth status`; if not logged in, the user must
run `gh auth login` (or provide a PAT) before pushing.

- [ ] **Step 2: Confirm with the user before the first push**

Report the remote state and the local commit log (`git log --oneline`). Get explicit
confirmation to push. **Never force-push.**

- [ ] **Step 3a: If remote is empty — push**

```bash
git branch -M main
git push -u origin main
```

- [ ] **Step 3b: If remote already has commits — integrate first, then push**

```bash
git fetch origin
git rebase origin/main      # resolve any conflicts, keeping both histories
git push -u origin main
```
Expected: push succeeds; `https://github.com/ruufaayl/LCMVRSI` shows the files and CI starts.

- [ ] **Step 4: Verify CI triggered**

Run: `gh run list --limit 3`
Expected: a CI run (and possibly a Paper run) listed as queued/in-progress.

---

## Self-Review

**1. Spec coverage (M0 rows of the roadmap):**
- uv project + deps → Task 1 ✓
- package skeleton with ABCs → Tasks 6, 7 ✓
- utils (seed/config/profiling) → Tasks 2, 3, 4 (+ generic registry Task 5) ✓
- `configs/base.yaml` → Task 3 ✓
- CI + passing test → Task 8 ✓
- `paper/` LaTeX skeleton + PDF workflow → Task 9 ✓
- `docs/` structure → Task 10 ✓
- MIT license, README → Task 1 ✓
- git init, first commit, push (check remote, never force-push, confirm first) → Tasks 1, 11 ✓
- Honesty tags discipline surfaced in README + paper + docs stubs ✓
(M1 research-writing is intentionally a separate plan.)

**2. Placeholder scan:** No "TBD/TODO/implement later". The empty `__init__.py` files are intentional and labeled. The paper/docs say "populated in M1/M4/M5" — these describe future milestones, not missing plan content.

**3. Type consistency:** `set_seed`, `count_parameters`, `Registry(register/get/names)`,
`ExperimentConfig`/`ModelConfig`/`BenchmarkConfig`/`TrainConfig`, `load_config`,
`SequenceModel(forward/state_size/complexity)`, `register_model/get_model/list_models`,
`Benchmark(generate/evaluate)`, `register_benchmark/get_benchmark/list_benchmarks` — names
used consistently across tasks and tests. Model registry wraps the generic `Registry` from
Task 5; benchmark registry likewise. ✓
