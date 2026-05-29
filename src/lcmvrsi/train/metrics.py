from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch


def git_commit() -> str | None:
    """Return the current git commit hash, or None if unavailable.

    Recorded in results so every metric is traceable to the exact code that produced it.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def environment() -> dict[str, Any]:
    """Capture the software environment for reproducibility of recorded results."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
    }


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak-memory tracking (no-op on CPU, where peak RSS is not portable)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int | None:
    """Peak CUDA bytes allocated since the last reset; None on CPU.

    On CPU we deliberately return None rather than a noisy, platform-dependent RSS figure;
    the meaningful, deterministic memory quantity for the recall-memory frontier is the
    model's self-reported ``state_size`` (recorded separately), not process RSS.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None
