from __future__ import annotations

from lcmvrsi.benchmarks.base import Benchmark
from lcmvrsi.utils.registry import Registry

BENCHMARKS: Registry[Benchmark] = Registry("benchmark")
register_benchmark = BENCHMARKS.register
get_benchmark = BENCHMARKS.get
list_benchmarks = BENCHMARKS.names
